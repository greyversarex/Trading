"""CLI для запуска валидации детектора структур.

Запуск:
    python scripts/validate.py                # на исторических данных Binance
    python scripts/validate.py --synthetic    # на размеченной синтетике + ML

Без флага скрипт прогоняет детекцию по тестовым символам/таймфреймам из
``CONFIG.validation`` и печатает метрики (precision/recall/F1, перерисовка).
С флагом ``--synthetic`` прогоняет валидацию на ``data/synthetic_dataset.json``
с ML-фильтром и печатает true/false positive rate (raw vs ML) и распределение
ml_score. Результаты сохраняются в ``data/validation_results.json`` либо
``data/validation_synthetic.json``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

# гарантируем, что корень проекта в sys.path при запуске как отдельный скрипт
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.config import CONFIG  # noqa: E402
from src.backend.structure_extractor import StructureExtractor  # noqa: E402
from src.backend.validation.runner import ValidationRunner  # noqa: E402

RESULTS_PATH = os.path.join("data", "validation_results.json")
SYNTHETIC_RESULTS_PATH = os.path.join("data", "validation_synthetic.json")
SYNTHETIC_DATASET_PATH = os.path.join("data", "synthetic_dataset.json")


def run_synthetic() -> int:
    """Валидация на синтетике с ML-фильтром."""
    from src.backend.ml.pipeline import MLPipeline  # noqa: E402

    if not os.path.exists(SYNTHETIC_DATASET_PATH):
        print(
            f"[error] датасет не найден: {SYNTHETIC_DATASET_PATH}\n"
            "Сначала выполните: python scripts/generate_synthetic.py"
        )
        return 1

    with open(SYNTHETIC_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    detector = StructureExtractor()
    runner = ValidationRunner()
    pipeline = MLPipeline(model_path=CONFIG.ml.ml_model_path)

    print("Синтетическая валидация...")
    print(f"Образцов: {len(dataset)}, модель ML: {'есть' if pipeline.model_exists else 'нет'}")
    print(f"Порог ml_score: {CONFIG.ml.ml_score_threshold}")
    print("-" * 64)

    report = runner.run_synthetic_validation(detector, dataset, ml_pipeline=pipeline)

    print(f"\nПаттернов: {report['n_pattern']}, шума: {report['n_noise']}")
    for mode in ("raw", "ml"):
        m = report[mode]
        title = "Только геометрия" if mode == "raw" else "Геометрия + ML-фильтр"
        print(f"\n[{title}]")
        print(f"  true_positive_rate (recall паттернов): {m['true_positive_rate']:.2%}")
        print(f"  false_positive_rate (срабатывания на шуме): {m['false_positive_rate']:.2%}")
        print("  По типам (precision / recall / f1):")
        for ptype, s in sorted(m["by_pattern"].items()):
            print(
                f"    {ptype:<18} P={s['precision']:.2%} R={s['recall']:.2%} "
                f"F1={s['f1']:.2%} (tp={s['tp']} fp={s['fp']} fn={s['fn']})"
            )

    dist = report["ml_score_distribution"]
    print(f"\nРаспределение ml_score (всего оценено: {dist['n_scored']}):")
    for b, c in zip(dist["bins"], dist["counts"]):
        bar = "#" * min(40, c)
        print(f"  {b}  {c:>4}  {bar}")

    os.makedirs("data", exist_ok=True)
    with open(SYNTHETIC_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены в {SYNTHETIC_RESULTS_PATH}")
    return 0


async def main() -> None:
    detector = StructureExtractor()
    runner = ValidationRunner()

    print("Запуск валидации...")
    print(f"Символы:    {CONFIG.validation.test_symbols}")
    print(f"Таймфреймы: {CONFIG.validation.test_timeframes}")
    print(f"История:    {CONFIG.validation.max_history_limit} свечей, окно {runner.window}")
    print("-" * 64)

    report = await runner.run_multi_symbol(detector)

    summary = report["summary"]
    print(
        f"\nПрогонов: {summary['num_runs']}, с данными: {summary['runs_with_data']}, "
        f"событий: {summary['total_events']}"
    )
    print(
        f"Общие метрики: precision={summary['precision']:.2%} "
        f"recall={summary['recall']:.2%} f1={summary['f1']:.2%} "
        f"(tp={summary['tp']} fp={summary['fp']} fn={summary['fn']})"
    )
    if summary["tp"] + summary["fn"] == 0:
        print("  (разметка отсутствует — precision/recall предварительные)")

    print("\nПо типам паттернов:")
    if report["by_pattern"]:
        for ptype, stats in sorted(report["by_pattern"].items()):
            total = stats["total"]
            confirmed = stats["confirmed"]
            ratio = confirmed / total if total else 0.0
            print(
                f"  {ptype:<22} всего={total:<6} подтв.={confirmed:<6} "
                f"доля подтв.={ratio:.2%}"
            )
    else:
        print("  (паттерны не обнаружены)")

    print("\nПо таймфреймам:")
    if report["by_timeframe"]:
        for tf, stats in sorted(report["by_timeframe"].items()):
            print(
                f"  {tf:<5} всего={stats['total']:<6} подтв.={stats['confirmed']:<6} "
                f"перерисовка={stats['avg_repaint_rate']:.2%}"
            )
    else:
        print("  (нет данных)")

    os.makedirs("data", exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены в {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Валидация детектора структур")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="прогнать валидацию на размеченной синтетике с ML-фильтром",
    )
    args = parser.parse_args()

    if args.synthetic:
        sys.exit(run_synthetic())
    else:
        asyncio.run(main())
