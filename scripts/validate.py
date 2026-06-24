"""CLI для запуска валидации детектора структур на исторических данных.

Запуск:
    python scripts/validate.py

Скрипт прогоняет детекцию по тестовым символам/таймфреймам из
``CONFIG.validation``, печатает метрики (precision/recall/F1, перерисовка) по
типам паттернов и таймфреймам и сохраняет результаты в
``data/validation_results.json``.
"""
from __future__ import annotations

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
    asyncio.run(main())
