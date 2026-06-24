"""CLI: обучение ML-фильтра релевантности.

Загружает синтетический датасет (``data/synthetic_dataset.json``), при наличии
подмешивает обратную связь пользователя из SQLite, обучает классификатор и
сохраняет модель в ``data/ml_relevance_model.pkl``. Печатает метрики и важности
признаков.

Запуск:
    python scripts/train_ml.py [--dataset PATH] [--no-feedback]
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backend.config import CONFIG
from src.backend.database import Database
from src.backend.ml.trainer import ModelTrainer


async def _load_feedback() -> list:
    """Загружает обратную связь из БД (мягко, при ошибке — пустой список)."""
    try:
        db = Database()
        await db.initialize()
        return await db.get_all_feedback()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] не удалось загрузить обратную связь: {exc}")
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Обучение ML-фильтра релевантности")
    parser.add_argument(
        "--dataset",
        default="data/synthetic_dataset.json",
        help="Путь к синтетическому датасету",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Не подмешивать обратную связь из БД",
    )
    parser.add_argument(
        "--out",
        default=CONFIG.ml.ml_model_path,
        help="Путь сохранения модели",
    )
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(
            f"[error] датасет не найден: {args.dataset}\n"
            "Сначала выполните: python scripts/generate_synthetic.py"
        )
        return 1

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"[info] загружено образцов: {len(dataset)}")

    feedback = []
    if not args.no_feedback:
        feedback = asyncio.run(_load_feedback())
        usable = sum(1 for r in feedback if r.get("features"))
        print(f"[info] записей обратной связи: {len(feedback)} (пригодных: {usable})")

    trainer = ModelTrainer(model_path=args.out)
    result = trainer.train(dataset, feedback_records=feedback, save=True)

    print("\n=== Результат обучения ===")
    print(f"Образцов всего: {result['n_samples']} "
          f"(положительных={result['n_positive']}, отрицательных={result['n_negative']})")

    if not result.get("trained"):
        print(f"[error] модель не обучена: {result.get('reason')}")
        return 1

    print(f"Точность (hold-out): {result['accuracy']:.4f}")
    print(f"Модель сохранена: {result['model_path']}")
    print("\nClassification report:")
    print(result["report"])

    print("Топ-15 важных признаков:")
    for name, imp in list(result["feature_importances"].items())[:15]:
        print(f"  {name:32s} {imp:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
