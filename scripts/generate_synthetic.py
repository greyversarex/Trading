"""CLI для генерации размеченного синтетического датасета паттернов.

Запуск:
    python scripts/generate_synthetic.py [--n-per-class N] [--n-noise M] [--seed S]

Сохраняет датасет в ``data/synthetic_dataset.json`` — список образцов
``{candles, label, target_confirmed}``, используемый для обучения ML-фильтра
(Phase 2.3) и для синтетической валидации (Phase 2.5).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.synthetic import SyntheticChartGenerator

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_dataset.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Сгенерировать синтетический датасет паттернов")
    parser.add_argument("--n-per-class", type=int, default=100)
    parser.add_argument("--n-noise", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gen = SyntheticChartGenerator()
    dataset = gen.generate_labeled_dataset(
        n_per_class=args.n_per_class, n_noise=args.n_noise, seed=args.seed
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(dataset, f)

    counts: dict = {}
    for s in dataset:
        counts[s["label"]] = counts.get(s["label"], 0) + 1
    print(f"Сохранено {len(dataset)} образцов в {os.path.relpath(OUT_PATH)}")
    for label in sorted(counts):
        print(f"  {label}: {counts[label]}")


if __name__ == "__main__":
    main()
