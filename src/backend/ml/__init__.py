"""ML-подсистема: извлечение признаков, классификатор релевантности и обучение.

Назначение пакета — снизить долю ложных срабатываний геометрических детекторов
за счёт обучаемого фильтра. Каждый эмитируемый матч получает оценку
``ml_score`` (вероятность того, что паттерн действительно релевантен), и матчи
ниже порога отсекаются.

Модули:
  * ``features``  — извлечение числовых признаков из ``StructureFeatures``.
  * ``model``     — обёртка над ``RandomForestClassifier`` с save/load.
  * ``trainer``   — сборка обучающего набора и обучение/переобучение.
  * ``pipeline``  — удобный оркестратор для загрузки модели и скоринга.
"""

from .features import extract_ml_features, feature_names, FEATURE_ORDER
from .model import RelevanceClassifier
from .trainer import ModelTrainer
from .pipeline import MLPipeline

__all__ = [
    "extract_ml_features",
    "feature_names",
    "FEATURE_ORDER",
    "RelevanceClassifier",
    "ModelTrainer",
    "MLPipeline",
]
