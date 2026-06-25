"""Centralized configuration for the chart scanner.

Все настраиваемые параметры детекторов, сканера и валидации собраны здесь
в виде dataclass-ов. Доступ к ним осуществляется через глобальный синглтон
``CONFIG``. Модули проекта читают значения отсюда вместо «магических чисел».

Изменения значений по умолчанию, внесённые осознанно (Phase 0.1):
  * ``StructureConfig.min_quality``: 0.15 -> 0.35 (более строгий порог качества).
  * ``StructureConfig.recency_weight_enabled``: отключено старое смещение
    в сторону недавних данных.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DataConfig:
    """Параметры загрузки рыночных данных и работы сканера Binance."""

    timframes: Dict[str, int] = field(
        default_factory=lambda: {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    )
    default_limit: int = 100
    window_sizes: List[int] = field(default_factory=lambda: [30, 50, 70])
    num_symbols: int = 50
    poll_interval_sec: float = 60.0
    error_sleep_sec: float = 10.0
    batch_sleep_sec: float = 0.15

    # Real-time feed (Phase 3.1)
    use_websocket: bool = True
    ws_reconnect_max_delay_sec: float = 30.0
    ws_heartbeat_interval_sec: float = 30.0
    # Сколько ждать первого успешного подключения WS перед фаллбэком на REST.
    ws_startup_timeout_sec: float = 10.0

    # Performance (Phase 3.3)
    max_concurrent_symbol_tasks: int = 10
    cache_structures: bool = True


@dataclass
class StructureConfig:
    """Параметры извлечения структуры и детекции пивотов."""

    resample_points: int = 100
    num_pivots: int = 10
    min_quality: float = 0.35          # raised from 0.15
    use_ohlc_for_pivots: bool = True   # NEW
    recency_weight_enabled: bool = False  # disable old recency bias

    # Pivot detection
    zigzag_atr_mult_min: float = 1.5
    zigzag_atr_mult_max: float = 3.5
    zigzag_min_swing_floor: float = 0.02
    prominence_min: float = 0.08
    prominence_max: float = 0.25
    prominence_base: float = 0.12
    multi_scale_order: int = 3

    # Confirmation (NEW)
    confirmation_bars: int = 1          # bars after breakout to confirm
    confirmation_retreat_fraction: float = 0.5  # how much of breakout can be retraced without invalidation


@dataclass
class PatternConfig:
    """Пороги детекции графических паттернов (доли ATR или ценового диапазона)."""

    # thresholds are fractions of ATR or price range, not fixed normalized values
    double_top_tolerance: float = 0.03
    double_top_min_conf: float = 0.45
    double_bottom_min_conf: float = 0.45

    # ATR-adaptive tolerances (NEW, T1.2): применяются в каузальном/сыром слое.
    # Итоговый допуск = max(fraction * price_range, mult * volatility_scale),
    # поэтому ATR-адаптация только ОСЛАБЛЯЕТ пороги (не теряет паттерны при
    # высокой волатильности) и никогда не делает их строже фиксированных.
    double_top_atr_mult: float = 0.5      # k * volatility_scale для допуска равенства вершин
    fit_residual_atr_mult: float = 2.0    # допустимый остаток фита в единицах volatility_scale

    hs_shoulder_diff_max: float = 0.18
    hs_head_prominence_min: float = 0.06
    hs_shoulder_head_ratio_min: float = 0.4
    hs_shoulder_head_ratio_max: float = 0.95
    hs_min_conf: float = 0.5

    flag_min_conf: float = 0.5
    flag_breakout_retreat: float = 0.03

    wedge_convergence_min: float = 0.15
    wedge_fit_min: float = 0.7
    wedge_min_conf: float = 0.45

    triangle_convergence_min: float = 0.15
    triangle_convergence_max: float = 0.95
    triangle_norm_conv_rate_max: float = 3.0
    triangle_fit_min: float = 0.75
    triangle_min_conf: float = 0.45

    trend_move_threshold: float = 0.15
    swing_ratio_threshold: float = 0.5

    # Channel (NEW)
    channel_fit_min: float = 0.80
    channel_parallel_tolerance: float = 0.05
    channel_min_conf: float = 0.45

    # Missing patterns (Phase 2.2). Допуски — доли ценового диапазона либо
    # множители volatility_scale, как у каузальных детекторов из Phase 1.
    triple_top_tolerance: float = 0.035
    # Ослабленный допуск асимметрии вершин (Phase 3.4): когда у тройной вершины
    # есть чёткий нэклайн (выраженная глубина), разрешаем большую разницу высот
    # вершин — иначе паттерн ошибочно классифицируется как двойная вершина.
    triple_top_asymmetry_tolerance: float = 0.06
    triple_top_min_conf: float = 0.45
    triple_bottom_tolerance: float = 0.035
    triple_bottom_min_conf: float = 0.45
    cup_and_handle_depth_min: float = 0.08      # мин. глубина чаши (доля диапазона)
    cup_and_handle_handle_retrace_max: float = 0.5  # макс. откат ручки от глубины чаши
    # Ослабленный предел отката ручки (Phase 3.4): используется в fallback-ветке,
    # когда левый край чаши не оформлен отдельным пивотом-хаем. Также допускается
    # «чаша без ручки» (cup without handle) при выраженной симметричной чаше.
    cup_and_handle_relaxed_handle_retrace_max: float = 0.7
    cup_and_handle_min_conf: float = 0.45
    pennant_pole_min_move: float = 0.12         # мин. движение полюса (доля цены)
    pennant_convergence_min: float = 0.15       # мин. схождение границ вымпела
    pennant_min_conf: float = 0.45
    rounding_bottom_curvature_min: float = 0.10  # мин. «выпуклость» дна (доля диапазона)
    rounding_bottom_min_conf: float = 0.45


@dataclass
class LevelConfig:
    """Параметры детекции уровней поддержки/сопротивления и трендовых линий."""

    deviation_pct: float = 0.15
    min_touches: int = 3
    breakout_threshold_factor: float = 0.003
    dedup_rel_dist: float = 0.005
    dedup_slope_diff: float = 0.01
    # NEW
    use_ransac: bool = True
    ransac_min_samples: int = 2
    ransac_residual_threshold_factor: float = 0.5  # * ATR
    use_dbscan: bool = True
    dbscan_eps_factor: float = 0.5  # * ATR
    dbscan_min_samples: int = 2
    use_volume_profile: bool = True
    volume_profile_bins: int = 20
    use_round_numbers: bool = True
    round_number_step_factor: float = 1.0  # * ATR


@dataclass
class FiboConfig:
    """Параметры анализа уровней Фибоначчи."""

    touch_tolerance: float = 0.003
    min_swing_pct: float = 0.02


@dataclass
class CandleConfig:
    """Параметры детекции свечных паттернов."""

    max_age: int = 10
    volume_floor: float = 0.6
    lookback: int = 5


@dataclass
class SimilarityConfig:
    """Веса и пороги сопоставления структур."""

    line_weight: float = 0.25
    pivot_weight: float = 0.20
    distance_weight: float = 0.10
    shape_weight: float = 0.25
    slope_weight: float = 0.10
    geometry_weight: float = 0.10
    dtw_band_window: int = 10
    type_sim_same: float = 1.0
    type_sim_same_category: float = 0.60
    type_sim_mismatch: float = 0.35
    type_sim_opposite: float = 0.15
    pattern_overlap_bonus_cap: float = 0.30
    mirror_penalty_opposite: float = 0.60
    mirror_penalty_default: float = 0.85
    mirror_penalty_consolidation: float = 0.90
    dedup_score_window: float = 5.0
    mtf_bonus_2tf: float = 1.04
    mtf_bonus_3tf: float = 1.08


@dataclass
class ValidationConfig:
    """Параметры валидационного/бэктест-харнесса."""

    test_symbols: List[str] = field(
        default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    )
    test_timeframes: List[str] = field(default_factory=lambda: ["15m", "1h", "4h"])
    horizon_candles: int = 5
    max_history_limit: int = 500


@dataclass
class MLConfig:
    """Параметры ML-фильтра релевантности (снижение ложных срабатываний)."""

    ml_min_training_samples: int = 200   # минимум образцов для обучения
    ml_feedback_weight: float = 1.0      # вес одной записи обратной связи vs синтетика
    ml_retrain_interval_hours: float = 24.0  # период авто-переобучения
    ml_score_threshold: float = 0.5      # порог фильтрации матчей по ml_score
    # Жёсткий ML-фильтр (отсечение матчей ниже порога) включается только когда
    # модель дообучена минимум на таком количестве записей реальной обратной
    # связи. До этого модель обучена лишь на синтетике и не переносится на
    # реальный рынок, поэтому ml_score используется только как бейдж/сортировка.
    ml_hard_filter_min_feedback: int = 20
    ml_model_path: str = "data/ml_relevance_model.pkl"


@dataclass
class AppConfig:
    """Корневой агрегат всех групп настроек."""

    data: DataConfig = field(default_factory=DataConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    pattern: PatternConfig = field(default_factory=PatternConfig)
    level: LevelConfig = field(default_factory=LevelConfig)
    fibo: FiboConfig = field(default_factory=FiboConfig)
    candle: CandleConfig = field(default_factory=CandleConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    ml: MLConfig = field(default_factory=MLConfig)


# global singleton
CONFIG = AppConfig()
