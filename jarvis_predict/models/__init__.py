"""
Predictive model adapters — uniform interface over TimesFM, Chronos,
qlib's many algorithms, FinGPT, FinRobot.

Every model implements `PredictiveModel` and exposes
`forecast(values, horizon) -> ModelForecast`.

Use `build_fleet(specs_from_config(config.section("models")))` to load
the whole fleet from `config.yaml`.
"""

from .base import PredictiveModel, ModelForecast
from .mock import MockModel
from .registry import (
    ModelSpec,
    ModelFleet,
    build_fleet,
    specs_from_config,
)
from .timesfm_adapter import TimesFMAdapter
from .chronos_adapter import ChronosAdapter
from .qlib_adapter import (
    QlibAdapter,
    list_qlib_algorithms,
    make_all_qlib_adapters,
)
from .fingpt_adapter import FinGPTForecasterAdapter, FinGPTSentimentAdapter
from .finrobot_adapter import FinRobotAnalystAdapter
from .deepseek_v41_adapter import DeepSeekV41Adapter

__all__ = [
    "PredictiveModel",
    "ModelForecast",
    "MockModel",
    "ModelSpec",
    "ModelFleet",
    "build_fleet",
    "specs_from_config",
    "TimesFMAdapter",
    "ChronosAdapter",
    "QlibAdapter",
    "list_qlib_algorithms",
    "make_all_qlib_adapters",
    "FinGPTForecasterAdapter",
    "FinGPTSentimentAdapter",
    "FinRobotAnalystAdapter",
    "DeepSeekV41Adapter",
]
