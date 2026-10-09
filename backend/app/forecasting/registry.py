"""Name-to-factory registry so pipelines select models from configuration."""

from collections.abc import Callable

from app.config import ForecastSettings
from app.forecasting.base import ItemForecaster

ItemFactory = Callable[[ForecastSettings], ItemForecaster]

_ITEM_MODELS: dict[str, ItemFactory] = {}


def register_item_model(name: str) -> Callable[[ItemFactory], ItemFactory]:
    def decorator(factory: ItemFactory) -> ItemFactory:
        if name in _ITEM_MODELS:
            raise ValueError(f"item model {name!r} registered twice")
        _ITEM_MODELS[name] = factory
        return factory

    return decorator


def item_model_names() -> list[str]:
    return list(_ITEM_MODELS)


def create_item_model(name: str, settings: ForecastSettings) -> ItemForecaster:
    try:
        factory = _ITEM_MODELS[name]
    except KeyError:
        raise ValueError(f"unknown item model {name!r}; known: {item_model_names()}") from None
    return factory(settings)
