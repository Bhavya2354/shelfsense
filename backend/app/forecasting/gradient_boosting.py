"""Direct multi-horizon LightGBM: one model per forecast day over shared features."""

import logging

import lightgbm as lgb
import numpy as np

from app.config import ForecastSettings
from app.features.windows import DesignMatrix
from app.forecasting.registry import register_item_model

logger = logging.getLogger(__name__)


class LightGBMForecaster:
    name = "lightgbm"

    def __init__(self, settings: ForecastSettings) -> None:
        self._settings = settings
        self._models: list[lgb.Booster] = []
        self.feature_importance: dict[str, float] = {}

    def _params(self) -> dict[str, object]:
        s = self._settings
        return {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": s.lgbm_learning_rate,
            "num_leaves": s.lgbm_num_leaves,
            "min_data_in_leaf": s.lgbm_min_data_in_leaf,
            "feature_fraction": s.lgbm_feature_fraction,
            "bagging_fraction": s.lgbm_bagging_fraction,
            "bagging_freq": 1,
            "lambda_l2": s.lgbm_lambda_l2,
            "num_threads": s.n_jobs,
            "seed": s.random_seed,
            "verbosity": -1,
        }

    def fit(self, train: DesignMatrix, valid: DesignMatrix) -> None:
        if train.target is None or valid.target is None:
            raise ValueError("LightGBM needs targets for training and validation")
        # Bin the features once; only the label changes between horizons.
        train_set = lgb.Dataset(
            train.x,
            label=train.target[:, 0],
            weight=train.weight,
            feature_name=train.names,
            categorical_feature=train.categorical,
            free_raw_data=False,
        ).construct()
        valid_set = lgb.Dataset(
            valid.x,
            label=valid.target[:, 0],
            weight=valid.weight,
            feature_name=valid.names,
            categorical_feature=valid.categorical,
            reference=train_set,
        ).construct()

        self._models = []
        gains = np.zeros(len(train.names))
        for day in range(train.target.shape[1]):
            train_set.set_label(train.target[:, day])
            valid_set.set_label(valid.target[:, day])
            booster = lgb.train(
                self._params(),
                train_set,
                num_boost_round=self._settings.lgbm_max_rounds,
                valid_sets=[valid_set],
                callbacks=[
                    lgb.early_stopping(self._settings.lgbm_early_stopping_rounds, verbose=False)
                ],
            )
            gains += booster.feature_importance("gain", iteration=booster.best_iteration)
            self._models.append(booster)
            logger.info("trained horizon", extra={"day": day, "rounds": booster.best_iteration})
        self.feature_importance = dict(
            sorted(
                zip(train.names, (gains / gains.sum()).tolist(), strict=True), key=lambda kv: -kv[1]
            )
        )

    def predict(self, data: DesignMatrix) -> np.ndarray:
        preds = [m.predict(data.x, num_iteration=m.best_iteration) for m in self._models]
        forecast: np.ndarray = np.clip(np.column_stack(preds), 0, None).astype(np.float32)
        return forecast


@register_item_model(LightGBMForecaster.name)
def _lightgbm(settings: ForecastSettings) -> LightGBMForecaster:
    return LightGBMForecaster(settings)
