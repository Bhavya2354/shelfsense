"""Bayesian hyperparameter search (Optuna TPE) for the learned item models.

Tuning uses only the earliest backtest fold's training and validation windows,
which end before any backtest fold starts, so reported scores stay honest. To
keep it affordable, LightGBM is tuned on three representative forecast days and
on a row sample; the network trains for a few epochs per trial.
"""

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna

from app.config import ForecastSettings
from app.features.windows import DesignMatrix
from app.forecasting.embedding_mlp import EmbeddingMLPForecaster

logger = logging.getLogger(__name__)

_TUNED_DAYS = (0, 7, 15)


def _sample(data: DesignMatrix, fraction: float, seed: int) -> DesignMatrix:
    if fraction >= 1:
        return data
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(data.x.shape[0], int(data.x.shape[0] * fraction), replace=False))
    target = None if data.target is None else data.target[rows]
    return DesignMatrix(data.x[rows], data.names, target, data.weight[rows], data.origin)


def tune_lightgbm(
    train: DesignMatrix, valid: DesignMatrix, settings: ForecastSettings
) -> dict[str, Any]:
    if train.target is None or valid.target is None:
        raise ValueError("tuning needs targets")
    train = _sample(train, settings.tuning_row_fraction, settings.random_seed)
    days = [d for d in _TUNED_DAYS if d < train.target.shape[1]]  # type: ignore[union-attr]
    train_set = lgb.Dataset(
        train.x,
        label=train.target[:, 0],  # type: ignore[index]
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

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "l2",
            "verbosity": -1,
            "seed": settings.random_seed,
            "num_threads": settings.n_jobs if settings.n_jobs > 0 else 0,
            "bagging_freq": 1,
            "learning_rate": trial.suggest_float("lgbm_learning_rate", 0.02, 0.12, log=True),
            "num_leaves": trial.suggest_int("lgbm_num_leaves", 31, 255, log=True),
            "min_data_in_leaf": trial.suggest_int("lgbm_min_data_in_leaf", 50, 1000, log=True),
            "feature_fraction": trial.suggest_float("lgbm_feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("lgbm_bagging_fraction", 0.6, 1.0),
            "lambda_l2": trial.suggest_float("lgbm_lambda_l2", 1e-3, 10.0, log=True),
        }
        losses = []
        for step, day in enumerate(days):
            train_set.set_label(train.target[:, day])  # type: ignore[index]
            valid_set.set_label(valid.target[:, day])  # type: ignore[index]
            booster = lgb.train(
                params,
                train_set,
                num_boost_round=settings.tuning_max_rounds,
                valid_sets=[valid_set],
                callbacks=[lgb.early_stopping(settings.lgbm_early_stopping_rounds, verbose=False)],
            )
            losses.append(booster.best_score["valid_0"]["l2"])
            trial.report(float(np.mean(losses)), step)
            if trial.should_prune():
                raise optuna.TrialPruned
        return float(np.mean(losses))

    return _optimise("lightgbm", objective, settings.tuning_lightgbm_trials, settings.random_seed)


def tune_mlp(
    train: DesignMatrix, valid: DesignMatrix, settings: ForecastSettings
) -> dict[str, Any]:
    train = _sample(train, settings.tuning_row_fraction, settings.random_seed)
    shapes = {"wide": (1024, 512, 256), "medium": (512, 256, 128), "narrow": (256, 128)}

    def objective(trial: optuna.Trial) -> float:
        candidate = settings.model_copy(
            update={
                "mlp_hidden_sizes": shapes[trial.suggest_categorical("shape", list(shapes))],
                "mlp_dropout": trial.suggest_float("mlp_dropout", 0.0, 0.4),
                "mlp_learning_rate": trial.suggest_float("mlp_learning_rate", 5e-4, 5e-3, log=True),
                "mlp_epochs": settings.tuning_mlp_epochs,
            }
        )
        model = EmbeddingMLPForecaster(candidate)
        model.fit(train, valid)
        pred = model.predict(valid)
        w = valid.weight[:, None]
        return float(np.sum(w * (pred - valid.target) ** 2) / (np.sum(w) * pred.shape[1]))

    best = _optimise("embedding_mlp", objective, settings.tuning_mlp_trials, settings.random_seed)
    if "shape" in best:
        best["mlp_hidden_sizes"] = shapes[best.pop("shape")]
    return best


def _optimise(name: str, objective: Any, trials: int, seed: int) -> dict[str, Any]:
    if trials == 0:
        return {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=trials)
    logger.info(
        "tuning finished",
        extra={"model": name, "best_value": study.best_value, "trials": len(study.trials)},
    )
    return dict(study.best_params)
