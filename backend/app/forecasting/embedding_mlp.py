"""Entity-embedding network (Guo & Berkhahn, 2016) with a 16-day output head.

Categorical attributes (store, family, city, ...) get learned embeddings; the
numeric features are standardised with statistics from the training rows only.
"""

import copy
import logging

import numpy as np
import torch
from torch import nn

from app.config import ForecastSettings
from app.features.windows import DesignMatrix
from app.forecasting.registry import register_item_model

logger = logging.getLogger(__name__)

_MIN_STD = 1e-3


class _Network(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        n_numeric: int,
        hidden: tuple[int, ...],
        dropout: float,
        horizon: int,
    ) -> None:
        super().__init__()
        dims = [min(16, (card + 1) // 2) for card in cardinalities]
        self.embeddings = nn.ModuleList(
            nn.Embedding(card, dim) for card, dim in zip(cardinalities, dims, strict=True)
        )
        width = n_numeric + sum(dims)
        layers: list[nn.Module] = []
        for size in hidden:
            layers += [nn.Linear(width, size), nn.BatchNorm1d(size), nn.SiLU(), nn.Dropout(dropout)]
            width = size
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, horizon)

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        embedded = [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        out: torch.Tensor = self.head(self.body(torch.cat([numeric, *embedded], dim=1)))
        return out


class EmbeddingMLPForecaster:
    name = "embedding_mlp"

    def __init__(self, settings: ForecastSettings) -> None:
        self._s = settings
        self._net: _Network | None = None
        self._cat_idx: list[int] = []
        self._num_idx: list[int] = []
        self._mean: np.ndarray = np.zeros(0, dtype=np.float32)
        self._std: np.ndarray = np.ones(0, dtype=np.float32)

    def _split(self, data: DesignMatrix) -> tuple[torch.Tensor, torch.Tensor]:
        categorical = torch.from_numpy(data.x[:, self._cat_idx].astype(np.int64))
        numeric = (data.x[:, self._num_idx] - self._mean) / self._std
        return categorical, torch.from_numpy(numeric.astype(np.float32))

    def fit(self, train: DesignMatrix, valid: DesignMatrix) -> None:
        if train.target is None or valid.target is None:
            raise ValueError("the network needs targets for training and validation")
        torch.manual_seed(self._s.random_seed)
        torch.set_num_threads(self._s.n_jobs)
        self._cat_idx = [train.names.index(c) for c in train.categorical]
        self._num_idx = [i for i in range(len(train.names)) if i not in self._cat_idx]
        numeric = train.x[:, self._num_idx]
        self._mean = numeric.mean(axis=0)
        # Columns constant in training (e.g. a holiday flag that never fires) keep scale 1,
        # otherwise any later non-zero value would be blown up by a near-zero divisor.
        std = numeric.std(axis=0)
        self._std = np.where(std < _MIN_STD, 1.0, std).astype(np.float32)
        # Codes seen only in later windows still need an embedding row.
        cardinalities = [
            int(max(train.x[:, i].max(), valid.x[:, i].max())) + 2 for i in self._cat_idx
        ]
        net = _Network(
            cardinalities,
            len(self._num_idx),
            self._s.mlp_hidden_sizes,
            self._s.mlp_dropout,
            train.target.shape[1],
        )

        cat_t, num_t = self._split(train)
        y_t = torch.from_numpy(train.target)
        w_t = torch.from_numpy(train.weight)
        cat_v, num_v = self._split(valid)
        y_v, w_v = torch.from_numpy(valid.target), torch.from_numpy(valid.weight)

        steps_per_epoch = -(-len(y_t) // self._s.mlp_batch_size)
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=self._s.mlp_learning_rate, weight_decay=1e-5
        )
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self._s.mlp_learning_rate,
            total_steps=steps_per_epoch * self._s.mlp_epochs,
        )
        best_loss, best_state = float("inf"), copy.deepcopy(net.state_dict())
        for epoch in range(self._s.mlp_epochs):
            net.train()
            order = torch.randperm(len(y_t))
            for start in range(0, len(order), self._s.mlp_batch_size):
                batch = order[start : start + self._s.mlp_batch_size]
                pred = net(cat_t[batch], num_t[batch])
                loss = ((pred - y_t[batch]) ** 2).mean(dim=1).mul(w_t[batch]).sum() / w_t[
                    batch
                ].sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                schedule.step()
            valid_loss = self._weighted_mse(net, cat_v, num_v, y_v, w_v)
            logger.info("epoch", extra={"epoch": epoch, "valid_loss": round(valid_loss, 5)})
            if valid_loss < best_loss:
                best_loss, best_state = valid_loss, copy.deepcopy(net.state_dict())
        net.load_state_dict(best_state)
        self._net = net

    @torch.no_grad()
    def _weighted_mse(
        self,
        net: _Network,
        cat: torch.Tensor,
        num: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
    ) -> float:
        net.eval()
        pred = self._batched(net, cat, num)
        return float((((pred.clamp(min=0) - y) ** 2).mean(dim=1) * w).sum() / w.sum())

    def _batched(self, net: _Network, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        size = self._s.mlp_batch_size * 4
        return torch.cat(
            [net(cat[i : i + size], num[i : i + size]) for i in range(0, len(cat), size)]
        )

    @torch.no_grad()
    def predict(self, data: DesignMatrix) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("fit must run before predict")
        self._net.eval()
        cat, num = self._split(data)
        return self._batched(self._net, cat, num).clamp(min=0).numpy().astype(np.float32)


@register_item_model(EmbeddingMLPForecaster.name)
def _embedding_mlp(settings: ForecastSettings) -> EmbeddingMLPForecaster:
    return EmbeddingMLPForecaster(settings)
