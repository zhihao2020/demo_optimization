"""电价残差 BiLSTM（风格对齐 smartml BiLSTMForecaster / DeepTimeSeries）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class PriceBiLSTMConfig:
    lookback: int = 168
    horizon: int = 24
    hidden: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 30
    device: str = "cpu"
    in_features: int = 6
    out_features: int = 48


class BiLSTMCore(nn.Module):
    def __init__(self, cfg: PriceBiLSTMConfig) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.in_features,
            hidden_size=cfg.hidden,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden * 2, cfg.out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class PriceBiLSTM:
    def __init__(self, cfg: PriceBiLSTMConfig | None = None) -> None:
        self.cfg = cfg or PriceBiLSTMConfig()
        self.device = torch.device(self.cfg.device)
        self.model = BiLSTMCore(self.cfg).to(self.device)
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None

    @staticmethod
    def calendar_features(hour_index: np.ndarray) -> np.ndarray:
        h = hour_index % 24
        ang = 2.0 * np.pi * h / 24.0
        return np.stack([np.sin(ang), np.cos(ang)], axis=-1)

    def build_arrays(self, buy_da, sell_da, eps_buy, eps_sell):
        n = len(buy_da)
        hours = np.arange(n, dtype=np.float64)
        cal = self.calendar_features(hours)
        feats = np.column_stack([buy_da, sell_da, eps_buy, eps_sell, cal[:, 0], cal[:, 1]]).astype(
            np.float32
        )
        L, H = self.cfg.lookback, self.cfg.horizon
        xs, ys = [], []
        for t in range(L - 1, n - H):
            xs.append(feats[t - L + 1 : t + 1])
            y = np.empty(H * 2, dtype=np.float32)
            y[0::2] = eps_buy[t + 1 : t + 1 + H]
            y[1::2] = eps_sell[t + 1 : t + 1 + H]
            ys.append(y)
        return np.stack(xs), np.stack(ys)

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        self._feat_mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
        self._feat_std = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
        Xn = (X - self._feat_mean) / self._feat_std
        n = Xn.shape[0]
        n_val = max(1, int(0.1 * n))
        tr, va = np.arange(n - n_val), np.arange(n - n_val, n)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        loss_fn = nn.MSELoss()
        best, best_state = float("inf"), None
        history = {"train_mse": 0.0, "val_mse": 0.0}
        for epoch in range(self.cfg.epochs):
            self.model.train()
            perm = np.random.permutation(tr)
            total, count = 0.0, 0
            for i in range(0, len(perm), self.cfg.batch_size):
                b = perm[i : i + self.cfg.batch_size]
                xb = torch.as_tensor(Xn[b], device=self.device)
                yb = torch.as_tensor(y[b], device=self.device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(b)
                count += len(b)
            train_mse = total / max(count, 1)
            self.model.eval()
            with torch.no_grad():
                xv = torch.as_tensor(Xn[va], device=self.device)
                yv = torch.as_tensor(y[va], device=self.device)
                val_mse = float(loss_fn(self.model(xv), yv).item())
            history = {"train_mse": train_mse, "val_mse": val_mse, "epoch": float(epoch + 1)}
            if val_mse < best:
                best = val_mse
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def predict_eps(self, window_feats: np.ndarray) -> np.ndarray:
        assert self._feat_mean is not None and self._feat_std is not None
        x = (window_feats - self._feat_mean) / self._feat_std
        self.model.eval()
        with torch.no_grad():
            xt = torch.as_tensor(x[None, ...], dtype=torch.float32, device=self.device)
            return self.model(xt).cpu().numpy()[0].astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.cfg),
                "state_dict": self.model.state_dict(),
                "feat_mean": self._feat_mean,
                "feat_std": self._feat_std,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PriceBiLSTM":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(PriceBiLSTMConfig(**blob["config"]))
        obj.model.load_state_dict(blob["state_dict"])
        obj._feat_mean = np.asarray(blob["feat_mean"], dtype=np.float64)
        obj._feat_std = np.asarray(blob["feat_std"], dtype=np.float64)
        return obj
