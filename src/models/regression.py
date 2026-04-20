"""Regression from (embedding + surprisal + baseline) features to log mean RT."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def _log_freq(table: pd.DataFrame) -> np.ndarray:
    counts = table["word"].str.lower().value_counts()
    return np.log(table["word"].str.lower().map(counts).to_numpy() + 1)


def build_splits(
    table: pd.DataFrame,
    features_npz: dict,
    layer: Optional[int],
    use_surprisal: bool,
    use_baseline: bool,
    val_stories: list[int],
    test_stories: list[int],
    standardize: bool = True,
) -> FeatureBundle:
    """Join the word table with a saved features.npz and build train/val/test splits.

    layer=None means concat of all layers; layer=-1 means final layer.
    """
    emb = features_npz["embeddings"]     # [N, num_layers, H]
    surp = features_npz["surprisal"]     # [N]
    story = features_npz["story"]
    zone = features_npz["zone"]
    feat_df = pd.DataFrame({"story": story, "zone": zone, "_row": np.arange(len(story))})
    merged = table.merge(feat_df, on=["story", "zone"], how="inner").reset_index(drop=True)
    rows = merged["_row"].to_numpy()

    if layer is None:
        E = emb[rows].reshape(len(rows), -1)
        feat_names = [f"emb_all_{i}" for i in range(E.shape[1])]
    else:
        E = emb[rows, layer, :]
        feat_names = [f"emb_L{layer}_{i}" for i in range(E.shape[1])]

    extras: list[np.ndarray] = []
    if use_surprisal:
        s = surp[rows].astype(np.float32)
        s = np.nan_to_num(s, nan=np.nanmean(s))
        extras.append(s[:, None]); feat_names.append("surprisal")
    if use_baseline:
        extras.append(merged["log_word_len"].to_numpy()[:, None]); feat_names.append("log_word_len")
        extras.append(_log_freq(merged)[:, None]); feat_names.append("log_freq")
        extras.append(merged["zone"].to_numpy()[:, None].astype(np.float32)); feat_names.append("zone")

    X = np.concatenate([E] + extras, axis=1) if extras else E
    y = merged["log_mean_rt"].to_numpy(dtype=np.float32)

    val_mask = merged["story"].isin(val_stories).to_numpy()
    test_mask = merged["story"].isin(test_stories).to_numpy()
    train_mask = ~(val_mask | test_mask)

    if standardize:
        scaler = StandardScaler().fit(X[train_mask])
        X = scaler.transform(X)

    return FeatureBundle(
        X[train_mask], y[train_mask],
        X[val_mask], y[val_mask],
        X[test_mask], y[test_mask],
        feat_names,
    )


def build_features(*args, **kwargs):
    return build_splits(*args, **kwargs)


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot
    rho, _ = spearmanr(y_true, y_pred)
    return {"r2": float(r2), "spearman": float(rho), "rmse": float(np.sqrt(ss_res / len(y_true)))}


def fit_and_evaluate(bundle: FeatureBundle, cfg) -> dict:
    """Fit a regression model (cfg.kind in {ridge, lasso, mlp}) and return val/test metrics."""
    if cfg.kind in ("ridge", "lasso"):
        best = None
        for a in cfg.alphas:
            est = Ridge(alpha=a) if cfg.kind == "ridge" else Lasso(alpha=a, max_iter=10_000)
            est.fit(bundle.X_train, bundle.y_train)
            val = _score(bundle.y_val, est.predict(bundle.X_val))
            cand = {"alpha": a, "val": val, "est": est}
            if best is None or val["r2"] > best["val"]["r2"]:
                best = cand
        est = best["est"]
        return {
            "alpha": best["alpha"],
            "val": best["val"],
            "test": _score(bundle.y_test, est.predict(bundle.X_test)),
        }

    if cfg.kind == "mlp":
        import torch
        import torch.nn as nn

        device = "cuda" if torch.cuda.is_available() else "cpu"
        in_dim = bundle.X_train.shape[1]
        layers: list = []
        prev = in_dim
        for h in cfg.hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        net = nn.Sequential(*layers).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        def to_t(a): return torch.as_tensor(a, dtype=torch.float32, device=device)
        Xt, yt = to_t(bundle.X_train), to_t(bundle.y_train)
        Xv, yv = to_t(bundle.X_val), to_t(bundle.y_val)
        Xte, yte = to_t(bundle.X_test), to_t(bundle.y_test)

        best_val, best_state, patience = -1e9, None, 0
        for epoch in range(cfg.epochs):
            net.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), cfg.batch_size):
                idx = perm[i:i + cfg.batch_size]
                opt.zero_grad()
                loss = ((net(Xt[idx]).squeeze(-1) - yt[idx]) ** 2).mean()
                loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                val_pred = net(Xv).squeeze(-1).cpu().numpy()
            val_metrics = _score(bundle.y_val, val_pred)
            if val_metrics["r2"] > best_val:
                best_val = val_metrics["r2"]; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}; patience = 0
            else:
                patience += 1
                if patience >= cfg.early_stopping_patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            val_pred = net(Xv).squeeze(-1).cpu().numpy()
            test_pred = net(Xte).squeeze(-1).cpu().numpy()
        return {
            "val": _score(bundle.y_val, val_pred),
            "test": _score(bundle.y_test, test_pred),
        }

    raise ValueError(f"Unknown regression kind: {cfg.kind}")
