"""
Tree-based model builders for SC3 benchmark.

Each function takes (params, seed, X_train, y_train, X_eval, y_eval)
and returns a trained model with a .predict(X) method.
"""

import numpy as np
from ..registry import NJOBS


def train_lgb(params, seed, X_train, y_train, X_eval, y_eval):
    from lightgbm import LGBMRegressor
    import lightgbm as lgb
    m = LGBMRegressor(random_state=seed, n_jobs=NJOBS, verbose=-1, **params)
    m.fit(X_train, y_train, eval_set=[(X_eval, y_eval)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def train_catboost(params, seed, X_train, y_train, X_eval, y_eval):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(
        random_seed=seed, thread_count=NJOBS, verbose=0,
        early_stopping_rounds=50, eval_metric="RMSE", **params,
    )
    m.fit(X_train, y_train, eval_set=(X_eval, y_eval), verbose=False)
    return m


def train_xgb(params, seed, X_train, y_train, X_eval, y_eval):
    from xgboost import XGBRegressor
    m = XGBRegressor(
        random_state=seed, n_jobs=NJOBS, verbosity=0,
        early_stopping_rounds=50, eval_metric="rmse", **params,
    )
    m.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
    return m


def train_rf(params, seed, X_train, y_train, X_eval, y_eval):
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(random_state=seed, n_jobs=NJOBS, **params)
    m.fit(X_train, y_train)
    return m


def train_dt(params, seed, X_train, y_train, X_eval, y_eval):
    from sklearn.tree import DecisionTreeRegressor
    m = DecisionTreeRegressor(random_state=seed, **params)
    m.fit(X_train, y_train)
    return m


def train_tayyebi(params, seed, X_train, y_train, X_eval, y_eval):
    """Tayyebi pipeline: variance filter -> correlation filter -> RF."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import VarianceThreshold

    p = dict(params)
    var_thresh = p.pop("var_threshold", 0.1)
    corr_cut   = p.pop("corr_cutoff", 0.8)

    vt = VarianceThreshold(threshold=var_thresh)
    vt.fit(X_train)
    X_var = vt.transform(X_train)

    corr = np.nan_to_num(np.corrcoef(X_var, rowvar=False))
    upper = np.triu(np.abs(corr), k=1)
    to_drop = set()
    for i in range(upper.shape[1]):
        if i in to_drop:
            continue
        for j in range(i + 1, upper.shape[1]):
            if upper[i, j] > corr_cut:
                to_drop.add(j)
    keep_idx = sorted(set(range(X_var.shape[1])) - to_drop)

    rf = RandomForestRegressor(random_state=seed, n_jobs=NJOBS, **p)
    rf.fit(X_var[:, keep_idx], y_train)

    def _predict(X):
        return rf.predict(vt.transform(X)[:, keep_idx])

    return type("TayyebiModel", (), {"predict": staticmethod(_predict)})()


def train_gp(params, seed, X_train, y_train, X_eval, y_eval):
    """GP with Tanimoto kernel (subset-of-data approximation)."""
    import torch
    import gpytorch

    subset_size = params.get("subset_size", 3000)
    n_iters     = params.get("n_train_iters", 50)
    lr          = params.get("lr", 0.1)

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    n = len(y_train)
    if n > subset_size:
        idx = rng.choice(n, size=subset_size, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]
    else:
        X_sub, y_sub = X_train, y_train

    y_mean = float(np.mean(y_sub))
    y_std  = max(float(np.std(y_sub)), 1e-6)
    y_norm = (y_sub - y_mean) / y_std

    train_x = torch.tensor(X_sub, dtype=torch.float64)
    train_y = torch.tensor(y_norm, dtype=torch.float64)

    class TanimotoKernel(gpytorch.kernels.Kernel):
        is_stationary = False
        has_lengthscale = False
        def forward(self, x1, x2, diag=False, **kw):
            if diag:
                return torch.ones(x1.shape[:-1], dtype=x1.dtype, device=x1.device)
            dot = x1 @ x2.transpose(-1, -2)
            x1n = (x1 ** 2).sum(-1, keepdim=True)
            x2n = (x2 ** 2).sum(-1, keepdim=True)
            return (dot / (x1n + x2n.transpose(-1, -2) - dot + 1e-6)).clamp_min_(0)

    class _GPModel(gpytorch.models.ExactGP):
        def __init__(self, tx, ty, lik):
            super().__init__(tx, ty, lik)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x))

    lik = gpytorch.likelihoods.GaussianLikelihood().double()
    model = _GPModel(train_x, train_y, lik).double()

    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    for _ in range(n_iters):
        opt.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        opt.step()

    model.eval(); lik.eval()

    def _predict(X):
        Xt = torch.tensor(X, dtype=torch.float64)
        preds = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, len(Xt), 2000):
                preds.append(lik(model(Xt[i:i+2000])).mean.numpy())
        return np.concatenate(preds) * y_std + y_mean

    return type("GPModel", (), {"predict": staticmethod(_predict)})()


TREE_BUILDERS = {
    "lgb": train_lgb,
    "catboost": train_catboost,
    "xgb": train_xgb,
    "rf": train_rf,
    "dt": train_dt,
    "tayyebi": train_tayyebi,
    "gp": train_gp,
}
