"""Supplied utilities for ACTSC 972 guided deep hedging.

The three student functions are passed as arguments: rollout, terminal_loss,
and es_objective. There is no reference solution to those functions here.
CPU tensors and one PyTorch thread are used for the reproducible course pilot.
All paths are synthetic. No market data or credentials are required.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import copy
import csv
import json
import math
import platform
import time

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    S0: float = 100.0
    K: float = 100.0
    T: float = 20 / 252
    N: int = 20
    sigma: float = 0.20
    c: float = 0.001
    alpha: float = 0.95
    batch_size: int = 1024
    n_steps: int = 1000
    learning_rate: float = 1e-3
    validate_every: int = 100
    n_validation: int = 5000
    n_test: int = 20000
    seed: int = 17
    validation_seed: int = 271828
    test_seed: int = 314159

    @property
    def premium(self):
        # Always the ORIGINAL contract premium, including under stresses.
        return bs_call_price(self.S0, self.K, self.T, self.sigma)


def environment_info():
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": str(np.__version__), "platform": platform.platform(),
            "device": "cpu", "torch_threads": torch.get_num_threads()}


def bs_call_price(S0, K, T, sigma):
    """Zero rates and dividends; sigma is annualized volatility."""
    if min(S0, K, T, sigma) <= 0:
        raise ValueError("Positive S0, K, T, and sigma are required.")
    d1 = (math.log(S0 / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return S0 * phi(d1) - K * phi(d2)


def fixed_shocks(n_paths, N, seed, dtype=torch.float32):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((n_paths, N), generator=gen, dtype=dtype)


def gbm_paths(cfg, *, n_paths=None, generator=None, shocks=None, volatility=None):
    """Exact GBM grid law with zero drift; volatility may be scalar or length N.

    Supplying the same shocks couples scenarios without making their price
    paths identical. Training uses its own generator, never the test generator.
    """
    if shocks is None:
        if n_paths is None or generator is None:
            raise ValueError("Supply shocks or both n_paths and a generator.")
        shocks = torch.randn((n_paths, cfg.N), generator=generator)
    if shocks.ndim != 2 or shocks.shape[1] != cfg.N:
        raise ValueError("shocks must have shape (number of paths, cfg.N).")
    sigma = torch.as_tensor(cfg.sigma if volatility is None else volatility,
                            dtype=shocks.dtype, device=shocks.device)
    if sigma.ndim > 1 or (sigma.ndim == 1 and sigma.numel() != cfg.N):
        raise ValueError("volatility must be a scalar or a length-N vector.")
    if not torch.isfinite(sigma).all() or (sigma < 0).any():
        raise ValueError("volatility must be finite and nonnegative.")
    dt = cfg.T / cfg.N
    increments = -0.5 * sigma.square() * dt + sigma * math.sqrt(dt) * shocks
    initial = torch.zeros((shocks.shape[0], 1), dtype=shocks.dtype)
    log_paths = torch.cat((initial, increments.cumsum(dim=1)), dim=1)
    return cfg.S0 * log_paths.exp()


def make_policy(width=32, depth=2, seed=17):
    """depth counts hidden layers; the same module is reused at every date."""
    if width < 1 or depth < 1:
        raise ValueError("width and depth must be positive integers.")
    torch.manual_seed(seed)
    layers = []
    dim = 3
    for _ in range(depth):
        layers.extend([nn.Linear(dim, width), nn.Tanh()])
        dim = width
    layers.extend([nn.Linear(dim, 1), nn.Sigmoid()])
    return nn.Sequential(*layers)


def features(Sk, previous, k, cfg):
    """Three causal inputs; previous must remain attached to autograd."""
    return torch.stack((torch.full_like(Sk, (cfg.N - k) / cfg.N),
                        torch.log(Sk / cfg.K), previous), dim=-1)


def delta_holdings(S, cfg):
    """Benchmark keeps cfg.sigma (20% by default) fixed in stress tests."""
    tau = cfg.T * (cfg.N - torch.arange(cfg.N, dtype=S.dtype)) / cfg.N
    d1 = (torch.log(S[:, :-1] / cfg.K) + 0.5 * cfg.sigma**2 * tau)
    d1 = d1 / (cfg.sigma * tau.sqrt())
    return 0.5 * (1 + torch.erf(d1 / math.sqrt(2)))


def empirical_es(losses, alpha=0.95):
    """Exact empirical upper-tail ES, including fractional tail mass and ties.

    This is an EVALUATION function. Students implement the differentiable
    OCE minibatch objective separately. The trained eta is not used here.
    """
    if losses.ndim != 1 or losses.numel() == 0 or not torch.isfinite(losses).all():
        raise ValueError("Provide a nonempty vector of finite losses.")
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1).")
    ordered = losses.sort(descending=True).values
    mass = (1 - alpha) * losses.numel()
    nearest = round(mass)
    if abs(mass - nearest) < 1e-10:
        mass = float(nearest)
    whole = int(math.floor(mass))
    fraction = mass - whole
    total = ordered[:whole].sum()
    if fraction > 0:
        total = total + fraction * ordered[whole]
    return total / mass


def _validate_arrays(S, delta, L):
    if delta.shape != (S.shape[0], S.shape[1] - 1) or L.shape != (S.shape[0],):
        raise AssertionError("Expected holdings (B,N) and losses (B,).")
    if not torch.isfinite(delta).all() or not torch.isfinite(L).all():
        raise AssertionError("Non-finite holdings or losses.")
    if (delta < 0).any() or (delta > 1).any():
        raise AssertionError("Holdings must lie in [0,1].")


def metrics(S, delta, L, cfg, cost_rate=None):
    """Financial metrics in currency units, except turnover (shares)."""
    _validate_arrays(S, delta, L)
    c = cfg.c if cost_rate is None else cost_rate
    zero = torch.zeros_like(delta[:, :1])
    trades = torch.diff(torch.cat((zero, delta, zero), dim=1), dim=1)
    return {"mean_loss": L.mean().item(), "sd_loss": L.std(unbiased=True).item(),
            "rmse": L.square().mean().sqrt().item(),
            "es95": empirical_es(L, cfg.alpha).item(),
            "turnover": trades.abs().sum(dim=1).mean().item(),
            "mean_cost": (c * S * trades.abs()).sum(dim=1).mean().item()}


def train_es(cfg, rollout, terminal_loss, es_objective, *, width=32, depth=2,
             verbose=True):
    """Supplied training loop; validation selects theta by empirical ES.

    The initial network is also a checkpoint candidate. Architecture runs use
    the same seed, independent training-path stream, and validation paths.
    Equal update budgets do not imply equal computing time.
    """
    if cfg.n_steps < 1 or cfg.validate_every < 1:
        raise ValueError("Positive training and validation intervals required.")
    policy = make_policy(width, depth, cfg.seed)
    eta = nn.Parameter(torch.tensor(0.0))  # normalized loss units
    opt = torch.optim.Adam([*policy.parameters(), eta], lr=cfg.learning_rate)
    train_gen = torch.Generator().manual_seed(cfg.seed + 10000)
    Sval = gbm_paths(cfg, shocks=fixed_shocks(cfg.n_validation, cfg.N,
                                           cfg.validation_seed))
    best_value, best_state, best_eta, best_step = math.inf, None, None, None
    history = []
    start = time.perf_counter()

    def validate(step, train_value):
        nonlocal best_value, best_state, best_eta, best_step
        policy.eval()
        with torch.no_grad():
            delta = rollout(policy, Sval, cfg)
            L = terminal_loss(Sval, delta, cfg)
            _validate_arrays(Sval, delta, L)
            value = empirical_es(L / cfg.S0, cfg.alpha).item()
        if not math.isfinite(value):
            raise RuntimeError("Non-finite validation ES.")
        if value < best_value:
            best_value = value
            best_state = copy.deepcopy(policy.state_dict())
            best_eta = eta.detach().item()
            best_step = step
        history.append({"step": step, "train_oce": train_value,
                        "validation_es": value, "eta": eta.detach().item(),
                        "elapsed_seconds": time.perf_counter() - start})
        if verbose:
            print(f"step {step:4d}: validation ES = {cfg.S0*value:.4f}; "
                  f"eta (normalized) = {eta.detach().item():.5f}")

    validate(0, None)
    for step in range(1, cfg.n_steps + 1):
        policy.train()
        S = gbm_paths(cfg, n_paths=cfg.batch_size, generator=train_gen)
        opt.zero_grad(set_to_none=True)
        delta = rollout(policy, S, cfg)
        L = terminal_loss(S, delta, cfg)
        _validate_arrays(S, delta, L)
        objective = es_objective(L / cfg.S0, eta, cfg.alpha)
        if objective.ndim != 0 or not torch.isfinite(objective):
            raise RuntimeError("The objective must be a finite scalar.")
        objective.backward()  # one reverse pass through the entire rollout
        if eta.grad is None or not torch.isfinite(eta.grad):
            raise RuntimeError("The ES objective must differentiate eta.")
        if any(p.grad is None or not torch.isfinite(p.grad).all()
               for p in policy.parameters()):
            raise RuntimeError("Missing or non-finite policy gradient.")
        opt.step()
        if step % cfg.validate_every == 0 or step == cfg.n_steps:
            validate(step, objective.detach().item())
    policy.load_state_dict(best_state)
    policy.eval()
    return {"policy": policy, "eta": best_eta, "history": history,
            "best_step": best_step, "validation_es": best_value * cfg.S0,
            "width": width, "depth": depth,
            "n_parameters": sum(p.numel() for p in policy.parameters()),
            "training_seconds": time.perf_counter() - start, "config": asdict(cfg)}


def evaluate(S, cfg, rollout, terminal_loss, policies, *, cost_rate=None,
             scenario="baseline"):
    """All methods receive identical paths, premium, and cost convention."""
    rows, losses = [], {}
    with torch.no_grad():
        holdings = {"Zero hedge": torch.zeros_like(S[:, :-1]),
                    f"Delta (fixed {100 * cfg.sigma:g}% volatility)": delta_holdings(S, cfg)}
        for name, policy in policies.items():
            policy.eval()
            holdings[name] = rollout(policy, S, cfg)
        for name, delta in holdings.items():
            L = terminal_loss(S, delta, cfg, cost_rate=cost_rate)
            rows.append({"scenario": scenario, "method": name,
                         **metrics(S, delta, L, cfg, cost_rate)})
            losses[name] = L.detach().clone()
    return rows, losses


def stress_experiment(kind, cfg, rollout, terminal_loss, policy, *, levels=None):
    """Supplied one-factor evaluation sweeps; no training or policy selection.

    Select the experiment BEFORE examining test results. Edit only the declared
    levels if proposing different values. Premium and delta volatility remain
    at their original values. The same Gaussian shocks couple all scenarios.
    """
    shocks = fixed_shocks(cfg.n_test, cfg.N, cfg.test_seed)
    if kind == "volatility":
        levels = [.10, .15, .20, .25, .30] if levels is None else levels
        scenarios = [(f"sigma={v:.1%}", v, cfg.c) for v in levels]
    elif kind == "cost":
        levels = [0.0, .0005, .001, .002] if levels is None else levels
        if min(levels) < 0:
            raise ValueError("Transaction-cost rates must be nonnegative.")
        scenarios = [(f"cost={c*10000:g} bps", cfg.sigma, c) for c in levels]
    elif kind == "volatility_timing":
        if cfg.N % 2:
            raise ValueError("Timing example requires an even number of intervals.")
        high_low = [.30] * (cfg.N // 2) + [.10] * (cfg.N // 2)
        # The last two scenarios have equal integrated variance and hence the
        # same terminal-stock distribution, but not identical paired endpoints.
        scenarios = [("20% throughout", .20, cfg.c),
                     ("30% then 10%", high_low, cfg.c),
                     ("10% then 30%", list(reversed(high_low)), cfg.c)]
    else:
        raise ValueError("Choose volatility, cost, or volatility_timing.")
    rows = []
    for label, volatility, cost_rate in scenarios:
        S = gbm_paths(cfg, shocks=shocks, volatility=volatility)
        part, _ = evaluate(S, cfg, rollout, terminal_loss, {"Neural": policy},
                           cost_rate=cost_rate, scenario=label)
        rows.extend(part)
    return rows


def save_result(result, path):
    """Save a portable CPU state_dict plus configuration and provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in result.items() if k != "policy"}
    payload["state_dict"] = result["policy"].state_dict()
    payload["environment"] = environment_info()
    torch.save(payload, path)


def load_result(path):
    """Load only a locally trusted assignment checkpoint, using weights_only."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    policy = make_policy(payload["width"], payload["depth"], payload["config"]["seed"])
    policy.load_state_dict(payload.pop("state_dict"))
    policy.eval()
    return {**payload, "policy": policy}


def save_table(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    print(f"{'scenario':20s} {'method':29s} {'mean':>8s} {'SD':>8s} "
          f"{'RMSE':>8s} {'ES95':>8s} {'turnover':>9s} {'cost':>8s}")
    for r in rows:
        print(f"{r['scenario']:20s} {r['method']:29s} "
              + " ".join(f"{r[k]:8.4f}" for k in
                         ["mean_loss", "sd_loss", "rmse", "es95", "turnover", "mean_cost"]))


def plot_training(result):
    import matplotlib.pyplot as plt
    h = result["history"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot([r["step"] for r in h[1:]], [r["train_oce"] for r in h[1:]],
            "o--", color="0.5", label="Training batch OCE")
    ax.plot([r["step"] for r in h], [r["validation_es"] for r in h],
            "s-", color="#174b80", label="Validation empirical ES")
    ax.axvline(result["best_step"], color="0.25", linestyle=":", label="Selected checkpoint")
    ax.set(xlabel="Optimizer updates", ylabel="Normalized loss units")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_experiment(rows, include_zero=False):
    import matplotlib.pyplot as plt
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    if not include_zero:
        methods = [m for m in methods if m != "Zero hedge"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    colors = ["0.15", "#174b80", "#965a16", "0.6"]
    styles = ["o:", "s--", "D-", "^-"]
    for j, method in enumerate(methods):
        subset = [next(r for r in rows if r["scenario"] == s and r["method"] == method)
                  for s in scenarios]
        for ax, key in zip(axes, ["es95", "mean_cost"]):
            ax.plot(range(len(scenarios)), [r[key] for r in subset],
                    styles[j % len(styles)], color=colors[j % len(colors)], label=method)
            ax.set_xticks(range(len(scenarios)), scenarios, rotation=20, ha="right")
    axes[0].set_ylabel("95% ES (currency units)")
    axes[1].set_ylabel("Mean transaction cost (currency units)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_architecture(rows):
    import matplotlib.pyplot as plt
    rows = [r for r in rows if r["method"] != "Zero hedge"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, key, label in zip(axes, ["es95", "mean_cost"],
                              ["95% ES", "Mean transaction cost"]):
        ax.bar([r["method"] for r in rows], [r[key] for r in rows],
               color=["0.7", "0.25", "#174b80", "#965a16"][:len(rows)])
        ax.set_ylabel(label + " (currency units)")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
    fig.tight_layout()
    return fig


def run_checks(rollout, terminal_loss, es_objective, cfg):
    """Independent analytic checks of accounting, causality, and gradients.

    These checks are evidence about specified cases, not a proof of correctness.
    Read each check and explain what error it is intended to detect.
    """
    short = replace(cfg, N=2)
    S = torch.tensor([[100., 110., 105.], [100., 95., 99.]], dtype=torch.float64)
    zero = torch.zeros((2, 2), dtype=torch.float64)
    payoff = (S[:, -1] - cfg.K).clamp_min(0)
    torch.testing.assert_close(terminal_loss(S, zero, short), payoff - cfg.premium)
    a = .4
    constant = torch.full_like(zero, a)
    expected = payoff - cfg.premium - a*(S[:, -1]-S[:, 0]) + cfg.c*a*(S[:, 0]+S[:, -1])
    torch.testing.assert_close(terminal_loss(S, constant, short), expected)
    known = terminal_loss(S[:1], torch.tensor([[.4, .6]], dtype=S.dtype), short)
    torch.testing.assert_close(known, torch.tensor([4+125*cfg.c-cfg.premium], dtype=S.dtype))
    # Independent cash ledger: explicitly buy/sell shares and pay each cost.
    positions = torch.tensor([[.4, .6], [.7, .2]], dtype=S.dtype)
    cash = torch.full((2,), cfg.premium, dtype=S.dtype)
    previous = torch.zeros(2, dtype=S.dtype)
    for k in range(3):
        current = positions[:, k] if k < 2 else torch.zeros_like(previous)
        trade = current - previous
        cash = cash - S[:, k]*trade - cfg.c*S[:, k]*trade.abs()
        previous = current
    torch.testing.assert_close(terminal_loss(S, positions, short), payoff-cash)

    class TestPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(.2, dtype=torch.float64))
            self.b = nn.Parameter(torch.tensor(.4, dtype=torch.float64))
        def forward(self, x):
            return torch.sigmoid(self.a + self.b*x[:, 2] + .1*x[:, 1]).unsqueeze(-1)

    policy = TestPolicy()
    d = rollout(policy, S, short)
    _validate_arrays(S, d, terminal_loss(S, d, short))
    assert rollout(policy, S[:1], short).shape == (1, 2)
    for k in range(short.N):
        changed = S.clone()
        changed[:, k+1:] *= 1.15
        torch.testing.assert_close(rollout(policy, changed, short)[:, :k+1], d[:, :k+1])

    # Directional derivative with no transaction-cost kink and fixed paths.
    no_cost = replace(short, c=0.0)
    objective = terminal_loss(S, rollout(policy, S, no_cost), no_cost).square().mean()
    gradients = torch.autograd.grad(objective, [policy.a, policy.b])
    direction = [.6, .8]
    ad = sum(g.item()*v for g, v in zip(gradients, direction))
    original = [p.detach().clone() for p in [policy.a, policy.b]]
    def displaced(amount):
        with torch.no_grad():
            for p, value, v in zip([policy.a, policy.b], original, direction):
                p.copy_(value + amount*v)
            return terminal_loss(S, rollout(policy, S, no_cost), no_cost).square().mean().item()
    errors = []
    for eps in [1e-3, 1e-4, 1e-5]:
        fd = (displaced(eps)-displaced(-eps))/(2*eps)
        errors.append(abs(fd-ad)/max(1.0, abs(ad)))
    displaced(0)
    assert min(errors) < 1e-6, f"Gradient check failed: relative errors {errors}"

    ell = torch.tensor([-.02, .01, .03], dtype=torch.float64, requires_grad=True)
    eta = torch.tensor(.015, dtype=torch.float64, requires_grad=True)
    H = es_objective(ell, eta, .95)
    gl, ge = torch.autograd.grad(H, [ell, eta])
    torch.testing.assert_close(H, torch.tensor(.115, dtype=torch.float64))
    torch.testing.assert_close(gl, torch.tensor([0., 0., 20/3], dtype=torch.float64))
    torch.testing.assert_close(ge, torch.tensor(1-20/3, dtype=torch.float64))
    report = {"zero_hedge": "passed", "constant_holding": "passed",
              "opening_and_liquidation": "passed", "cash_ledger": "passed",
              "causality": "passed", "batch_size_one": "passed",
              "policy_gradient_relative_errors": errors, "ES_value_and_gradients": "passed"}
    print(json.dumps(report, indent=2))
    return report
