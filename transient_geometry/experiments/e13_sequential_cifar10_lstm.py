"""
E13: Sequential CIFAR-10 LSTM post-input settling check.

This is a reviewer-armour experiment rather than a core result. A small LSTM is
trained end-to-end on row-sequential CIFAR-10. After the full image sequence has
been processed, we run a post-input recurrent settling trajectory and compare
terminal, early-window, adaptive-stopping, and GRACE readouts across a test-time
recurrent-gain sweep.

The key design choice is that readouts are taken only after the whole image has
been seen. This avoids confounding "early readout" with missing input pixels.

Typical pilot:
    python -m transient_geometry.experiments.e13_sequential_cifar10_lstm --pilot

Full-ish local run:
    python -m transient_geometry.experiments.e13_sequential_cifar10_lstm

Outputs:
    results/e13_seq_cifar10_lstm/e13_seq_cifar10_lstm.json
    results/e13_seq_cifar10_lstm/e13_seq_cifar10_lstm.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results" / "e13_seq_cifar10_lstm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METHODS = ("final", "early", "adaptive", "grace")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true", help="Small quick run for signal/smoke testing.")
    p.add_argument("--sequence", choices=("rows", "pixels"), default="rows")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--lambda-train", type=float, default=1.0)
    p.add_argument("--lambda-test", type=str, default="0.85,1.0,1.2,1.4,1.6,1.8,2.0")
    p.add_argument("--settle-train", type=int, default=40)
    p.add_argument("--settle-test", type=int, default=40)
    p.add_argument("--early-k", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grace-tau", type=float, default=1.2)
    p.add_argument("--grace-alpha", type=float, default=10.0)
    p.add_argument("--stop-tau", type=float, default=1.5)
    p.add_argument("--download", action="store_true", help="Allow torchvision to download CIFAR-10.")
    p.add_argument("--resume", action="store_true", help="Reuse matching seeds already saved in the output JSON.")
    p.add_argument(
        "--per-timestep-audit",
        action="store_true",
        help="Also fit separate single-timestep probes for selected gains.",
    )
    p.add_argument("--audit-lambdas", type=str, default="1.0,1.6,2.0")
    p.add_argument("--audit-alpha", type=float, default=1.0)
    p.add_argument("--audit-threshold", type=float, default=0.40)
    args = p.parse_args()
    if args.pilot:
        args.n_train = min(args.n_train, 2000)
        args.n_test = min(args.n_test, 1000)
        args.epochs = min(args.epochs, 4)
        args.hidden_dim = min(args.hidden_dim, 96)
        args.seeds = args.seeds.split(",")[0]
        args.lambda_test = "0.85,1.0,1.3,1.7,2.0"
    return args


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cifar_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )


def stratified_indices(targets, n: int, seed: int) -> np.ndarray:
    targets = np.asarray(targets)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(splitter.split(np.zeros(len(targets)), targets))
    return idx.astype(int)


def load_tensor_split(ds, indices: np.ndarray, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(Subset(ds, indices.tolist()), batch_size=batch_size, shuffle=False, num_workers=0)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0).long()


def images_to_sequence(x: torch.Tensor, mode: str) -> torch.Tensor:
    # x: (N, 3, 32, 32)
    if mode == "rows":
        return x.permute(0, 2, 1, 3).reshape(x.shape[0], 32, 96)
    # Pixel-sequential CIFAR-10: (N, 1024, 3), row-major.
    return x.permute(0, 2, 3, 1).reshape(x.shape[0], 1024, 3)


def load_cifar_sequences(args: argparse.Namespace, seed: int):
    tf = cifar_transform()
    train_ds = datasets.CIFAR10(DATA_DIR, train=True, download=args.download, transform=tf)
    test_ds = datasets.CIFAR10(DATA_DIR, train=False, download=args.download, transform=tf)
    train_idx = stratified_indices(train_ds.targets, args.n_train, seed)
    test_idx = stratified_indices(test_ds.targets, args.n_test, seed)
    x_train, y_train = load_tensor_split(train_ds, train_idx, args.batch_size)
    x_test, y_test = load_tensor_split(test_ds, test_idx, args.batch_size)
    return (
        images_to_sequence(x_train, args.sequence),
        y_train,
        images_to_sequence(x_test, args.sequence),
        y_test,
    )


class GainLSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.x2g = nn.Linear(input_dim, 4 * hidden_dim)
        self.h2g = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        self.head = nn.Linear(hidden_dim, n_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.x2g.weight)
        nn.init.zeros_(self.x2g.bias)
        nn.init.orthogonal_(self.h2g.weight)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # Encourage retention at the start of training.
        with torch.no_grad():
            h = self.hidden_dim
            self.x2g.bias[h : 2 * h].fill_(1.0)

    def step(self, x_t: torch.Tensor, h: torch.Tensor, c: torch.Tensor, lam: float):
        gates = self.x2g(x_t) + lam * self.h2g(h)
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def rollout(self, x: torch.Tensor, settle_steps: int, lam: float, return_trajectory: bool = False):
        batch = x.shape[0]
        h = torch.zeros(batch, self.hidden_dim, device=x.device)
        c = torch.zeros(batch, self.hidden_dim, device=x.device)

        for t in range(x.shape[1]):
            h, c = self.step(x[:, t], h, c, lam)

        traj = [] if return_trajectory else None
        zero_x = torch.zeros(batch, x.shape[2], device=x.device)
        for _ in range(settle_steps):
            h, c = self.step(zero_x, h, c, lam)
            if return_trajectory:
                traj.append(h)

        if return_trajectory:
            return h, torch.stack(traj, dim=1)
        return h

    def forward(self, x: torch.Tensor, settle_steps: int, lam: float):
        h = self.rollout(x, settle_steps, lam, return_trajectory=False)
        return self.head(h)


def train_model(model, x_train, y_train, args, seed: int):
    model.train()
    x_train = x_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    n = x_train.shape[0]
    best_acc = 0.0
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=DEVICE)
        total_loss = 0.0
        total_correct = 0
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            opt.zero_grad(set_to_none=True)
            logits = model(xb, args.settle_train, args.lambda_train)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item()) * len(idx)
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        sched.step()
        acc = total_correct / n
        best_acc = max(best_acc, acc)
        print(
            f"    seed={seed} epoch={epoch + 1:02d}/{args.epochs} "
            f"loss={total_loss / n:.4f} train_acc={acc:.3f}"
        )
    return best_acc


@torch.no_grad()
def extract_trajectory(model, x: torch.Tensor, args, lam: float) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, x.shape[0], args.batch_size):
        xb = x[start : start + args.batch_size].to(DEVICE)
        _, traj = model.rollout(xb, args.settle_test, lam, return_trajectory=True)
        chunks.append(traj.float().cpu())
    return torch.cat(chunks, dim=0).numpy()


def readout_features(traj: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    # traj: (N, settle_T, H)
    final = traj[:, -1]
    early = traj[:, : min(args.early_k, traj.shape[1])].mean(axis=1)
    norms = np.linalg.norm(traj, axis=2)
    ratio = norms / np.maximum(norms[:, :1], 1e-10)

    crossed = ratio > args.stop_tau
    first = crossed.argmax(axis=1)
    never = ~crossed.any(axis=1)
    first[never] = traj.shape[1] - 1
    adaptive = traj[np.arange(traj.shape[0]), first]

    w_raw = np.exp(-args.grace_alpha * np.maximum(0.0, ratio - args.grace_tau))
    w = w_raw / np.maximum(w_raw.sum(axis=1, keepdims=True), 1e-10)
    grace = np.einsum("nt,nth->nh", w, traj)
    return {
        "final": final.astype(np.float32),
        "early": early.astype(np.float32),
        "adaptive": adaptive.astype(np.float32),
        "grace": grace.astype(np.float32),
    }


def score_features(train_feats, test_feats, y_train, y_test):
    scores = {}
    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()
    for method in METHODS:
        clf = RidgeClassifierCV(alphas=[0.1, 1.0, 10.0, 100.0])
        clf.fit(train_feats[method], y_train_np)
        scores[method] = float(clf.score(test_feats[method], y_test_np))
    return scores


def score_per_timestep(train_traj: np.ndarray, test_traj: np.ndarray, y_train, y_test, alpha: float):
    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()
    scores = []
    for t in range(train_traj.shape[1]):
        clf = RidgeClassifier(alpha=alpha)
        clf.fit(train_traj[:, t], y_train_np)
        scores.append(float(clf.score(test_traj[:, t], y_test_np)))
    return scores


def run_seed(seed: int, args: argparse.Namespace, lambda_tests: list[float]):
    seed_everything(seed)
    print(f"\nSeed {seed}: loading CIFAR-10 sequence data...")
    x_train, y_train, x_test, y_test = load_cifar_sequences(args, seed)
    input_dim = x_train.shape[2]
    print(f"  train={tuple(x_train.shape)} test={tuple(x_test.shape)} device={DEVICE}")
    model = GainLSTMClassifier(input_dim, args.hidden_dim).to(DEVICE)
    best_train_acc = train_model(model, x_train, y_train, args, seed)

    seed_results = {}
    audit_lambdas = {float(x) for x in args.audit_lambdas.split(",") if x.strip()}
    for lam in lambda_tests:
        t0 = time.time()
        train_traj = extract_trajectory(model, x_train, args, lam)
        test_traj = extract_trajectory(model, x_test, args, lam)
        train_feats = readout_features(train_traj, args)
        test_feats = readout_features(test_traj, args)
        scores = score_features(train_feats, test_feats, y_train, y_test)
        if args.per_timestep_audit and lam in audit_lambdas:
            scores["per_timestep_probe"] = score_per_timestep(
                train_traj,
                test_traj,
                y_train,
                y_test,
                args.audit_alpha,
            )
        seed_results[str(lam)] = scores
        print(
            f"  lambda={lam:.2f} final={scores['final']:.3f} "
            f"early={scores['early']:.3f} adaptive={scores['adaptive']:.3f} "
            f"grace={scores['grace']:.3f} ({time.time() - t0:.1f}s)"
        )

    return {"best_train_acc": float(best_train_acc), "lambda_results": seed_results}


def summarise(per_seed: dict, lambda_tests: list[float]):
    summary = {}
    for lam in lambda_tests:
        key = str(lam)
        summary[key] = {}
        for method in METHODS:
            vals = [per_seed[str(seed)]["lambda_results"][key][method] for seed in per_seed]
            summary[key][method] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return summary


def plot_summary(summary: dict, lambda_tests: list[float], stem: str = "e13_seq_cifar10_lstm") -> None:
    colors = {
        "final": "#d84a4a",
        "early": "#2f8f46",
        "adaptive": "#8162b4",
        "grace": "#2468b2",
    }
    labels = {
        "final": "Terminal",
        "early": "Early window",
        "adaptive": "Adaptive stop",
        "grace": "GRACE",
    }
    fig, ax = plt.subplots(figsize=(6.9, 4.0))
    for method in METHODS:
        mean = np.array([summary[str(lam)][method]["mean"] for lam in lambda_tests])
        std = np.array([summary[str(lam)][method]["std"] for lam in lambda_tests])
        ax.plot(lambda_tests, mean, marker="o", lw=2, color=colors[method], label=labels[method])
        ax.fill_between(lambda_tests, mean - std, mean + std, color=colors[method], alpha=0.14)
    ax.set_xlabel("Test-time recurrent gain")
    ax.set_ylabel("Linear-probe accuracy")
    ax.set_title("Sequential CIFAR-10 LSTM: post-input settling readouts")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=220)
    plt.close(fig)


def summarise_per_timestep(per_seed: dict, audit_lambdas: list[float]):
    summary = {}
    for lam in audit_lambdas:
        key = str(lam)
        curves = []
        for seed in per_seed:
            arr = per_seed[str(seed)]["lambda_results"].get(key, {}).get("per_timestep_probe")
            if arr is not None:
                curves.append(arr)
        if curves:
            arr = np.asarray(curves, dtype=float)
            summary[key] = {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
            }
    return summary


def prefix_length(curve: list[float], threshold: float) -> int:
    length = 0
    for idx, value in enumerate(curve, start=1):
        if value >= threshold:
            length = idx
        else:
            break
    return length


def plot_timestep_audit(per_timestep_summary: dict, audit_lambdas: list[float], threshold: float, stem: str) -> None:
    if not per_timestep_summary:
        return
    colors = ["#2468b2", "#b7791f", "#d84a4a", "#2f8f46"]
    fig, ax = plt.subplots(figsize=(6.9, 3.6))
    for color, lam in zip(colors, audit_lambdas):
        key = str(lam)
        if key not in per_timestep_summary:
            continue
        mean = np.asarray(per_timestep_summary[key]["mean"], dtype=float)
        std = np.asarray(per_timestep_summary[key]["std"], dtype=float)
        x = np.arange(1, len(mean) + 1)
        g_len = prefix_length(mean.tolist(), threshold)
        ax.plot(x, mean, lw=2, color=color, label=rf"$\lambda={lam:g}$, $G={g_len}$")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.14)
    ax.axhline(threshold, color="#444444", lw=1.2, ls="--", label=rf"$\alpha={threshold:.2f}$")
    ax.set_xlabel("Post-input settling timestep")
    ax.set_ylabel("Single-timestep probe accuracy")
    ax.set_title("Sequential CIFAR-10 LSTM: accuracy-defined prefix audit")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}_per_timestep.png", dpi=220)
    plt.close(fig)


def build_result(args, seeds, lambda_tests, per_seed, t_start):
    audit_lambdas = [float(x) for x in args.audit_lambdas.split(",") if x.strip()]
    timestep_summary = summarise_per_timestep(per_seed, audit_lambdas)
    return {
        "experiment": "e13_sequential_cifar10_lstm",
        "description": "Row/pixel sequential CIFAR-10 LSTM with post-input settling readouts.",
        "config": {
            "sequence": args.sequence,
            "hidden_dim": args.hidden_dim,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lambda_train": args.lambda_train,
            "lambda_test": lambda_tests,
            "settle_train": args.settle_train,
            "settle_test": args.settle_test,
            "early_k": args.early_k,
            "grace_tau": args.grace_tau,
            "grace_alpha": args.grace_alpha,
            "stop_tau": args.stop_tau,
            "seeds": seeds,
            "device": str(DEVICE),
            "pilot": bool(args.pilot),
            "per_timestep_audit": bool(args.per_timestep_audit),
            "audit_lambdas": audit_lambdas,
            "audit_alpha": args.audit_alpha,
            "audit_threshold": args.audit_threshold,
        },
        "per_seed": per_seed,
        "summary": summarise(per_seed, lambda_tests),
        "per_timestep_summary": timestep_summary,
        "runtime_seconds": time.time() - t_start,
        "complete": len(per_seed) == len(seeds),
    }


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    lambda_tests = [float(x) for x in args.lambda_test.split(",") if x.strip()]
    t_start = time.time()
    print(
        "E13 sequential CIFAR-10 LSTM | "
        f"sequence={args.sequence} hidden={args.hidden_dim} "
        f"train={args.n_train} test={args.n_test} epochs={args.epochs} "
        f"seeds={seeds} device={DEVICE}"
    )

    output_stem = "e13_seq_cifar10_lstm_timestep_audit" if args.per_timestep_audit else "e13_seq_cifar10_lstm"
    out_json = OUT_DIR / f"{output_stem}.json"
    per_seed = {}
    if args.resume and out_json.exists():
        existing = json.loads(out_json.read_text())
        existing_cfg = existing.get("config", {})
        compatible = (
            existing_cfg.get("sequence") == args.sequence
            and existing_cfg.get("hidden_dim") == args.hidden_dim
            and existing_cfg.get("n_train") == args.n_train
            and existing_cfg.get("n_test") == args.n_test
            and existing_cfg.get("epochs") == args.epochs
            and existing_cfg.get("batch_size") == args.batch_size
            and existing_cfg.get("lambda_train") == args.lambda_train
            and existing_cfg.get("lambda_test") == lambda_tests
            and existing_cfg.get("settle_train") == args.settle_train
            and existing_cfg.get("settle_test") == args.settle_test
            and existing_cfg.get("per_timestep_audit") == bool(args.per_timestep_audit)
            and existing_cfg.get("audit_lambdas") == [float(x) for x in args.audit_lambdas.split(",") if x.strip()]
            and existing_cfg.get("audit_alpha") == args.audit_alpha
            and existing_cfg.get("audit_threshold", args.audit_threshold) == args.audit_threshold
        )
        if compatible:
            per_seed.update(existing.get("per_seed", {}))
            print(f"Resuming from {out_json}: loaded seeds {sorted(per_seed)}")
        else:
            print(f"Ignoring existing {out_json}: config does not match current run.")

    for seed in seeds:
        if str(seed) in per_seed:
            print(f"\nSeed {seed}: already present, skipping.")
            continue
        per_seed[str(seed)] = run_seed(seed, args, lambda_tests)
        out_json.write_text(json.dumps(build_result(args, seeds, lambda_tests, per_seed, t_start), indent=2))
        print(f"  partial save: {out_json}")

    summary = summarise(per_seed, lambda_tests)
    result = build_result(args, seeds, lambda_tests, per_seed, t_start)
    out_json.write_text(json.dumps(result, indent=2))
    plot_summary(summary, lambda_tests, output_stem)
    if args.per_timestep_audit:
        plot_timestep_audit(
            result["per_timestep_summary"],
            result["config"]["audit_lambdas"],
            args.audit_threshold,
            output_stem,
        )
    print(f"\nSaved {out_json}")
    print(f"Saved {OUT_DIR / (output_stem + '.png')}")
    print("Summary:")
    for lam in lambda_tests:
        row = summary[str(lam)]
        print(
            f"  lambda={lam:.2f} final={row['final']['mean']:.3f} "
            f"early={row['early']['mean']:.3f} adaptive={row['adaptive']['mean']:.3f} "
            f"grace={row['grace']['mean']:.3f}"
        )


if __name__ == "__main__":
    main()
