import json
from collections import defaultdict

import numpy as np
from scipy import stats


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def adaptive_acc(row):
    if "acc_adaptive" in row:
        return row["acc_adaptive"]
    adaptive = row.get("adaptive", {})
    if "adaptive_1.5" in adaptive:
        return adaptive["adaptive_1.5"]["accuracy"]
    vals = [entry["accuracy"] for entry in adaptive.values() if isinstance(entry, dict) and "accuracy" in entry]
    if vals:
        return max(vals)
    raise KeyError("adaptive accuracy missing")


def best_oracle_acc(row):
    for key in ("acc_d1_best", "best_d1_acc", "acc_best_timestep"):
        if key in row:
            return row[key]
    raise KeyError("best-oracle accuracy missing")


def print_wilcoxon_table():
    data = load_json("results/overnight_10seed/results.json")
    rhos = sorted(set(row["spectral_radius"] for row in data))

    print("=== WILCOXON: GRACE vs each strategy (one-sided, greater) ===")
    print(f"{'rho':>5}  {'vs Final p':>12}  {'vs Early p':>12}  {'vs Adaptive p':>14}  {'W(Final)':>10}  {'W(Early)':>10}  {'W(Adap)':>10}")
    for rho in rhos:
        rows = [row for row in data if row["spectral_radius"] == rho]
        grace = np.array([best_oracle_acc(row) for row in rows])
        final = np.array([row["acc_final"] for row in rows])
        early = np.array([row["acc_early_window"] for row in rows])
        adap = np.array([adaptive_acc(row) for row in rows])

        def ws(a, b):
            diff = a - b
            if np.all(diff == 0):
                return "n/a", "n/a"
            try:
                stat, p = stats.wilcoxon(a, b, alternative="greater")
                return f"{p:.4f}", f"{stat:.0f}"
            except Exception:
                return "n/a", "n/a"

        pf, wf = ws(grace, final)
        pe, we = ws(grace, early)
        pa, wa = ws(grace, adap)
        print(f"{rho:>5.2f}  {pf:>12}  {pe:>12}  {pa:>14}  {wf:>10}  {we:>10}  {wa:>10}")


def print_fashion_table():
    print("\n=== FASHION MNIST ===")
    try:
        data = load_json("results/a3_fashion/results.json")
        rhos = sorted(set(row["spectral_radius"] for row in data))
        print(f"{'rho':>5}  {'Final':>8}  {'Early':>8}  {'Adaptive':>10}  {'Best-t':>8}")
        for rho in rhos:
            rows = [row for row in data if row["spectral_radius"] == rho]
            fin = np.mean([row["acc_final"] for row in rows])
            earl = np.mean([row["acc_early_window"] for row in rows])
            adap = np.mean([adaptive_acc(row) for row in rows])
            best = np.mean([row["acc_best_timestep"] for row in rows])
            print(f"{rho:>5.2f}  {fin:>8.4f}  {earl:>8.4f}  {adap:>10.4f}  {best:>8.4f}")
    except Exception as exc:
        print(f"Error: {exc}")


def print_esn_d1_table():
    print("\n=== ESN D1 SENSITIVITY ===")
    try:
        data = load_json("results/d1_esn_tune/results.json")
        print("Keys in first record:", list(data[0].keys()))

        configs = sorted({cfg for row in data for cfg in row.get("d1_accs", {}).keys()})
        if configs:
            print(f"configs={configs}")
            for cfg in configs:
                rho_accs = defaultdict(list)
                for row in data:
                    if cfg in row.get("d1_accs", {}):
                        rho_accs[row["spectral_radius"]].append(row["d1_accs"][cfg])
                print(f"\nConfig: {cfg}")
                for rho in sorted(rho_accs):
                    vals = rho_accs[rho]
                    print(f"  rho={rho:.2f}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
            return

        systems = sorted(set(row.get("system", "?") for row in data))
        configs = sorted(set(row.get("d1_config", "?") for row in data))
        print(f"systems={systems}, configs={configs}")
        for cfg in configs:
            rows = [row for row in data if row.get("d1_config") == cfg]
            rho_accs = defaultdict(list)
            for row in rows:
                rho_accs[row["spectral_radius"]].append(row.get("d1_acc", row.get("acc_d1", row.get("best_d1_acc", 0))))
            print(f"\nConfig: {cfg}")
            for rho in sorted(rho_accs):
                vals = rho_accs[rho]
                print(f"  rho={rho:.2f}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    except Exception as exc:
        print(f"Error: {exc}")
        try:
            data = load_json("results/d1_cross_system/results.json")
            configs = sorted({cfg for row in data for cfg in row.get("d1_accs", {}).keys()})
            print("D1 configs seen:", configs)
            for cfg in configs:
                rho_accs = defaultdict(list)
                for row in data:
                    if cfg in row.get("d1_accs", {}):
                        rho_accs[row["spectral_radius"]].append(row["d1_accs"][cfg])
                print(f"\nConfig {cfg}:")
                for rho in sorted(rho_accs):
                    vals = rho_accs[rho]
                    print(f"  rho={rho:.2f}: mean={np.mean(vals):.4f}")
        except Exception as fallback_exc:
            print(f"Also failed: {fallback_exc}")


if __name__ == "__main__":
    print_wilcoxon_table()
    print_fashion_table()
    print_esn_d1_table()
