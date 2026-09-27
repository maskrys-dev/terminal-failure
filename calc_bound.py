import json, numpy as np

# Get observed early-window advantage at rho=1.5 from main MNIST sweep
res = json.loads(open('results/c1_c2/results.json').read())
print("Keys sample:", list(res[0].keys()) if isinstance(res, list) else list(res.keys()))

if isinstance(res, list):
    recs_15 = [r for r in res if abs(r.get('spectral_radius', r.get('rho', 0)) - 1.5) < 0.02]
    print(f"Records at rho~1.5: {len(recs_15)}")
    if recs_15:
        for k, v in recs_15[0].items():
            if not isinstance(v, list):
                print(f"  {k}: {v}")
        # Compute mean acc_final and acc_early
        af = np.mean([r.get('acc_final', r.get('final', np.nan)) for r in recs_15])
        ae = np.mean([r.get('acc_early_window', r.get('early',np.nan)) for r in recs_15])
        print(f"\nMean at rho=1.5: final={af:.3f}  early={ae:.3f}  gap={100*(ae-af):.1f}pp")

# For bound: G <= t0 + log(R_max)/log(kappa)
# From paper: R_max = norm(z_T)/norm(z_1) threshold ~ 1.5, kappa = rho
rho = 1.5; R_max = 1.5; t0 = 5  # conservative t0
bound = t0 + np.log(R_max) / np.log(rho)
print(f"\nProp 4 bound at rho=1.5: G <= {t0} + log({R_max})/log({rho}) = {bound:.1f} steps")
print(f"Actual T=40, so observed G is the fraction where accuracy is still good")
print(f"If accuracy stays good until t~20, observed G~20 vs bound~{bound:.0f}")
