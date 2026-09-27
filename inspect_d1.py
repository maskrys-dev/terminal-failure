import json
records = json.loads(open('results/d1_cross_system/results.json').read())

for sys in set(r.get('system','?') for r in records):
    samples = [r for r in records if r.get('system') == sys]
    sample = samples[0]
    print(f"System: {sys}  ({len(samples)} records)")
    print(f"  Keys: {list(sample.keys())}")
    sr = sample.get('spectral_radius')
    af = sample.get('acc_final')
    ae = sample.get('acc_early_window', sample.get('acc_early', 'MISSING'))
    print(f"  Sample: sr={sr}  acc_final={af}  acc_early={ae}")
    radii = sorted(set(round(r.get('spectral_radius',0),2) for r in samples))
    print(f"  Radii: {radii}")
    print()
