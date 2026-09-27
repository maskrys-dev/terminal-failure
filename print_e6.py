import json, numpy as np

for cond, label in [('cond_A','A'), ('cond_B1','B1'), ('cond_B2','B2')]:
    d = json.load(open(f'results/e6_trained_at_op/{cond}.json'))
    s = d['summary']
    print(f'Condition {label}:')
    print(f'  lam    final          early          grace          adaptive')
    for lam in [0.85, 1.20, 1.30, 1.50, 2.00]:
        k = str(lam)
        r = s[k]
        fin = f"{r['final']['mean']:.3f}+-{r['final']['std']:.3f}"
        ear = f"{r['early']['mean']:.3f}+-{r['early']['std']:.3f}"
        grc = f"{r['grace']['mean']:.3f}+-{r['grace']['std']:.3f}"
        adp = f"{r['adaptive']['mean']:.3f}+-{r['adaptive']['std']:.3f}"
        print(f'  {lam:.2f}  {fin:<14}  {ear:<14}  {grc:<14}  {adp}')
    print()
