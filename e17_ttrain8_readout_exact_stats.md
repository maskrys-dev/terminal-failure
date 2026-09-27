# E17 \(T_{\mathrm{train}}=8\), \(T_{\mathrm{test}}=32\) exact readout statistics

These values were extracted directly from the frozen seed-level
`readout_metrics.json` artifacts. No evaluation was rerun and no source
artifact was modified.

Sample standard deviations use \(n-1\) in the denominator (`ddof=1`).

| Readout | Seed 101 | Seed 102 | Seed 103 | Arithmetic mean | Sample SD |
|---|---:|---:|---:|---:|---:|
| Adaptive stopping | 79.39% | 79.21% | 79.60% | 79.400000% | 0.195192% |
| GRACE | 77.47% | 76.77% | 77.23% | 77.156667% | 0.355715% |
| Uniform averaging | 78.27% | 78.64% | 78.55% | 78.486667% | 0.192959% |

## Verification

- Adaptive stopping mean: 79.400000%, reproducing the previously reported
  approximately 79.40%.
- GRACE mean: 77.156667%, reproducing the previously reported approximately
  77.16%.
- Uniform averaging mean: 78.486667%, reproducing the previously reported
  approximately 78.49%.

## Frozen source artifacts

1. Seed 101  
   Path: `C:\Users\maskr\Documents\NeurIPS26\Terminal Failure\results\modern_iterative_panel\attention_t8\seed_101\readout_metrics.json`  
   SHA-256: `7383a457f2c4c46524bd73e0ef2ea5f8622f3f98fbe790942465014e409a9382`
2. Seed 102  
   Path: `C:\Users\maskr\Documents\NeurIPS26\Terminal Failure\results\modern_iterative_panel\attention_t8\seed_102\readout_metrics.json`  
   SHA-256: `17fd87ba4159565212ad8195c943ea794fbff444950c3915ddfa84ac04d766b5`
3. Seed 103  
   Path: `C:\Users\maskr\Documents\NeurIPS26\Terminal Failure\results\modern_iterative_panel\attention_t8\seed_103\readout_metrics.json`  
   SHA-256: `dedc6325b674a69498eaf5b8337d797f1d8006408a3f0b0492f3eee920e74bfc`
