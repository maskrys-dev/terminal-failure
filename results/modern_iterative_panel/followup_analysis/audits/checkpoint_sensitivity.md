# Checkpoint sensitivity audit

Only saved checkpoint states were evaluated. The preregistered selected checkpoint remains the headline checkpoint.

| Model | Seed | Checkpoint | Epoch | T=8 | T=32 | Drop |
|---|---:|---|---:|---:|---:|---:|
| e16_t8 | 101 | selected | 3 | 80.98% | 76.05% | +4.93 pp |
| e16_t8 | 101 | final_epoch | 30 | 80.71% | 80.12% | +0.59 pp |
| e16_t8 | 102 | selected | 4 | 80.82% | 74.69% | +6.13 pp |
| e16_t8 | 102 | final_epoch | 30 | 80.62% | 79.96% | +0.66 pp |
| e16_t8 | 103 | selected | 3 | 80.82% | 76.55% | +4.27 pp |
| e16_t8 | 103 | final_epoch | 30 | 80.67% | 79.99% | +0.68 pp |
| e17_t8 | 101 | selected | 5 | 79.53% | 76.83% | +2.70 pp |
| e17_t8 | 101 | saved_rank_2 | 4 | 78.91% | 76.08% | +2.83 pp |
| e17_t8 | 101 | saved_rank_3 | 3 | 78.52% | 75.03% | +3.49 pp |
| e17_t8 | 102 | selected | 7 | 79.61% | 77.49% | +2.12 pp |
| e17_t8 | 102 | saved_rank_2 | 5 | 79.31% | 76.76% | +2.55 pp |
| e17_t8 | 102 | saved_rank_3 | 4 | 79.27% | 76.28% | +2.99 pp |
| e17_t8 | 103 | selected | 5 | 79.72% | 77.27% | +2.45 pp |
| e17_t8 | 103 | saved_rank_2 | 4 | 79.30% | 76.63% | +2.67 pp |
| e17_t8 | 103 | saved_rank_3 | 3 | 78.38% | 75.06% | +3.32 pp |
