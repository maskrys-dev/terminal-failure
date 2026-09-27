# Classwise robustness audit

All 200 Tiny ImageNet classes are retained. The primary classwise drop is the three-seed mean T=8 accuracy minus the three-seed mean T=32 accuracy.

| Model | Median drop | IQR | Macro mean | Lower at T=32 | Drop >=2 pp |
|---|---:|---:|---:|---:|---:|
| e16_t8 | 2.67 pp | [0.00, 8.67] pp | 5.11 pp | 144/200 (72.0%) | 124/200 (62.0%) |
| e17_t8 | 2.00 pp | [0.67, 4.00] pp | 2.42 pp | 153/200 (76.5%) | 105/200 (52.5%) |
