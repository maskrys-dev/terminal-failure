# Model and data scale audit

| Item | E16 SwiGLU | E17 Attention |
|---|---:|---:|
| Dataset | Tiny ImageNet-200 | Tiny ImageNet-200 |
| Training images | 100000 | 100000 |
| Validation images | 10000 | 10000 |
| Classes | 200 | 200 |
| Frozen backbone | ConvNeXt-Tiny | ConvNeXt-Tiny |
| Input representation | pooled 768-d | 49 x 768 spatial tokens |
| Recursive state/model width | 512 | 256 |
| Intermediate width | 2048 | 1024 |
| Attention heads | N/A | 8 |
| Trainable head parameters | 3648712 | 1301960 |
| Executed frozen backbone parameters | 27820128 | 27820128 |
| Executed total system parameters | 31468840 | 29122088 |
| Training horizons | 8 and 32 | 8 and 32 |
| Mean terminal accuracy at trained horizon (T_train=8 / 32) | 80.87 +/- 0.09% / 80.84 +/- 0.15% | 79.62 +/- 0.10% / 79.16 +/- 0.22% |
| Dominant recurrent FLOPs / step | 6291456 | 107517600 |

Recurrent FLOPs use one multiply-add = two FLOPs and exclude the frozen backbone, one-time input projection, final classifier, LayerNorm, and lower-order elementwise costs.
