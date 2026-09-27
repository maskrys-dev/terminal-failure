# Modern Iterative Architecture Panel

**Attention classification:** endpoint_invalidity.

E16 remains the locked headline causal horizon result. E17 adds a Universal-Transformer-style shared-attention system as architectural breadth; the spatial-token and pooled-feature systems are compared as tested systems rather than as a controlled isolation of attention.

## Final architecture panel

| Model | Family | T_train | Terminal T=8 | Terminal T=32 | Early T=32 | Best steps | Classification |
|---|---|---:|---:|---:|---:|:---:|---|
| SwiGLU T_train=8 | residual_recursive | 8 | 80.87% +/- 0.09% | 75.76% +/- 0.96% | 81.22% +/- 0.05% | 4/4/5 | endpoint_invalidity |
| SwiGLU T_train=32 | residual_recursive | 32 | 80.43% +/- 0.05% | 80.84% +/- 0.15% | 79.72% +/- 0.14% | 17/24/22 | matched_horizon_control |
| Attention T_train=8 | recursive_attention | 8 | 79.62% +/- 0.08% | 77.20% +/- 0.33% | 79.88% +/- 0.11% | 4/5/4 | endpoint_invalidity |
| Attention T_train=32 | recursive_attention | 32 | 79.19% +/- 0.11% | 79.15% +/- 0.22% | 78.58% +/- 0.29% | 23/14/12 | matched_horizon_control |

## Rebuttal-ready summary

Endpoint degradation under excess test-time recurrence appears in both the residual SwiGLU and shared-attention weight-tied recursive systems tested here.

All values are individual-seed-preserving means +/- sample standard deviations across seeds 101--103. No significance test is used. GRACE remains secondary and was not tuned.

## Phase status

- `locked_e14_e15_e16_audit`: complete_pass.
- `attention_protocol_and_configs`: complete_frozen_before_smoke.
- `spatial_feature_cache`: complete_validated.
- `attention_t8_engineering_gate`: complete_pass.
- `attention_t8_scientific_seeds`: complete_101_102_103.
- `attention_t8_probe_and_classification`: complete.
- `attention_t32_trigger`: passed_and_control_complete.
- `optional_deq`: skipped_optional_conditions_not_verified.
- `panel_aggregation`: complete.

## Protocol deviations

- Git commit unavailable; exact source and artifact hashes are recorded. No scientific protocol deviation occurred.
