# Provenance audit

**Status:** `pass`. Seeds: 101, 102, 103.

## Protocol hashes

- `e14`: `10f92ca12272cfc411c8b3919a4023952c405fdddba2e232a61f431bc18b18de` (Rebuttal\Terminal_Failure_Final_Recursive_Experiment_Protocol.md)
- `e15`: `2807f091fd569f2adc144ab1d67be789edfa9e5414c9105c7ad63b8c9429fda9` (Rebuttal\Terminal_Failure_E15_Aligned_Horizon_Protocol.md)
- `e16`: `9cc8f8204b2fb92cb9e42c0f73ba172a3ba994264d4fe80596d35f00c30c9c28` (Rebuttal\Terminal_Failure_E16_Paired_Horizon_Protocol.md)
- `e17`: `d626081b7c81396fbf00377a37eedbc6bcce5096083345f898c92dc42ae19e42` (results\modern_iterative_panel\protocol\PANEL_PROTOCOL.md)
- `followup`: `1ee5276d886d9d7477f7855a16f8672697d20e729119ce4965e592d113f45b9d` (results\modern_iterative_panel\followup_analysis\FOLLOWUP_ANALYSIS_PROTOCOL.md)

## Config hashes

- `e14`: `093e28a08fbc0d90f666bea8d0107b8881da2f2a448a1f2cd200647e4a9a316e` (results\rebuttal_recursive_tinyimagenet\config_frozen.json)
- `e15`: `48e0405e2e2ff59cfa348e3f9958b7699f1ec0f2440ac618e8f0350943f25f8f` (results\rebuttal_recursive_tinyimagenet_aligned_t32\config_frozen.json)
- `e16_panel`: `e637a6854ac8b990d5a4822b4e7beece1b88226839b88360a86b84c89b60b605` (results\rebuttal_recursive_tinyimagenet_paired_horizon\config_frozen.json)
- `e16_t8`: `6bdb5bdcc994f8e7f145042f6f2e5f5b85b0cb73c99dbeb740e13ead80084741` (results\rebuttal_recursive_tinyimagenet_e16_ttrain8\config_frozen.json)
- `e16_t32`: `33a36523d381d7996f5e4f11a5d3e129f9d90542085227a3f7b8932e4c3f7bac` (results\rebuttal_recursive_tinyimagenet_e16_ttrain32\config_frozen.json)
- `e17_t8`: `b5d610a01747e1235ac4bcaab32b42fbe589569381b37628a447c6db4d3ffa21` (results\modern_iterative_panel\protocol\attention_t8_config.json)
- `e17_t32`: `d9aa18dc6136ec595986c6327e3d5fb7194b1b48ffcd54ba4f1586041a42cf3d` (results\modern_iterative_panel\protocol\attention_t32_control_config.json)
- `followup`: `f11fda7bc718679b7b6cf3f0058c4b41cfabd9872b144155e9c61f7a666f54c5` (results\modern_iterative_panel\followup_analysis\followup_config.json)

## Deviations

- `engineering_file_copy_repair`: Python text reading normalised Windows newlines in the copied JSON, so its byte hash differed even though the parsed frozen configuration was identical. Scientific effect: none.
- `execution_channel_crash_recovery`: The interrupted checkpoints contain model weights but not optimizer, scheduler, data-generator, or RNG state, so an epoch-8 resume would not exactly reproduce the frozen trajectory. A clean deterministic restart is the scientifically conservative recovery. Scientific effect: none.
- `execution_timeout_recovery`: Later epochs slowed when the laptop GPU operated at approximately 28 W. Because optimizer, scheduler, data-generator, and RNG state were not checkpointed, a clean deterministic restart is required instead of an inexact epoch-29 resume. Scientific effect: none.
- `no_op_system_python_invocation`: A recovery invocation resolved to the system Python and stopped immediately on a missing matplotlib import. Scientific effect: none; no scientific file or metric was created.
- `posthoc_analysis_runner_no_op`: An initial one-second classwise command allowance and a background-launch attempt ended before producing a scientific artifact. Scientific effect: none; no classwise result, prediction archive, or aggregate was committed.
- `posthoc_bf16_reproduction_guard_repair`: The first classwise reproduction check classified step 8 and step 32 in separate 2-D BF16 GEMMs, yielding two to four boundary predictions different from the frozen evaluator's single full-trajectory GEMM. Scientific effect: none; the failed guard stopped before emitting predictions or classwise statistics.
