# Experiment and artifact index

| Experiment | Source | Saved evidence | Role |
| --- | --- | --- | --- |
| Original submission | See `ORIGINAL_SUPPLEMENT.md` | Original directories under `results/` | Reservoir discovery, trained LeakyReLU, CIFAR-10, LSTM, noise and operator diagnostics |
| E14 | `e14_recursive_tinyimagenet_rebuttal.py` | `results/rebuttal_recursive_tinyimagenet/` | Eight-step SwiGLU seed-0 pilot, probes, readouts and runtime/memory benchmark |
| E14 horizon audit | `e14_horizon_extension_postprocess.py` | Same directory, `horizon_extension_*` | Saved-trajectory analysis; no training |
| E15 | `e15_recursive_tinyimagenet_aligned_t32.py` | `results/rebuttal_recursive_tinyimagenet_aligned_t32/` | Aligned 32-step seed-0 pilot, using the E14 driver |
| E16 | `e16_paired_inference_horizon.py` | `results/rebuttal_recursive_tinyimagenet_e16_ttrain8/`, `..._ttrain32/`, and `..._paired_horizon/` | Six paired SwiGLU runs |
| E17/E17C | `e17_recursive_attention_tinyimagenet.py` | `results/modern_iterative_panel/attention_t8/` and `attention_t32_control/` | Six paired shared-attention runs |
| Follow-up analyses | `e17_causal_panel_audit.py` | `results/modern_iterative_panel/followup_analysis/` | Fixed-alpha probes, classwise and checkpoint sensitivity, model scale, provenance and final panel |

All source filenames above are under `transient_geometry/experiments/`.

The definitive four-condition summary is
`results/modern_iterative_panel/final_architecture_panel.json`. Figures 1 and 3
from that panel are combined in the preprint's main modern figure; panel 2 is
in the appendix. `paper/generate_arxiv_assets.py` regenerates those figures and
six supporting tables from the saved evidence.

E14 and E15 did not continue past seed 0. E17C's `failed_attempts/` directory
contains interrupted attempts, followed by clean restarts with the same frozen
seed/configuration. None of these files changes the final count of 12 completed
scientific runs. Older canonical-DEQ proposals were not executed as new
rebuttal experiments and are not advertised as completed work here.

`docs/import_origins.json` maps imported release paths to workspace sources.
`docs/release_manifest.json` verifies the assembled repository;
`docs/checkpoint_manifest.json` verifies its companion archive. Frozen
experimental manifests remain unchanged, including historical references to
source revisions and checkpoint files. `.gitattributes` disables line-ending
conversion so Git preserves these byte hashes across platforms.
