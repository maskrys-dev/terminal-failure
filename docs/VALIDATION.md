# Release validation

Packaging checks completed on 26 September 2026:

- Parsed all 51 Python source files and all included JSON files.
- Verified the assembled repository against its SHA-256 manifest.
- Ran `scripts/reproduce_figures.py` in a separate copy, without training,
  dataset downloads, or access to the original workspace's result paths.
- Confirmed all six regenerated modern LaTeX tables match the supplied
  preprint tables byte-for-byte.
- Created clean SwiGLU and attention run directories with `prepare_run.py`.
- Confirmed E14, E15, and E16 command-line entry points load in the clean
  SwiGLU directory.
- Passed E17 frozen-protocol and locked-E16 verification for both training
  horizons in the clean attention directory.
- Verified all 67 checkpoint files inside the companion ZIP against their
  individual SHA-256 hashes.
- Scanned included text for common private-key and access-token patterns;
  no matches were found.

Full training and GPU re-evaluation were not repeated for packaging. Recorded
scientific results, configurations, protocols, and training code are preserved.
The figure wrappers and clean-run helpers are packaging additions. The original
dependency list gains an explicit Pillow dependency, and the dataset notes
now cover Tiny ImageNet. The provided preprint PDF is unchanged by packaging.
