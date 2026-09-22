# Source import scope — 2026-09-22
The IAC-B Python package, the case-balanced trainer, and the GeoFlow package/configs/scripts were copied from Drive. Model mathematics were not changed. Existing flow/ and iacflow/ stay separate.

Before the workspace connection was lost, the source preparation passed:
- Python syntax parsing.
- IAC-B selftest --skip_e2e: SDF round-trip, zero-init identity, correlated noise, candidate decoding, gap classification, oracle HD95.
- GeoFlow test_geoflow_contracts.py: moving Newton evidence, zero-init, gauges, area weights, trust region, terminal bridge contracts.
- Case-balanced trainer --help.

CPU environment: Python 3.12, PyTorch 2.14.0+cpu, NumPy 2.5.3, SciPy 1.18.1, scikit-image 0.26.0, SimpleITK 2.5.6. No real-data training, full-volume evaluation or GPU run was performed. This is an archival code import, not a new benchmark.

After reconnection failed, these source files were fetched again via Drive for publication; their modification timestamps precede the completed source checks. Source notebooks, weights, volume arrays and all old code variants are not part of this smaller recovery import. The local-only initial commit 3a1ea33 was not pushed.

Install an appropriate PyTorch build plus numpy, scipy, pandas, scikit-image and SimpleITK for IAC-B. GeoFlow dependencies are described in its pyproject.toml/requirements-colab.txt. Run each package from its directory. IAC-B has no SDF autoencoder in its default path; ae_gate.py is a separate reconstruction check.
