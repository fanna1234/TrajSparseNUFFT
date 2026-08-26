# Evidence Layout

`source/` contains the complete sanitized JSON summaries from the accepted
runs. Absolute machine paths and GPU UUIDs are removed, but numerical fields,
binary hashes, run configuration, per-row records, process-pair records, and
negative diagnostics are preserved.

`frozen_results.json` is derived only from these tracked summaries.
`tools/build_evidence.py` reconstructs it, and `tools/verify_evidence.py`
checks every source hash before validating the quality, performance,
long-iteration, and hardening gates. A `repo://` string records an original
repository-relative run location; it is provenance metadata, not an input
required by the verifier.

Large MRI arrays, compiled binaries, profiler databases, and raw sanitizer
logs are intentionally excluded. Fresh runs regenerate those artifacts under
`reproduced-data/` and `reproduced-results/`.
