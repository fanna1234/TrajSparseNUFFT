# Validation status

The 2026-09-14 repository check exercised the delivered code independently
of the development workspace. The CUDA implementation and frozen result
record remain unchanged; this update repairs the reproduction layer and
improves its documentation.

## Executed checks

| Check | Outcome |
|---|---|
| CPU structural and reproduction tests | 21 passed from a clean source archive, including missing/duplicate results and invalid metrics |
| Frozen evidence | Source hashes and derived result record verified |
| Production and dense-control build | Built from the standalone source on SM120a / CUDA 13.2 |
| Native cuFINUFFT build | Built from the locked upstream revision |
| Raw data | All three OCMR object hashes verified; cached downloads reused |
| Data preparation | Three acquisitions, all runtime inputs, sparse packs, and dense packs regenerated |
| Reconstruction quality | 27/27 regenerated cases passed; eager/Graph outputs were bitwise equal |
| External baseline admission | 18/18 fresh outputs passed; the separate repeat diagnostic passed 8/9 pairs |
| External complete-CG comparison | 2.5443× over native cuFINUFFT, six process pairs; reproduction gate passed |
| Matched dense-control comparison | 1.4028× eager / 1.4047× Graph; both within the existing 3% tolerance |
| README | Rendered at repository reading width; diagram and relative links checked |

The GPU checks used an RTX PRO 6000 Blackwell, driver 610.43.02, toolkit
13.2.51, Python 3.13.9, and the direct dependencies in `requirements-gpu.txt`.
The baseline source was fetched at its pinned revision and copied to an
isolated build tree because the GPU host's Git proxy was unavailable. Its
dependencies were fetched with a process-local network override; no shared
proxy configuration was changed.

## Reproduction fixes

The previous preparation wrapper passed the operator normalization scale
to measurement export, multiplying observations by 512. This was a packaging
error: the evaluated runtime inputs were not scaled that way. Recreating the
case from raw data exposed FP16 overflow and nonfinite CG output.

Measurements now remain unscaled, while raw-reference operator scale 512
and native scale 1 are preserved. Runtime manifests reject the old scaling
mistake and verify every payload's size and hash. Full regeneration then
passed the original 27-row quality criterion; no threshold was relaxed.

The chosen GPU Python interpreter now carries through all quality and
measurement groups. New `TRAJTC_` configuration names accept the existing
`TRAJSPARSE_` aliases but reject conflicting values. Result checks examine
individual cases and process pairs rather than trusting summary flags alone.

## Interpreting this record

The [fresh validation record](../evidence/reproduction_20260914.json) contains
the aggregate metrics, binary identities, and source-summary hashes. The
frozen benchmark record remains the reference for the README's reported
results; it was not replaced with the new measurements. Raw data, generated
arrays, machine-specific logs, and compiled binaries remain local outputs
rather than tracked source files.

Sanitizer stress and the longer-CG sweep retain their existing evidence and
reproduction commands; they were not rerun as part of the documentation and
data-preparation repair.
