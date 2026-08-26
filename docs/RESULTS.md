# Frozen Result Record

The authoritative record is `evidence/frozen_results.json`. Run
`./reproduce.sh evidence` to recompute it from the hashed source summaries and
validate every result gate.

## Quality

- 27 of 27 reconstruction rows pass across three retrospective OCMR
  acquisitions, three trajectory families, and three cardiac phases.
- Worst native-versus-FP32 magnitude NRMSE: 0.1857%.
- Minimum native-versus-FP32 SSIM: 0.9999628.
- Eager and Graph reconstruction outputs are bitwise equal for every row.
- The selected difficult row remains finite with positive curvature through
  50 CG steps; its 50-step magnitude NRMSE is 0.0333%.

## Performance

- Native cuFINUFFT full CG: 2.514x paired geometric-mean speedup, 95% CI
  [2.484, 2.542], with 6 of 6 process-pair wins.
- Equal-quality dense control: 1.408x in eager mode and 1.414x in Graph mode,
  with 6 of 6 wins in each mode.

All performance values use complete device-resident ten-step reconstruction.
Setup, transfers, offline packing, and baseline planning are excluded. The
external baseline is quality-admitted before timing; the dense control differs
only in the interpolation implementation.

## Non-claims

The evidence does not establish numerical equivalence, clinical equivalence,
prospective-trajectory performance, portability to other shapes or GPU
generations, or one-shot workloads that cannot amortize packing.
