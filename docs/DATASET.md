# Dataset and Provenance

## OCMR source

- Dataset: Ohio State Cardiac MRI Raw Data (OCMR).
- License: Creative Commons Attribution-NonCommercial 4.0.
- Source registry: <https://registry.opendata.aws/ocmr_data/>.

| Role | Object URL | Bytes | SHA-256 |
|---|---|---:|---|
| Development, `fs0152`, 0.55 T | `https://ocmr.s3.us-east-2.amazonaws.com/data/fs_0152_0_55T.h5` | 72,027,848 | `65ff79868b4b273aeee550c5bbf3e4ce266d12f0d9c23d435e406d5467e026b5` |
| Held-out, `fs0005`, 1.5 T | `https://ocmr.s3.us-east-2.amazonaws.com/data/fs_0005_1_5T.h5` | 220,258,096 | `1f3beee40b9186337b18f5acb6ff803c03e7b3ce4b68b3d8dc9b619cbcd320d4` |
| Held-out, `fs0016`, 3 T | `https://ocmr.s3.us-east-2.amazonaws.com/data/fs_0016_3T.h5` | 417,590,520 | `763525771bf7617a2e5cea5c3d106f503df553d2de78f20d5fd2a3f6d4404740` |

`./reproduce.sh get-data` consumes these exact URLs and refuses any file whose
size or SHA-256 differs.

The source files contain fully sampled cardiac cine k-space. The evaluation
uses real anatomy and coil measurements, selects phases 0, 6, and 13, center
crops/pads to 256x256, and compresses each acquisition to 8 virtual coils with
one frozen covariance eigenbasis per acquisition.

## Retrospective trajectory boundary

The fixed spiral trajectory is versioned to preserve the evaluated sample
locations. Its SHA-256 checksums are:

```text
fe1955cbcd0e343b7724357a833ea83b6d47e3ffbc749ac3208393a879483eba  spiral_standard_256.npy
faa3f82b6f55ea26bb562ea2ebd501047db0bb184ac55ca8db37db11070a97e  spiral_standard_256.f32xy.bin
```

The OCMR acquisition is Cartesian. Spiral, radial, and golden-angle
measurements are generated retrospectively from the common real image and coil
maps. They must not be described as prospectively acquired non-Cartesian OCMR
measurements.

## Repository policy

No OCMR HDF5, derived coil arrays, sensitivity maps, measurements, or packed
matrices are tracked. Only source code, hashes, compact metrics, and the fixed
trajectory fixture are staged. Reproduction downloads data from OCMR under
the upstream license.
