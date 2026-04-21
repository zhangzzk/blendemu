# blendemu

End-to-end weak lensing blending pipeline: **simulate → measure → train emulators → apply emulators to correct n(z)**.

Paper: [Zhang et al. 2025, arXiv:2507.19130](https://arxiv.org/pdf/2507.19130)

## Pipeline overview

| Step | Script / notebook | Description |
|------|------------------|-------------|
| 1 | `run_pipeline.py --steps 1` | Generate galaxy catalogues + sim configs from a base catalogue (FS2 / galsbi) |
| 2 | `run_pipeline.py --steps 2` | Run MultiBand_ImSim image simulation + SExtractor detection (MPI) |
| 3 | `run_pipeline.py --steps 3` | Shape measurement with ngmix at detected primary positions (MPI) |
| 3b | `run_pipeline.py --steps 3b` | Shape measurement for **secondaries** (self-response targets) |
| 4 | `run_pipeline.py --steps 4` | Build blending-response + detection catalogues |
| 4b | `run_pipeline.py --steps 4b` | Build **self-response** catalogue |
| 5 | `train_emulator.py --mode {tune,train}` | Optuna hyperparameter tuning + XGBoost training |
| **6** | **`run_inference.py`** | **Apply trained emulators to an input catalogue → corrected n(z)** |

## Directory layout

```
blendemu/
├── blendemu/                       # Python package
│   ├── catalog.py                  # galaxy catalogue generation (FS2, galsbi)
│   ├── shape.py                    # ngmix + GalSim HSM shape measurement
│   ├── response.py                 # response, detection, self-response catalogues
│   ├── data_utils.py               # loading, rescaling, feature engineering
│   ├── utils.py                    # KDTree helpers, bright-neighbor removal
│   ├── nz_utils.py                 # low-level n(z) correction math
│   ├── inference.py                # BlendingPredictor (high-level inference)
│   └── config.py                   # YAML config loader
├── scripts/                        # CLI entry points
│   ├── run_pipeline.py             # simulation + catalogue assembly stages
│   ├── run_sim.py                  # MPI worker for image sim
│   ├── run_shape.py                # MPI worker for shape measurement
│   ├── train_emulator.py           # Optuna tune / final train
│   └── run_inference.py            # apply emulators, produce corrected n(z)
├── configs/                        # YAML configs (fs2_lsst_r, galsbi_f24, ...)
├── notebooks/                      # analysis + inspection
│   ├── test_inspection.ipynb            # simulation run diagnostics
│   ├── inspect_emulator.ipynb           # blending-response emulator results
│   ├── inspect_self_response.ipynb      # self-response emulator results
│   └── inference_nz_correction.ipynb    # apply emulators to correct n(z)
├── models/                         # trained XGBoost models + standardization
├── data/                           # example galaxy catalogues
├── jobs/                           # SLURM submission scripts
└── README.md
```

## Setup

Clone the repo and put it on your `PYTHONPATH`:

```bash
git clone https://github.com/<your-org>/blendemu.git
cd blendemu
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Copy the example config and fill in paths for your environment:

```bash
cp configs/fs2_lsst_r.example.yaml configs/fs2_lsst_r.yaml
# edit configs/fs2_lsst_r.yaml to point at your data / output directories
```

If you intend to run the image-simulation steps, also point at your local
clone of MultiBand_ImSim:

```bash
export BLENDEMU_SIM_RUN=/path/to/MultiBand_ImSim/modules/Run.py
```

## Quick inference example

```python
from blendemu import BlendingPredictor
import pandas as pd
import numpy as np

predictor = BlendingPredictor.load(
    './models',
    conditions={'pixel_size': 0.2, 'zero_point': 30,
                'psf_fwhm': 0.73, 'moffat_beta': 2.224, 'pixel_rms': 0.312},
)

icat = pd.read_feather('data/example_catalog.feather')

z = np.linspace(0, 2, 201)
dndz = np.exp(-(z - 0.7)**2 / (2 * 0.1**2))
dndz /= dndz.sum() * (z[1] - z[0])

_, z, delta_n = predictor.correct_nz(icat, (dndz, z))
print(f'<z> shift: {np.trapz(delta_n * z, z):+.4f}')
```

## Full pipeline via CLI

```bash
# End-to-end for 200 cases
python scripts/run_pipeline.py --config configs/fs2_lsst_r.yaml --steps 1-5

# Self-response branch only
python scripts/run_pipeline.py --config configs/fs2_lsst_r.yaml --steps 3b,4b
python scripts/train_emulator.py --config configs/fs2_lsst_r.yaml --mode tune --task self_response

# Apply emulators to correct an n(z)
python scripts/run_inference.py \
    --config configs/fs2_lsst_r.yaml \
    --catalogue data/example_catalog.feather \
    --nz-file hsc_nz.fits \
    --output corrected_nz.fits
```

## Dependencies

- Python 3.9+, numpy, pandas, scipy, astropy
- galsim, ngmix (shape measurement)
- mpi4py (parallel simulation + shape measurement)
- xgboost, optuna, scikit-learn (emulator training)
- joblib, tqdm (parallel catalogue assembly)
- cosmic_toolbox (for galsbi catalogue loading)
