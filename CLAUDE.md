# CLAUDE.md

End-to-end weak-lensing blending pipeline (simulate → measure → train emulators → inference) that corrects blending bias in cosmology n(z). Companion to Zhang et al. 2025 (arXiv:2507.19130). Imported as `from blendemu import BlendingPredictor, load_config`.

## Environment
- Conda env: `conda activate sims1` (Python 3.9+). Env path `/project/ls-gruen/users/zekang.zhang/envs/sims1`.
- NOT pip-installable — no setup.py/pyproject. Run with the repo on PYTHONPATH:
  `export PYTHONPATH="$PWD:$PYTHONPATH"`
- Key deps: numpy/pandas/scipy/astropy, galsim+ngmix (shape), mpi4py (parallel sim/shape), xgboost/optuna/scikit-learn (training), cosmic_toolbox (galsbi loading).

## Tests
- `pytest tests/` from the repo root (tests do `from scripts import run_pipeline`, so root must be cwd).
- Gotcha: many files in `tests/` are diagnostic/merge utilities, NOT tests (e.g. `merge_training_catalogues.py`, `diagnose_self_response_catalogues.py`).

## Jobs (SLURM — never run nontrivial work on the login node)
- Submit DAGs via wrappers, e.g. `jobs/submit_fs2_lsst_r_split.sh` (catalog → sim → shape → response → train, chained with `sbatch --parsable --dependency=afterok`).
- Env is centralized: jobs `source jobs/env_fs2_lsst_r.sh` (activates conda, default `sims1`).
- Sim steps need env vars: `BLENDEMU_SIM_RUN=/path/to/MultiBand_ImSim/modules/Run.py`, plus `BLENDEMU_CONFIG`. Sim jobs also `module load sextractor` and set `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`.
- Slurm logs go to `/home/z/Zekang.Zhang/logs/`.

## Entry points (scripts/)
- `run_pipeline.py --config <yaml> --steps {all|1-5|self_response}` — orchestrator. Steps: 1=catalogue, 2=image sim + SExtractor detection (MPI), 3/3b=shape via ngmix (MPI), 4/4b=response/self-response catalogues.
- `train_emulator.py --mode {tune,train} --task {regression,classification,self_response}` — Optuna + XGBoost.
- `run_inference.py --config --catalogue <feather> --nz-file <fits> --output <fits>`.
- Programmatic: `BlendingPredictor.load('./models', conditions={...}).correct_nz(catalog, (dndz, z))`.

## Config & conventions
- YAML configs in `configs/`; sections: `catalog`, `simulation`, `noise`, `shape_measurement`, `catalogues`, `training`, `inference`, `cluster`. `simulation.base_config` → `base_sim_config.ini` (path relative to the YAML). Real `*.yaml` are gitignored; `*.example.yaml` are templates.
- `training.model_tag` (e.g. `lsst_r`) is appended to every model filename and to `models/emulator_metadata_<tag>.json` (holds model filenames, feature boundaries, standardization, metrics).
- XGBoost does NOT extrapolate: `training.classification_cuts[1][1]` must reach `catalog.mag_cut` (the faint limit of the inference catalogue).
- Shear convention: sky spin-2 `(q1,q2)=q*(cos2θ,sin2θ)`, tagged `shear_component_convention="sky_cos_sin"`. Negative shear labels on CLI need `--flag=value` form (e.g. `--shear=-0.02`).
- Real data lives on `/project/ls-gruen/...`, not in-repo `data/`.

## Gotchas
- `.json.lock` files next to metadata = concurrent-write locks during parallel Optuna tuning; leave them.
- Two catalogue sources: `fs2` (FS2) and `galsbi` f24 (galsbi needs `cosmic_toolbox`).
