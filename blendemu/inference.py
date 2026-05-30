"""
Inference pipeline: use trained emulators to correct the source n(z) for blending.

Given a galaxy catalogue (columns: RA, DEC, redshift, r, Re, sersic_n,
axis_ratio) and a target lensing n(z), predicts the blending-induced delta-n(z)
correction using the trained classification (detection) and regression
(shear-response) emulators.

Main entry points:
  - BlendingPredictor.load(model_dir, conditions)   # load both emulators
  - predictor.correct_nz(icat, nz)                   # apply to a catalogue
"""

import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb

from . import data_utils, nz_utils


class BlendingPredictor:
    """
    Holds the trained emulator suite for a survey and applies them to an
    input galaxy catalogue.

    Three emulators may be loaded (all optional except classification+regression):
    - bst_cla:  classification — per-galaxy detection probability
    - bst_reg:  regression — blending response (delta_et from nearby sheared neighbors)
    - bst_self: regression — self-response (galaxy's own shear response)
    """

    REQUIRED_CONDITIONS = ('pixel_size', 'zero_point', 'psf_fwhm', 'moffat_beta', 'pixel_rms')

    def __init__(self, bst_cla, bst_reg, y_mean, y_std,
                 boundaries_cla=None, boundaries_reg=None, conditions=None,
                 bst_self=None, y_mean_self=None, y_std_self=None, boundaries_self=None):
        self.bst_cla = bst_cla
        self.bst_reg = bst_reg
        self.y_mean = y_mean
        self.y_std = y_std
        self.boundaries_cla = boundaries_cla
        self.boundaries_reg = boundaries_reg

        # Self-response (optional)
        self.bst_self = bst_self
        self.y_mean_self = y_mean_self
        self.y_std_self = y_std_self
        self.boundaries_self = boundaries_self

        if conditions is None:
            raise ValueError(
                "`conditions` is required: must be a dict with keys "
                f"{self.REQUIRED_CONDITIONS}. Survey-specific — do not guess.")
        missing = set(self.REQUIRED_CONDITIONS) - set(conditions)
        if missing:
            raise ValueError(f"`conditions` is missing keys: {sorted(missing)}")
        self.conditions = dict(conditions)

    @classmethod
    def load(cls, model_dir, tag=None, conditions=None, device='cuda',
             cla_file=None, reg_file=None, self_file=None,
             standardization_file=None, standardization_self_file=None,
             boundary_cla_file=None, boundary_reg_file=None, boundary_self_file=None,
             load_self=True, metadata_file=None):
        """
        Load a trained emulator suite from disk.

        Parameters
        ----------
        model_dir : str
            Directory containing the saved model files.
        tag : str, optional
            Model tag (e.g. 'lsst_r'). If provided, appends '_{tag}' to each
            default filename (e.g. 'regression_model_lsst_r.json').
        conditions : dict, optional
            Observing conditions for feature rescaling.
        device : str
            XGBoost device ('cuda' or 'cpu').
        load_self : bool
            If True (default) and the self-response model file exists, also
            load the self-response emulator. Set False to skip.
        metadata_file : str, optional
            JSON sidecar containing task model filenames, training boundaries,
            y-standardization, and training parameters. Defaults to
            'emulator_metadata_{tag}.json' when a tag is provided. If absent,
            legacy .npy metadata files are used.

        Returns
        -------
        BlendingPredictor
        """
        suffix = f'_{tag}' if tag else ''

        explicit_standardization_file = standardization_file is not None
        explicit_standardization_self_file = standardization_self_file is not None
        explicit_boundary_cla_file = boundary_cla_file is not None
        explicit_boundary_reg_file = boundary_reg_file is not None
        explicit_boundary_self_file = boundary_self_file is not None

        metadata_file = metadata_file or f'emulator_metadata{suffix}.json'
        metadata = _load_metadata(os.path.join(model_dir, metadata_file))
        meta_cla = _metadata_task(metadata, 'classification')
        meta_reg = _metadata_task(metadata, 'regression')
        meta_self = _metadata_task(metadata, 'self_response')

        cla_file = cla_file or meta_cla.get('model_file') or f'classification_model{suffix}.json'
        reg_file = reg_file or meta_reg.get('model_file') or f'regression_model{suffix}.json'
        self_file = self_file or meta_self.get('model_file') or f'self_response_model{suffix}.json'
        standardization_file = standardization_file or f'train_standardization{suffix}.npy'
        standardization_self_file = standardization_self_file or f'train_standardization_self{suffix}.npy'
        boundary_cla_file = boundary_cla_file or f'train_boundary_cla{suffix}.npy'
        boundary_reg_file = boundary_reg_file or f'train_boundary_reg{suffix}.npy'
        boundary_self_file = boundary_self_file or f'train_boundary_self{suffix}.npy'

        # Classification + blending regression (required)
        bst_cla = xgb.Booster({'device': device, 'n_jobs': -1})
        bst_cla.load_model(os.path.join(model_dir, cla_file))

        bst_reg = xgb.Booster({'device': device, 'n_jobs': -1})
        bst_reg.load_model(os.path.join(model_dir, reg_file))

        if not explicit_standardization_file:
            standardization = _metadata_standardization(meta_reg)
        else:
            standardization = None
        if standardization is None:
            standardization = np.load(os.path.join(model_dir, standardization_file))
        y_mean, y_std = standardization

        bcla = None if explicit_boundary_cla_file else _metadata_boundary(meta_cla)
        if bcla is None:
            bcla = _load_optional(os.path.join(model_dir, boundary_cla_file))
        breg = None if explicit_boundary_reg_file else _metadata_boundary(meta_reg)
        if breg is None:
            breg = _load_optional(os.path.join(model_dir, boundary_reg_file))

        # Self-response (optional)
        bst_self, y_mean_self, y_std_self, bself = None, None, None, None
        self_path = os.path.join(model_dir, self_file)
        if load_self and os.path.exists(self_path):
            bst_self = xgb.Booster({'device': device, 'n_jobs': -1})
            bst_self.load_model(self_path)
            std_self_path = os.path.join(model_dir, standardization_self_file)
            if not explicit_standardization_self_file:
                standardization_self = _metadata_standardization(meta_self)
            else:
                standardization_self = None
            if standardization_self is None and os.path.exists(std_self_path):
                standardization_self = np.load(std_self_path)
            if standardization_self is not None:
                y_mean_self, y_std_self = standardization_self

            bself = None if explicit_boundary_self_file else _metadata_boundary(meta_self)
            if bself is None:
                bself = _load_optional(os.path.join(model_dir, boundary_self_file))

        return cls(bst_cla, bst_reg, y_mean, y_std,
                   boundaries_cla=bcla, boundaries_reg=breg, conditions=conditions,
                   bst_self=bst_self, y_mean_self=y_mean_self, y_std_self=y_std_self,
                   boundaries_self=bself)

    def predict_detection(self, icat, warn_extrapolation=True):
        """
        Predict per-galaxy detection probability.

        Returns
        -------
        pd.DataFrame
            Feature DataFrame with an added 'detection_prob' column.
        """
        cla_fea = nz_utils.icat2cla(
            icat, icat, self.bst_cla, self.conditions, predict=False,
        )
        if warn_extrapolation and self.boundaries_cla is not None:
            _warn_if_out_of_bounds(
                cla_fea, self.bst_cla.feature_names, self.boundaries_cla, 'detection',
            )
        cla_fea['detection_prob'] = data_utils.xgb_pred(self.bst_cla, cla_fea)
        return cla_fea

    def predict_response(self, icat_pri, icat_sec):
        """
        Predict per-pair **blending response** (delta_et_primary / gamma due to
        sheared neighbor), de-standardized to physical units.
        """
        reg_fea = nz_utils.icat2reg(icat_pri, icat_sec, self.bst_reg, self.conditions)
        response = data_utils.xgb_pred(self.bst_reg, reg_fea)
        reg_fea['response'] = data_utils.reverse_standardize(response, self.y_mean, self.y_std)
        return reg_fea

    def predict_self_response(self, icat_pri, icat_sec):
        """
        Predict per-galaxy **self-response** (delta_et_galaxy / gamma in response
        to the galaxy's own shear), de-standardized to physical units.

        For self-response, `icat_pri` are the target galaxies and `icat_sec`
        provides their nearest neighbors (typically the same catalogue).
        """
        if self.bst_self is None:
            raise RuntimeError(
                "No self-response emulator loaded. Check that "
                "self_response_model_{tag}.json exists, or pass load_self=True to BlendingPredictor.load().")
        reg_fea = nz_utils.icat2reg(icat_pri, icat_sec, self.bst_self, self.conditions)
        response = data_utils.xgb_pred(self.bst_self, reg_fea)
        reg_fea['self_response'] = data_utils.reverse_standardize(
            response, self.y_mean_self, self.y_std_self,
        )
        return reg_fea

    # ─────────────────────────────────────────────────────────────
    # Pair-level API: for users who have a custom paired catalogue
    # already (e.g. from their own neighbor finder) and want to skip
    # the catalogue-level KDTree step.
    # ─────────────────────────────────────────────────────────────

    def predict_on_pairs(self, pair_df, task='response', rescaled=False, warn_extrapolation=True):
        """
        Run an emulator on a pre-paired DataFrame (primary + neighbor features).

        Use this when you already have a paired catalogue with columns like
        ``Re_input_p``, ``r_input_p``, ``sersic_n_input_p``, ``Re_input_s``,
        ``r_input_s``, ``sersic_n_input_s``, ``distance`` — for example the
        output of a custom neighbor finder, or a single-pair 1-row DataFrame.

        Parameters
        ----------
        pair_df : pd.DataFrame
            Pair-feature catalogue. Must contain the features listed above in
            raw units (arcsec for sizes/distance, magnitudes).
            If ``rescaled=True``, must instead contain ``*_scaled`` columns.
        task : {'detection', 'response', 'self_response'}
            Which emulator to apply.
        rescaled : bool
            If True, ``pair_df`` is assumed to already contain the rescaled
            features (e.g. ``Re_input_p_scaled``). Skips the rescaling step.
        warn_extrapolation : bool
            If True (default) and training boundaries are loaded for this
            task, print a warning when any scaled feature falls outside the
            training range (XGBoost predictions there are extrapolations).

        Returns
        -------
        pd.DataFrame
            Copy of ``pair_df`` with the prediction added as:
              - 'detection_prob'   (task='detection')
              - 'response'         (task='response')
              - 'self_response'    (task='self_response')
        """
        out = pair_df.copy()

        if not rescaled:
            out = data_utils.rescale(
                out,
                pixel_rms=self.conditions['pixel_rms'],
                pixel_size=self.conditions['pixel_size'],
                zero_mag=self.conditions['zero_point'],
                psf_fwhm=self.conditions['psf_fwhm'],
                moffat_beta=self.conditions['moffat_beta'],
            )

        if task == 'detection':
            model = self.bst_cla
            boundaries = self.boundaries_cla
        elif task == 'response':
            model = self.bst_reg
            boundaries = self.boundaries_reg
        elif task == 'self_response':
            if self.bst_self is None:
                raise RuntimeError("No self-response emulator loaded.")
            model = self.bst_self
            boundaries = self.boundaries_self
        else:
            raise ValueError(f"Unknown task: {task!r}. Use 'detection', 'response', or 'self_response'.")

        if warn_extrapolation and boundaries is not None:
            _warn_if_out_of_bounds(out, model.feature_names, boundaries, task)

        pred = data_utils.xgb_pred(model, out)
        if task == 'detection':
            out['detection_prob'] = pred
        elif task == 'response':
            out['response'] = data_utils.reverse_standardize(pred, self.y_mean, self.y_std)
        else:  # self_response
            out['self_response'] = data_utils.reverse_standardize(
                pred, self.y_mean_self, self.y_std_self,
            )
        return out

    def predict_one_pair(self, primary, neighbor=None, task='response'):
        """
        Convenience wrapper: predict for a single galaxy (or galaxy pair).

        Parameters
        ----------
        primary : dict
            Target galaxy properties. Required keys: Re (arcsec), r (mag),
            sersic_n.
        neighbor : dict, optional
            Nearest-neighbor properties. Same keys as ``primary``, plus
            ``distance`` (arcsec).
            - For ``task='detection'``: pass ``None`` for an isolated galaxy;
              secondary features are set to NaN to match the training
              "no neighbor" convention.
            - For ``task='response'`` or ``'self_response'``: ``neighbor`` is
              required (these tasks predict the effect of a neighbor on
              shear measurement; "no neighbor" has no defined semantics here).
        task : {'detection', 'response', 'self_response'}

        Returns
        -------
        float
            The single predicted value.

        Raises
        ------
        ValueError
            If ``neighbor=None`` is passed for regression tasks.

        Examples
        --------
        >>> predictor.predict_one_pair(
        ...     primary={'Re': 0.4, 'r': 23.5, 'sersic_n': 1.2},
        ...     neighbor={'Re': 0.3, 'r': 24.5, 'sersic_n': 2.0, 'distance': 2.0},
        ...     task='response')
        """
        if neighbor is None and task in ('response', 'self_response'):
            raise ValueError(
                f"task={task!r} requires a neighbor. 'no neighbor' has no defined "
                f"semantics for shear-response regression — the emulator only makes "
                f"sense when a nearby galaxy is present. Pass neighbor={{'Re': ..., "
                f"'r': ..., 'sersic_n': ..., 'distance': ...}}, or switch to task='detection'.")

        row = {
            'Re_input_p': primary['Re'],
            'r_input_p': primary['r'],
            'sersic_n_input_p': primary['sersic_n'],
        }
        if neighbor is None:
            # Isolated-galaxy detection: use NaN to match training convention
            # (icat2cla sets _s features to NaN when no neighbor within R_MAX_CLA).
            row['Re_input_s'] = np.nan
            row['r_input_s'] = np.nan
            row['sersic_n_input_s'] = np.nan
            row['distance'] = np.nan
        else:
            row['Re_input_s'] = neighbor['Re']
            row['r_input_s'] = neighbor['r']
            row['sersic_n_input_s'] = neighbor['sersic_n']
            if 'distance' not in neighbor:
                raise ValueError("neighbor dict must include 'distance' (arcsec)")
            row['distance'] = neighbor['distance']

        df = pd.DataFrame([row])
        out = self.predict_on_pairs(df, task=task)
        col = {'detection': 'detection_prob',
               'response': 'response',
               'self_response': 'self_response'}[task]
        return float(out[col].iloc[0])

    def correct_nz(self, icat, nz, resample_frac=0.25, seed=42):
        """
        Compute the blending-induced correction delta_n(z).

        Parameters
        ----------
        icat : pd.DataFrame
            Source galaxy catalogue with columns RA, DEC, redshift, r, Re,
            sersic_n, axis_ratio.
        nz : tuple (dndz, z_grid)
            Target lensing n(z): array of density values and corresponding redshifts.
        resample_frac : float
            Fraction of the input catalogue to resample (for target n(z)).
        seed : int

        Returns
        -------
        reg_fea : pd.DataFrame
            Pair-feature catalogue with predicted responses.
        z : np.ndarray
            Redshift grid (same as nz[1]).
        delta_n : np.ndarray
            Correction to n(z) at each z point.
        """
        # Step 1: detection probability for each input galaxy
        detect_pred = self.predict_detection(icat)['detection_prob'].values

        # Step 2: resample to match the target lensing n(z) distribution, weighted
        # by (target n(z) / empirical pre-n(z) / detection probability)
        resampled_idx = _nz_resample(icat['redshift'].values, nz,
                                     detec_prob=detect_pred, frac=resample_frac, seed=seed)

        # Step 3: build pair catalogue and predict responses
        reg_fea = self.predict_response(
            icat.iloc[resampled_idx].reset_index(drop=True), icat,
        )
        reg_fea['detection'] = np.ones(reg_fea.shape[0])

        # Step 4: compute delta-n(z) correction
        z = nz[1]
        step_w = np.mean(z[1:] - z[:-1])
        z_bins = np.array([[zi - step_w, zi + step_w] for zi in z])
        tot = resampled_idx.shape[0]

        additional_names = [
            'RA_input_p', 'r_input_p', 'r_input_s',
            'Re_input_p', 'Re_input_s', 'distance',
            'sersic_n_input_p', 'sersic_n_input_s',
            'redshift_input_s', 'redshift_input_p', 'detection',
        ]
        feature_cols = self.bst_reg.feature_names + additional_names
        delta_n = nz_utils.n_correction(
            reg_fea[feature_cols], reg_fea['response'], z_bins, tot,
        )

        return reg_fea, z, delta_n


def _load_optional(path):
    return np.load(path) if os.path.exists(path) else None


def _load_metadata(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _metadata_task(metadata, task):
    return (metadata or {}).get('tasks', {}).get(task, {})


def _metadata_boundary(task_metadata):
    boundary = task_metadata.get('boundary')
    if boundary is None:
        return None
    return np.asarray(boundary, dtype=float)


def _metadata_standardization(task_metadata):
    standardization = task_metadata.get('standardization')
    if standardization is None:
        return None
    if isinstance(standardization, dict):
        return (
            float(standardization['mean']),
            float(standardization['std']),
        )
    return tuple(float(value) for value in standardization)


def _warn_if_out_of_bounds(df, feature_names, boundaries, task):
    """Print a warning if any row has features outside the training boundaries.

    `boundaries` is shape (n_features, 2): [[min, max], ...] in the same
    order as `feature_names`. NaN values are ignored (those are the
    "no-neighbor" convention for classification).
    """
    import warnings
    n_rows = len(df)
    per_feature_oob = {}
    for name, (lo, hi) in zip(feature_names, boundaries):
        if name not in df.columns:
            continue
        vals = df[name].values
        finite = ~np.isnan(vals)
        oob = finite & ((vals < lo) | (vals > hi))
        n_oob = int(oob.sum())
        if n_oob > 0:
            per_feature_oob[name] = (n_oob, float(lo), float(hi),
                                     float(np.nanmin(vals[oob])),
                                     float(np.nanmax(vals[oob])))
    if per_feature_oob:
        msg = [f"[blendemu:{task}] {n_rows} input rows, extrapolation outside training boundaries:"]
        for name, (n_oob, lo, hi, v_lo, v_hi) in per_feature_oob.items():
            msg.append(f"  {name}: {n_oob}/{n_rows} outside [{lo:.3g}, {hi:.3g}] "
                       f"(input range [{v_lo:.3g}, {v_hi:.3g}])")
        msg.append("  Predictions for these rows are extrapolations and may be unreliable.")
        warnings.warn('\n'.join(msg), stacklevel=3)


def _nz_resample(redshifts, target_nz, detec_prob, frac=0.25, n_bins=50, seed=42):
    """Resample an input catalogue to match a target n(z), reweighting by detection probability."""
    rng = np.random.default_rng(seed)

    pre_counts, pre_z = np.histogram(redshifts, bins=n_bins, density=True)
    pre_centers = 0.5 * (pre_z[1:] + pre_z[:-1])

    n_pre = np.interp(redshifts, pre_centers, pre_counts)
    n_tgt = np.interp(redshifts, target_nz[1], target_nz[0])

    eps = 1e-12
    n_pre = np.clip(n_pre, eps, None)
    p_det = np.clip(detec_prob, eps, None)

    weights = np.clip(n_tgt / (n_pre * p_det), 0, None)
    probs = weights / weights.sum()

    n_samples = int(len(redshifts) * frac)
    return rng.choice(len(redshifts), size=n_samples, replace=True, p=probs)


def force_norm(dndz, dz):
    """Normalize n(z) using trapezoidal rule."""
    norm = ((dndz[:-1] + dndz[1:]) * 0.5 * dz).sum()
    return dndz / norm
