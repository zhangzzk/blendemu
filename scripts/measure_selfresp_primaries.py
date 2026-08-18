#!/usr/bin/env python
"""Measure PRIMARY shapes for the self_response set (g0.0, g0.05).

The stock pipeline only measures primaries for the ``response`` set (step 3 ->
shears {0.0, 0.2}) and secondaries for ``self_response`` (step 3b -> {0.0, 0.05}).
It has no step that measures *primaries* of the self_response set, so the g0.05
primary shape catalogues are not a regenerable product. They are needed to rebuild
the response-penalty target (RESP) on the c2-centroid-FIXED shapes; without a fresh
g0.05 primary re-measure the RESP stays stale and pins R_flow high (m ~ -2.6% on
the fixed constant cats). This driver fills exactly that gap: it reuses the existing
rendered images + detections and runs run_shape.py --targets primaries over the
self_response shears. g0.0 primaries already exist (re-measured) and are skipped
per-case by run_shape; only the missing g0.05 primaries get built.

Usage:
  python measure_selfresp_primaries.py --config ../configs/fs2_lsst_r.yaml \
      --n-mpi 100 [--case-offset 100] [--n-cases 100]
"""
import argparse

import run_pipeline as rp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--n-mpi', type=int, default=None)
    ap.add_argument('--case-offset', type=int, default=None,
                    help='Optional first self_response case to measure.')
    ap.add_argument('--n-cases', type=int, default=None,
                    help='Optional number of self_response cases to measure.')
    args = ap.parse_args()

    cfg = rp.load_config(args.config)
    if args.n_mpi is not None:
        cfg['cluster']['n_mpi'] = args.n_mpi
    if args.case_offset is not None:
        rp._set_simulation_override(cfg['simulation'], 'case_offset', args.case_offset)
    if args.n_cases is not None:
        rp._set_simulation_override(cfg['simulation'], 'n_cases', args.n_cases)

    print(rp.config_summary(cfg))
    rp._run_shape_measurement(cfg, targets='primaries',
                              step_label='Step 3-SR (self_response primaries)',
                              sim_set_name='self_response')


if __name__ == '__main__':
    main()
