"""
blendemu - Blending simulation and emulation for weak lensing.

The package covers the full pipeline:

    Simulation + shape measurement (catalog, shape, response)
    -> Emulator training (data_utils, utils)
    -> Inference / n(z) correction (inference, nz_utils)

Module overview:
    catalog     Galaxy catalogue generation for image simulations
    shape       Shape measurement with ngmix and GalSim
    response    Response, detection, and self-response catalogue construction
    data_utils  Data loading, preprocessing, feature rescaling
    utils       KDTree neighbor finding, bright-neighbor removal, prediction helpers
    nz_utils    Low-level n(z) correction helpers
    inference   High-level BlendingPredictor for applying trained emulators
    config      YAML configuration loader
"""

from . import catalog, shape, response, data_utils, utils, nz_utils, inference
from .inference import BlendingPredictor
from .config import load_config
