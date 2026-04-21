"""
Shape measurement utilities using ngmix and GalSim.

Provides PSF-corrected shape measurement via ngmix (Gaussian model fitting)
and GalSim HSM (KSB shear estimation), plus image cutout and shear utilities.
"""

import numpy as np
import galsim
import ngmix


# --- ngmix shape measurement ---

def ngmix_psf_correct(obs):
    """
    Measure PSF-corrected galaxy shape using ngmix.

    Uses adaptive moments for a T guess, then fits a Gaussian model
    with ngmix (Fitter + EM PSF fitter) via the Bootstrapper.

    Parameters
    ----------
    obs : ngmix.Observation
        Observation containing the galaxy image, weight map, PSF, and jacobian.

    Returns
    -------
    dict
        Fit result dict with 'g' (shear estimates [g1, g2]) and other fields.
    """
    rng = np.random.RandomState()

    Image = galsim.Image(obs.image, scale=obs.jacobian.scale)

    try:
        hsm_shape = galsim.hsm.FindAdaptiveMom(Image, strict=False)
    except Exception:
        hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
        hsm_shape = galsim.hsm.FindAdaptiveMom(
            Image, guess_sig=10.0, hsmparams=hsmparams, strict=False
        )

    error_msg = hsm_shape.error_message
    if error_msg != '':
        T_guess = 1.0
    else:
        T_guess = 2 * (hsm_shape.moments_sigma * obs.jacobian.scale) ** 2

    prior = _get_prior(rng, scale=obs.jacobian.scale)
    fitter = ngmix.fitting.Fitter(model='gauss', prior=prior)
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(
        rng=rng, T=T_guess, prior=prior,
    )

    psf_fitter = ngmix.em.EMFitter()
    psf_guesser = ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=1)

    psf_runner = ngmix.runners.PSFRunner(
        fitter=psf_fitter, guesser=psf_guesser, ntry=2,
    )
    runner = ngmix.runners.Runner(
        fitter=fitter, guesser=guesser, ntry=2,
    )

    boot = ngmix.bootstrap.Bootstrapper(
        runner=runner, psf_runner=psf_runner,
    )

    res = boot.go(obs)
    res['error_msg'] = error_msg
    return res


def _get_prior(rng, scale):
    """Build ngmix joint prior for Gaussian model fitting."""
    g_prior = ngmix.priors.GPriorBA(0.5, rng=rng)

    cen_prior = ngmix.priors.CenPrior(0.0, 0.0, scale, scale, rng=rng)

    T_prior = ngmix.priors.FlatPrior(-10.0, 1.e6, rng=rng)
    F_prior = ngmix.priors.FlatPrior(-1.e4, 1.e9, rng=rng)

    prior = ngmix.joint_prior.PriorSimpleSep(
        cen_prior, g_prior, T_prior, F_prior,
    )
    return prior


def make_obs(im, psf_im, scale):
    """
    Create an ngmix Observation from galaxy and PSF images.

    Noise is estimated from image edges using MAD.

    Parameters
    ----------
    im : np.ndarray
        Galaxy postage stamp.
    psf_im : np.ndarray
        PSF image.
    scale : float
        Pixel scale in arcsec/pixel.

    Returns
    -------
    ngmix.Observation
    """
    cen = (np.array(im.shape) - 1.0) / 2.0
    psf_cen = (np.array(psf_im.shape) - 1.0) / 2.0

    jacobian = ngmix.DiagonalJacobian(row=cen[0], col=cen[1], scale=scale)
    psf_jacobian = ngmix.DiagonalJacobian(row=psf_cen[0], col=psf_cen[1], scale=scale)

    gal_noise = compute_sky(im)
    psf_noise = compute_sky(psf_im)
    if gal_noise == 0.:
        gal_noise = 1.e-6
    if psf_noise == 0.:
        psf_noise = 1.e-6

    wt = im * 0 + 1.0 / gal_noise**2
    psf_wt = psf_im * 0 + 1.0 / psf_noise**2

    psf_obs = ngmix.Observation(psf_im, weight=psf_wt, jacobian=psf_jacobian)
    obs = ngmix.Observation(im, weight=wt, jacobian=jacobian, psf=psf_obs)
    return obs


def compute_sky(noisy_im):
    """Estimate sky noise sigma using MAD of the image border pixels."""
    nx, ny = noisy_im.shape
    noise = np.concatenate((
        noisy_im[0, 0:ny],
        noisy_im[1:nx, -1],
        noisy_im[-1, 0:-1],
        noisy_im[1:-1, 0],
    ))
    return 1.4826 * np.median(np.abs(noise - np.median(noise)))


# --- GalSim HSM shape measurement ---

def galsim_EstimateShear(im, psf_im, method='KSB'):
    """
    Measure PSF-corrected shear using GalSim HSM EstimateShear.

    Parameters
    ----------
    im : np.ndarray
        Galaxy postage stamp.
    psf_im : np.ndarray
        PSF image.
    method : str
        Shear estimation method ('KSB', 'REGAUSS', etc.).

    Returns
    -------
    np.ndarray
        [corrected_g1, corrected_g2]
    """
    Image = galsim.Image(im, scale=0.074)
    psf_Image = galsim.Image(psf_im, scale=0.074)

    hsm_shape = galsim.hsm.FindAdaptiveMom(Image, strict=False)
    error_msg = hsm_shape.error_message

    if error_msg != '':
        sig_guess = 1
    else:
        sig_guess = hsm_shape.moments_sigma

    if method == 'KSB':
        params = galsim.hsm.HSMParams(ksb_sig_weight=sig_guess)
        res = galsim.hsm.EstimateShear(
            Image, psf_Image, shear_est=method, strict=True, hsmparams=params,
        )
        return np.array([res.corrected_g1, res.corrected_g2])
    else:
        for it in range(3):
            flag = 0
            try:
                sig_guess += 2 * it
                res = galsim.hsm.EstimateShear(
                    Image, psf_Image, shear_est=method,
                    guess_sig_gal=sig_guess, guess_sig_PSF=1,
                )
                flag = 1
            except Exception:
                continue
            if flag == 1:
                break

        if flag == 1:
            return np.array([_e_to_g(res.corrected_e1), _e_to_g(res.corrected_e2)])
        else:
            return np.array([np.inf, np.inf])


def _e_to_g(e):
    """Convert ellipticity |e| to reduced shear |g|."""
    if (1 - e) / (1 + e) < 0:
        raise ValueError(f'Invalid ellipticity e={e}')
    axis_ratio = np.sqrt((1 - e) / (1 + e))
    return (1 - axis_ratio) / (1 + axis_ratio)


def galsim_FindMom(im):
    """Measure adaptive moments (e1, e2) using GalSim HSM."""
    Im = galsim.Image(im)
    moms = galsim.hsm.FindAdaptiveMom(Im)
    return moms.observed_shape.e1, moms.observed_shape.e2


# --- Image utilities ---

def cutout(image, x_source, y_source, stamp_size=None,
           effective_radius_source=None, enlarging_factor=None):
    """
    Extract a postage stamp cutout from an image.

    Parameters
    ----------
    image : np.ndarray
        Full image array.
    x_source, y_source : float
        Source position in pixels (1-indexed).
    stamp_size : int
        Size of the cutout in pixels.

    Returns
    -------
    np.ndarray
        Cutout image.
    """
    if effective_radius_source:
        stamp_size = effective_radius_source * enlarging_factor

    return image[
        int(y_source - stamp_size / 2) - 1:int(y_source + stamp_size / 2) - 1,
        int(x_source - stamp_size / 2) - 1:int(x_source + stamp_size / 2) - 1,
    ]


# --- Shear utilities ---

def angle2e(theta, axis_ratio):
    """Convert position angle and axis ratio to ellipticity (e1, e2)."""
    e2 = (1 - axis_ratio) / (1 + axis_ratio) * np.sin(2 * theta)
    e1 = (1 - axis_ratio) / (1 + axis_ratio) * np.cos(2 * theta)
    return e1, e2
