import os.path as pa

import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from myagn.distributions import ConstantPhysicalDensity
from myagn.flares import ConstantRate
from mygw.io.skymaps import GWTCCache, Skymap

from mygwagn.inference.palmese21 import Lambda, mock_data
from mygwagn.io import paths

##############################
###        GW cache        ###
##############################

# Initialize GW cache
cache_dir = pa.join(paths.cache_dir, "tmp")
gwcache = GWTCCache(cache_dir=cache_dir)

##############################
###       Mock data        ###
##############################

# Generate mock dataset
lambda_ = 0.3
gw_skymaps, agn_flares, assoc_matrix = mock_data(
    gwcache.cache_dir,
    lambda_=lambda_,
    agn_distribution=ConstantPhysicalDensity(10**-4.75 * u.Mpc**-3),
    flare_model=ConstantRate(1e-4 / 200 / u.day),
    n_gw=100,
    ci_followup=0.9,
    Dt_followup=200 * u.day,
    background_skymap_level=5,
    background_z_grid=np.linspace(0, 1, 100),
    background_z_frac=0.9,
    cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
    rng_np=np.random.default_rng(12345),
    use_exact_rates=False,
    independent_gws=False,
)

##############################
###       Inference        ###
##############################

# Initialize agn distribution
agn_distribution = ConstantPhysicalDensity(10**-4.75 * u.Mpc * -3)

# Initialize flare model
flare_model = ConstantRate(1e-4 / 200 / u.day)

# Initialize inference
inference = Lambda(
    gw_skymaps=gw_skymaps,
    agn_flares=agn_flares,
    assoc_matrix=assoc_matrix,
    agn_distribution=agn_distribution,
    flare_model=flare_model,
)

# Run inference
sampler = inference.run_mcmc(
    lambda_0=0.5,
)

# Save results
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
np.savetxt("flat_samples.npy", flat_samples)

##############################
###        Cleanup         ###
##############################

# Clear gw cache
# gwcache.clear_cache()
