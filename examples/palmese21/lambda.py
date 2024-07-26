import os.path as pa

import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from myagn.distributions import ConstantPhysicalDensity
from myagn.flares.models import ConstantRate
from mygw.io.skymaps import GWTCCache, Skymap

from mygwagn.inference.palmese21 import Framework, Lambda, mock_data
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
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
gw_skymaps, agn_flares, assoc_matrix = mock_data(
    pa.join(gwcache.cache_dir, "GWTC2.1"),
    lambda_=lambda_,
    agn_distribution=ConstantPhysicalDensity(10**-4.75 * u.Mpc**-3),
    flare_model=ConstantRate(1e-4 / 200 / u.day),
    n_gw=50,
    ci_followup=0.9,
    Dt_followup=200 * u.day,
    background_skymap_level=5,
    background_z_grid=np.linspace(0, 2, 201),
    background_z_frac=0.9,
    cosmo=cosmo,
    rng=np.random.default_rng(12345),
    use_exact_rates=False,
    independent_gws=False,
)

print("# gw_skymaps:", len(gw_skymaps))
print("# agn_flares:", len(agn_flares))
print("assoc_matrix.shape:", assoc_matrix.shape)

##############################
###       Inference        ###
##############################

# Initialize agn distribution
agn_distribution = ConstantPhysicalDensity(10**-4.75 * (u.Mpc**-3))

# Initialize flare model
flare_model = ConstantRate(1e-4 / 200 / u.day)

# Initialize inference
framework = Framework(
    gw_skymaps=gw_skymaps,
    agn_flares=agn_flares,
    assoc_matrix=assoc_matrix,
    agn_distribution=agn_distribution,
    flare_model=flare_model,
)

# Initialize Lambda
lambda_0 = 0.3
inf_lambda = Lambda()

# Run inference
sampler = inf_lambda.run_mcmc(
    inference=framework,
    lambda_0=lambda_0,
    cosmo=cosmo,
    n_steps=5000,
    n_walkers=32,
    n_proc=32,
)

# Save results
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
np.savetxt("flat_samples.npy", flat_samples)

##############################
###        Cleanup         ###
##############################

# Clear gw cache
# gwcache.clear_cache()
