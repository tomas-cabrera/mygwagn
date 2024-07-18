import os.path as pa

import astropy.units as u
import numpy as np
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
gw_skymaps, agn_flares, assoc_matrix = mock_data(
    gwcache,
    lambda_=0.3,
)

##############################
###       Inference        ###
##############################

# Initialize agn distribution
agn_distribution = ConstantPhysicalDensity(10**-4.75 * u.Mpc * -3)

# Initialize flare model
flare_model = ConstantRate(1e-4)

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
