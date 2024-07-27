import os.path as pa
import matplotlib.pyplot as plt

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

# Initialize agn distribution
agn_distribution = ConstantPhysicalDensity(10**-4.75 * (u.Mpc**-3))

# Initialize flare model
rate = 1e-6
flare_model = ConstantRate(rate / 200 / u.day)


##############################
###       Mock data        ###
##############################

# Generate mock dataset
lambda_mock = 0.2
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
gw_skymaps, agn_flares, assoc_matrix = mock_data(
    pa.join(gwcache.cache_dir, "GWTC2.1"),
    lambda_=lambda_mock,
    agn_distribution=agn_distribution,
    flare_model=flare_model,
    n_gw=50,
    ci_followup=0.9,
    Dt_followup=200 * u.day,
    background_skymap_level=5,
    background_z_grid=np.linspace(0, 2, 201),
    background_z_frac=0.9,
    cosmo=cosmo,
    rng=np.random.default_rng(12345),
    use_exact_rates=True,
    independent_gws=False,
)

##############################
###       Inference        ###
##############################

# Initialize inference
framework = Framework(
    gw_skymaps=gw_skymaps,
    agn_flares=agn_flares,
    assoc_matrix=assoc_matrix,
    agn_distribution=agn_distribution,
    flare_model=flare_model,
)

# Initialize Lambda
lambda_0 = 0.5
inf_lambda = Lambda()

# Iterate over likelihoods
results = {}
for likelihood in ["palmese21", "modified"]:
    # Run inference
    sampler = inf_lambda.run_mcmc(
        inference=framework,
        lambda_0=lambda_0,
        likelihood=likelihood,
        cosmo=cosmo,
        n_steps=5000,
        n_walkers=32,
        n_proc=32,
    )

    # Save results
    flat_samples = sampler.get_chain(
        discard=100,
        thin=15,
        flat=True,
    )

    # Save results
    results[likelihood] = flat_samples

# Make histogram
for likelihood, samples in results.items():
    plt.hist(
        samples,
        bins=50,
        label=likelihood,
        histtype="step",
    )
plt.gca().axvline(lambda_mock, label="True value")
plt.legend()
plt.savefig(f"likelihood_comparison.lambda_{lambda_mock}.rate-{rate:e}.png")

##############################
###        Cleanup         ###
##############################

# Clear gw cache
# gwcache.clear_cache()
