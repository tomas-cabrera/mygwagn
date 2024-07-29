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

# Initialize lambdas
lambda_mocks = [0.1, 0.2, 0.3]

# Initialize flare model
rates = [1e-4, 1e-5, 1e-6]

# Initialize random seeds
seeds = [12345, 54321, 67890]

for seed in seeds:
    for lambda_mock in lambda_mocks:
        for rate in rates:

            # Skip if figure already generated
            figpath = f"likelihood_comparison.lambda-{lambda_mock}.rate-{rate:.1e}.seed-{seed}.png"
            if pa.exists(figpath):
                continue

            ##############################
            ###       Mock data        ###
            ##############################

            # Initialize flare model
            flare_model = ConstantRate(rate / 200 / u.day)

            # Generate mock dataset
            cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
            gw_skymaps, agn_flares, assoc_matrix, lambda_actual = mock_data(
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
                rng=np.random.default_rng(seed),
                use_exact_rates=False,
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
                    rng=np.random.default_rng(seed),
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
            for (likelihood, samples), color in zip(
                results.items(), ["xkcd:blue", "xkcd:orange"]
            ):
                n, b, a = plt.hist(
                    samples,
                    bins=50,
                    label=likelihood,
                    histtype="step",
                    color=color,
                )
                # Get 1sigma interval
                low, high = np.percentile(samples, [16, 84])
                mask = [(bl >= low and bl <= high) for bl in b]
                plt.hist(
                    b[mask][:-1],
                    b[mask],
                    weights=n[mask[:-1]][:-1],
                    alpha=0.5,
                    color=color,
                )
            plt.gca().axvline(
                lambda_actual,
                label=r"$\lambda_{\rm actual}$" + f"={lambda_actual}",
                color="k",
                lw=2,
            )
            plt.gca().axvline(
                lambda_mock,
                label=r"$\lambda_{\rm input}$" + f"={lambda_mock}",
                color="k",
                alpha=0.5,
                ls="--",
            )
            plt.xlim(0, 1)
            plt.xlabel(r"$\lambda$")
            plt.ylabel(r"$n_{\rm samples}$")
            plt.legend(title=f"flares/AGN={rate:.1e}")
            plt.savefig(figpath)
            plt.close()

##############################
###        Cleanup         ###
##############################

# Clear gw cache
# gwcache.clear_cache()
