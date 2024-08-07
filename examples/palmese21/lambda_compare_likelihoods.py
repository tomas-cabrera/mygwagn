import json
import os.path as pa

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from myagn.distributions import ConstantPhysicalDensity
from myagn.flares.models import ConstantRate
from mygw.io.skymaps import GWTCCache, Skymap
from scipy.stats import gaussian_kde

from mygwagn.inference.palmese21 import Framework, Lambda, mock_data
from mygwagn.io import paths


def hpd_grid(sample, alpha=0.05, roundto=2, x=None):
    """Calculate highest posterior density (HPD) of array for given alpha.
    The HPD is the minimum width Bayesian credible interval (BCI).
    The function works for multimodal distributions, returning more than one mode
    (TC: This function is modified from https://github.com/aloctavodia/BAP/blob/master/first_edition/code/Chp1/hpd.py)

    Parameters
    ----------

    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    roundto: integer
        Number of digits after the decimal point for the results
    x : list-like
        x domain to use for kde
    Returns
    ----------
    hpd: array with the lower

    """
    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]
    density = gaussian_kde(sample)
    # get upper and lower bounds
    l = np.min(sample)
    u = np.max(sample)
    if type(x) == type(None):
        x = np.linspace(l, u, 2000)
    y = density.evaluate(x)
    # y = density.evaluate(x, l, u) waitting for PR to be accepted
    xy_zipped = zip(x, y / np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0
    hdv = []
    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1 - alpha):
            break
    hdv.sort()
    diff = (u - l) / 20  # differences of 5%
    hpd = []
    hpd.append(round(min(hdv), roundto))
    for i in range(1, len(hdv)):
        if hdv[i] - hdv[i - 1] >= diff:
            hpd.append(round(hdv[i - 1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))
    ite = iter(hpd)
    hpd = list(zip(ite, ite))
    modes = []
    # print("x:", x)
    # print("y:", y)
    # print("hpd:", hpd)
    for value in hpd:
        x_hpd = x[(x > value[0]) & (x < value[1])]
        y_hpd = y[(x > value[0]) & (x < value[1])]
        modes.append(round(x_hpd[np.argmax(y_hpd)], roundto))
    return hpd, x, y, modes, density


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
seeds = [12345, 13579, 54321, 67890, 98765]
seeds = np.random.default_rng(12345).choice(2**32, 30, replace=False)

for simseed in seeds:
    for lambda_mock in lambda_mocks:
        for rate in rates:
            print(f"lambda_mock={lambda_mock}, rate={rate:.1e}, seed={simseed}")

            # Skip if figure already generated
            figpath = f"likelihood_comparison.lambda-{lambda_mock}.rate-{rate:.1e}.seed-{simseed}.png"
            # figpath = f"likelihood_comparison.lambda-{lambda_mock}.rate-{rate:.1e}.png"
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
                rng=np.random.default_rng(simseed),
                use_exact_rates=False,
                independent_gws=False,
            )

            for infseed in seeds[:1]:

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
                lambda_kdes = {}
                for likelihood in ["palmese21", "multiflare", "multigw"]:
                    # Run inference
                    sampler = inf_lambda.run_mcmc(
                        inference=framework,
                        lambda_0=lambda_0,
                        likelihood=likelihood,
                        cosmo=cosmo,
                        n_steps=5000,
                        n_walkers=32,
                        n_proc=32,
                        rng=np.random.default_rng(infseed),
                    )

                    # Get samples
                    flat_samples = sampler.get_chain(
                        discard=100,
                        thin=15,
                        flat=True,
                    )

                    # Assemble return dict
                    tempdict = {
                        "samples": list(flat_samples.flatten()),
                        "s_arr": inf_lambda.s_arr.tolist(),
                        "b_arr": inf_lambda.b_arr.tolist(),
                        "assoc_matrix": inf_lambda.assoc_matrix.tolist(),
                    }

                    # Save results
                    results[likelihood] = tempdict

                # Make histogram
                for (likelihood, tempdict), color in zip(
                    results.items(), ["xkcd:blue", "xkcd:orange", "xkcd:green"]
                ):
                    # n, b, a = plt.hist(
                    #     samples,
                    #     bins=50,
                    #     label=likelihood,
                    #     histtype="step",
                    #     color=color,
                    # )

                    # # Get 1sigma interval
                    # low, high = np.percentile(samples, [16, 84])
                    # mask = [(bl >= low and bl <= high) for bl in b]
                    # plt.hist(
                    #     b[mask][:-1],
                    #     b[mask],
                    #     weights=n[mask[:-1]][:-1],
                    #     alpha=0.5,
                    #     color=color,
                    # )

                    samples = tempdict["samples"]

                    # Generate kde and intervals with hpd_grid
                    dict_hpd = {}
                    colors = ["r", "b"]
                    for ii, i in enumerate([0.68]):
                        hpd, x, y_kde, modes, kde = hpd_grid(
                            samples,
                            alpha=1 - i,
                            x=np.linspace(0, 1, 1000),
                        )
                        dict_hpd[str(i)] = dict(
                            zip(
                                ["hpd", "x", "y", "modes"],
                                [hpd, x, y_kde, modes],
                            )
                        )
                        lambda_kdes[f"{likelihood}_mode"] = x[np.argmax(y_kde)]
                        lambda_kdes[f"{likelihood}_0.68_lo"] = hpd[0][0]
                        lambda_kdes[f"{likelihood}_0.68_hi"] = hpd[0][1]

                        # Plot intervals
                        mask = [(v >= hpd[0][0] and v <= hpd[0][1]) for v in x]
                        kw_kde_fill = {
                            "color": color,
                            "alpha": 0.5,
                        }
                        if likelihood == "palmese21":
                            kw_kde_fill["zorder"] = 2
                        plt.fill_between(
                            x[mask],
                            np.zeros(x[mask].shape),
                            y_kde[mask],
                            **kw_kde_fill,
                        )
                        # kw_quants = {
                        #     "color": color,
                        # }
                        # plt.vlines(
                        #     hpd,
                        #     ymin=0,
                        #     ymax=kde.evaluate(hpd),
                        #     **kw_quants,
                        # )

                    # Plot
                    kw_kde_line = {
                        "color": color,
                        "alpha": 0.9,
                        "label": likelihood,
                    }
                    if likelihood == "palmese21":
                        kw_kde_line["zorder"] = 2
                    plt.plot(
                        x,
                        y_kde,
                        **kw_kde_line,
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
            plt.ylabel("PDF")
            plt.legend(title=f"flares/AGN={rate:.1e}")
            plt.savefig(figpath)
            plt.close()

            # Save file
            statpath = figpath.replace(".png", ".txt")
            data = {
                "lambda_actual": lambda_actual,
                "n_gw": len(gw_skymaps),
                "n_flares": len(agn_flares),
                "n_flares_gw": lambda_actual * len(gw_skymaps),
                "n_flares_bg": len(agn_flares) - lambda_actual * len(gw_skymaps),
            }
            for likelihood, tempdict in results.items():
                data[likelihood] = {
                    **tempdict,
                    "median": np.median(tempdict["samples"]),
                    "mode": lambda_kdes[f"{likelihood}_mode"],
                    "0.68_lo": lambda_kdes[f"{likelihood}_0.68_lo"],
                    "0.68_hi": lambda_kdes[f"{likelihood}_0.68_hi"],
                }
            with open(statpath, "w", encoding="utf-8") as f:
                json.dump(
                    data,
                    f,
                    ensure_ascii=False,
                    indent=4,
                )

##############################
###        Cleanup         ###
##############################

# Clear gw cache
# gwcache.clear_cache()
