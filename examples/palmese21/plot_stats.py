import json
import os
import os.path as pa

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from tqdm import tqdm


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


if __name__ == "__main__":

    # Define statsdir
    statsdir = "/hildafs/projects/phy220048p/tcabrera/bbhagn/palmese21_test/likelihood_comparison_3/stats"

    # Get stats files
    stats_files = os.listdir(statsdir)
    stats_files = [sf for sf in stats_files if sf.endswith(".txt")]

    # Iterate over stats files
    df_stats = []
    for sf in tqdm(stats_files):
        # Initialize tempdict
        tempdict = {}

        # Extract lambda, rate, seed
        sf_split = sf.split(".")
        for l in ["lambda_input", "rate", "seed"]:
            for sf_s in sf_split:
                if l in sf_s:
                    tempdict[l] = float(sf_s.split("-")[-1])
                    break

        # Load file
        with open(pa.join(statsdir, sf), "r") as f:
            stats = json.load(f)

        # Extract stats
        for l in ["lambda_actual", "n_gw", "n_flares", "n_flares_gw", "n_flares_bg"]:
            tempdict[l] = stats[l]

        # Iterate over likelihoods
        for lk in ["palmese21", "multiflare", "multigw"]:
            # Get lk stats
            lk_stats = stats[lk]

            # Extract stats
            tempdict[f"{lk}_median"] = np.median(lk_stats["samples"])

            # Calculate HPD
            hpd, x, y_kde, modes, kde = hpd_grid(
                lk_stats["samples"],
                alpha=1 - 0.68,
                x=np.linspace(0, 1, 1000),
            )

            # Extract stats
            tempdict[f"{lk}_mode"] = x[np.argmax(y_kde)]
            tempdict[f"{lk}_0.68_lo"] = hpd[0][0]
            tempdict[f"{lk}_0.68_hi"] = hpd[0][1]

        # Append to df_stats
        df_stats.append(tempdict)

    # Create DataFrame
    df_stats = pd.DataFrame(df_stats)

    # Errorbars
    for lk in ["palmese21", "multiflare", "multigw"]:
        df_stats[f"{lk}_0.68_errlo"] = (
            df_stats[f"{lk}_mode"] - df_stats[f"{lk}_0.68_lo"]
        )
        df_stats[f"{lk}_0.68_errhi"] = (
            df_stats[f"{lk}_0.68_hi"] - df_stats[f"{lk}_mode"]
        )

    # Create fractions
    df_stats["frac_flares_gw"] = df_stats["n_flares_gw"] / df_stats["n_flares"]
    df_stats["frac_flares_bg"] = df_stats["n_flares_bg"] / df_stats["n_flares"]
    df_stats["ratio_flares_gwbg"] = df_stats["n_flares_gw"] / df_stats["n_flares_bg"]

    ### Plot
    for x in [
        "n_flares",
        "n_flares_gw",
        "n_flares_bg",
        "frac_flares_gw",
        "frac_flares_bg",
        "ratio_flares_gwbg",
    ]:
        # Initialize figure
        fig, ax = plt.subplots()
        for lk, color in zip(
            ["palmese21", "multiflare", "multigw"],
            ["xkcd:blue", "xkcd:orange", "xkcd:green"],
        ):
            ax.errorbar(
                df_stats[x],
                df_stats[f"{lk}_mode"] - df_stats["lambda_actual"],
                [df_stats[f"{lk}_0.68_errlo"], df_stats[f"{lk}_0.68_errhi"]],
                color=color,
                label=lk,
                alpha=0.5,
                marker="o",
                markersize=3,
                linestyle="",
            )
        ax.set_xlabel(x)
        ax.set_ylabel(f"{lk}_mode - lambda_actual")
        ax.legend()
        plt.tight_layout()
        figpath = pa.join(statsdir, f"{x}.png")
        plt.savefig(figpath)
        plt.close()
