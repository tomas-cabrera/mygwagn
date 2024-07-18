import multiprocessing as mp
import glob
import os.path as pa

import astropy.units as u
import emcee
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM

from mygw.io.skymaps import Skymap
from myagn.flares.flares import Flare


def mock_data(
    skymap_dir,
    lambda_,
    agn_distribution,
    n_gw=100,
    ci_followup=0.9,
    cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
    rng_np=np.random.default_rng(12345),
    use_exact_rates=False,
    independent_gws=False,
):
    """Generates a mock observation dataset for this analysis."""
    ##############################
    ###        GW events       ###
    ##############################

    # Get skymap filenames
    skymap_filenames = glob.glob(pa.join(skymap_dir, "*.fits"))

    # Choose n_gw skymaps
    skymap_filenames = rng_np.choice(skymap_filenames, n_gw, replace=False)

    # Initialize skymaps
    gw_skymaps = [Skymap(f) for f in skymap_filenames]

    ##############################
    ###        AGN flares      ###
    ##############################

    ### GW counterpart flares

    # Draw n_flares_gw from Poisson distribution
    n_flares_gw_expected = lambda_ * n_gw
    if use_exact_rates:
        n_flares_gw = round(n_flares_gw_expected)
    else:
        n_flares_gw = rng_np.poisson(n_flares_gw_expected)
        n_flares_gw = max(n_flares_gw, n_gw)

    # Select gw events for flares
    i_gws = np.arange(n_gw)
    i_flare_gws = rng_np.choice(i_gws, n_flares_gw, replace=False)
    skymap_flare_gws = [gw_skymaps[i] for i in i_flare_gws]

    # Select flare locations
    coord_flare_gws = [
        sm.draw_random_location(np.linspace(0, 1, 1000), ci_followup)
        for sm in skymap_flare_gws
    ]

    # Set times using distribution after GW event
    TODO

    ### Background flares

    # Get flares
    TODO  # Need to be careful about overlapping GW events!
    # First idea:
    # - Initialize a time array spanning the entire followup period (t_first_gw to t_last_gw + 200 days)
    # - Have a time x gw boolean array indicating whether a follow-up is present at a given time
    # - Flatten all GW skymaps, initialize into boolean maps of the ci_followup regions
    # - Iterate through the timeline:
    #     - For each time, combine the active GW maps into a single map highlighting all the active HEALPix tiles
    #     - Get the number of AGN in the follow-up volume (via the given AGN distribution)
    #     - Given the timestep and flare model, calculate the expected number of AGN flares in this volume-time
    #     - Choose an actual number of flares as appropriate to use_exact_rates
    #     - Draw flare locations from the volume (via AGN distribution), flare times from the time interval (probably can just be uniformly chosen)
    #     - Record GW-flare associations (which GW follow-ups are active at the time of the flares?)
    #         - Just append boolean lists to a running list, one append per flare
    #         - The appended lists having dimensions of 1 x n_gw, indicating which gw follow-ups were active during the flare
    #         - NOTE: This will require transposing the assoc_matrix before returning
    #         - Include GW-sourced flares when determining associations!
    #         - NOTE: This will require appending to a running agn_flares list to keep the indexing correct

    ##############################
    ###        Return          ###
    ##############################

    # Transpose assoc_matrix to comply with n_gw x n_flares convention
    assoc_matrix = np.transpose(assoc_matrix)

    return gw_skymaps, agn_flares, assoc_matrix


class Palmese21:
    """Base class for Palmese21-like analyses."""

    def __init__(
        self,
        gw_skymaps,
        agn_flares,
        assoc_matrix,
        agn_distribution,
        flare_model,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        z_grid=np.linspace(0, 1, 1000),
        ci_prob=0.9,
    ):
        # Forward objects
        self.gw_skymaps = gw_skymaps
        self.agn_flares = agn_flares
        self.assoc_matrix = assoc_matrix  # Boolean array, shape: n_gw x n_flares
        self.agn_distribution = agn_distribution
        self.flare_model = flare_model
        self.cosmo = cosmo
        self.z_grid = z_grid
        self.ci_prob = ci_prob

        # Set gw/flare indices
        self.ind_gw, self.ind_flare = np.where(self.assoc_matrix)

        # Set flare hpxs (same dimensions as assoc_matrix)
        self.hpx_flares = np.array(
            [
                [sm.skycoord_to_healpix(SkyCoord(f.ra, f.dec)) for f in self.agn_flares]
                for sm in self.gw_skymaps
            ]
        )

    def get_cosmo(self, cosmo):
        """Returns default cosmo if None, else returns input."""
        if cosmo is None:
            return self.cosmo
        return cosmo

    def calc_s_arr(self, cosmo=None):
        """Calculate signal terms for likelihood.

        Returns
        -------
        _type_
            _description_
        """

        # Set cosmo
        cosmo = self.get_cosmo(cosmo)

        # Initialize s_arr
        s_arr = np.zeros((len(self.gw_skymaps), len(self.agn_flares)))

        # Get skymaps, flares
        skymaps = self.gw_skymaps[self.ind_gw]
        flares = self.agn_flares[self.ind_flare]

        # Get hpx indices of flare locations
        hpx_flares = self.hpx_flares[self.ind_gw, self.ind_flare]

        # Calculate dp_dOmega
        dp_dOmega = (
            u.quantity([sm.dp_dOmega(i) for sm, i in zip(skymaps, hpx_flares)])
            / self.ci_prob
        )

        # Calculate dp_dz
        dp_dz = u.quantity(
            [
                sm.dp_dz(self.z_grid, i, f.z)
                for sm, f, i in zip(skymaps, flares, hpx_flares)
            ]
        )

        # Combine probability densities
        s_values = dp_dOmega * dp_dz

        # Set values in s_arr
        s_arr[self.assoc_matrix] = s_values

        return s_arr  # Shape: (n_skymaps, n_flares)

    def calc_b_arr(self, cosmo=None, brightness_limits=None):
        """Calculate background terms for likelihood.

        Parameters
        ----------
        cosmo : _type_, optional
            _description_, by default None
        brightness_limits : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """

        # Set cosmo
        cosmo = self.get_cosmo(cosmo)

        # Initialize b_arr
        b_arr = np.zeros((len(self.gw_skymaps), len(self.agn_flares)))

        # Fetch flares, flare redshifts
        flares = self.agn_flares[self.ind_flare]
        z_flares = [f.z for f in flares]

        # Calculate values
        b_values = self.agn_distribution.dp_dOmega_dz(
            z_grid=self.z_grid,
            z_evaluate=z_flares,
            cosmo=cosmo,
            brightness_limits=brightness_limits,
        ) * self.flare_model.p_flare(flares)

        # Set values in b_arr
        b_arr[self.assoc_matrix] = b_values

        return b_arr  # Shape: (n_skymaps, n_flares)

    def calc_mu_arr(self, lambda_, cosmo=None):
        return lambda_ * np.ones(len(self.gw_skymaps))


class Lambda(Palmese21):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_arrs(self, cosmo=None):
        # Get cosmo
        cosmo = self.get_cosmo(cosmo)

        # Calculate signal terms
        self.s_arr = self.calc_s_arr(cosmo=cosmo)

        # Calculate background terms
        self.b_arr = self.calc_b_arr(cosmo=cosmo)

        # Calculate mu terms
        self.mu_arr = self.calc_mu_arr(cosmo=cosmo)

    def ln_prior(self, lambda_):
        # If in ok domain, return 0; otheriwse return -inf
        if 0.0 < lambda_ < 1.0:
            return 0.0
        return -np.inf

    def ln_likelihood(self, lambda_):
        # Calculate likelihood
        ln_likelihood = (
            np.sum(np.log(lambda_ * self.s_arr + self.b_arr)) - self.mu_arr.sum()
        )

        return ln_likelihood

    def ln_posterior(self, lambda_):
        # Evaluate prior
        ln_prior = self.ln_prior(lambda_)

        # Evaluate likelihood
        ln_likelihood = self.ln_likelihood(lambda_, self.s_arr, self.b_arr, self.mu_arr)

        # Combine into posterior
        ln_posterior = ln_prior + ln_likelihood

        return ln_posterior

    def run_mcmc(self, lambda_0=0.5, cosmo=None, n_walkers=32, n_steps=5000, n_proc=32):
        # Get cosmo
        cosmo = self.get_cosmo(cosmo)

        # Set arrays
        self.set_arrs(cosmo=cosmo)

        # Initial lambdas
        initial_state = lambda_0 + np.random.randn(n_walkers)

        # Define sampler args
        args_sampler = {
            "nwalkers": n_walkers,
            "ndim": 1,
            "log_prob_fn": self.ln_posterior,
        }

        # Define sampler kwargs
        kwargs_sampler = {}
        # Define run_mcmc args
        args_run_mcmc = {
            "initial_state": initial_state,
            "n_steps": n_steps,
        }
        # Define run_mcmc kwargs
        kwargs_run_mcmc = {
            "progress": True,
        }

        # If n_proc=1, run without multiprocessing
        if n_proc == 1:
            # Initialize sampler
            sampler = emcee.EnsembleSampler(
                **args_sampler,
                **kwargs_sampler,
            )

            # Run sampler
            sampler.run_mcmc(**args_run_mcmc, **kwargs_run_mcmc)
        # Else run with multiprocessing
        else:
            with mp.Pool(n_proc) as pool:
                # Initialize sampler
                sampler = emcee.EnsembleSampler(
                    **args_sampler,
                    **kwargs_sampler,
                    pool=pool,
                )

                # Run sampler
                sampler.run_mcmc(**args_run_mcmc, **kwargs_run_mcmc)

        return sampler


class LambdaH0(Palmese21):
    pass


class LambdaH0Om0(Palmese21):
    pass
