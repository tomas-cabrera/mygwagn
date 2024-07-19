import glob
import multiprocessing as mp
import os.path as pa

import astropy.units as u
import emcee
import healpy as hp
import ligo.skymap.moc as lsm_moc
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.time import Time
from astropy.table import QTable
from ligo.skymap.io.fits import write_sky_map
from myagn.flares.flares import Flare
from mygw.io.healvox import HEALVox
from mygw.io.skymaps import Skymap

from mygwagn.io import paths

# Define auxiliary directory
aux_dir = pa.join(paths.mygwagn_dir, "examples", "palmese21", "aux")


def _calc_active_skymap(active_skymaps):
    ### HPXs

    # Get all HEALPix indices
    hpx_arr = np.array([asm["INCIFOLLOWUP"] for asm in active_skymaps])

    # Highlight all active hpxs
    hpxs = np.any(hpxs, axis=0)

    ### ZMIN, ZMAX
    # Get zmins, zmaxs
    zmin_arr = np.array([asm["ZMIN"] for asm in active_skymaps])
    zmax_arr = np.array([asm["ZMAX"] for asm in active_skymaps])

    # Set inactive zmins, zmaxs to np.nan
    zmin_arr[~hpxs] = np.nan
    zmax_arr[~hpxs] = np.nan

    # Get min, max zmins, zmaxs
    zmin = np.nanmin(zmin_arr, axis=0)
    zmax = np.nanmax(zmax_arr, axis=0)

    # Set inactive hpx zs to 0.5  (this causes the volume to be 0 for these pixels)
    zmin[~hpxs] = 0.5
    zmax[~hpxs] = 0.5

    ### Wrap-up

    # Compile results
    active_skymap = QTable([hpxs, zmin, zmax], names=["INCIFOLLOWUP", "ZMIN", "ZMAX"])

    # Return
    return active_skymap


def mock_data(
    skymap_dir,
    lambda_,
    agn_distribution,
    flare_model,
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
    position_flare_gws = [
        sm.draw_random_location(np.linspace(0, 1, 1000), ci_followup)
        for sm in skymap_flare_gws
    ]

    # Set times using distribution after GW event
    if use_exact_rates:
        times_flare_gws = [Dt_followup / 2] * n_flares_gw
    else:
        times_flare_gws = (
            rng_np.uniform(0, Dt_followup.to(u.day).value, n_flares_gw) * u.day
        )

    # Combine coordinates
    coord_flare_gws = [[t, *x] for t, x in zip(times_flare_gws, position_flare_gws)]

    ### Background flares

    # Get GW event times, order by time
    # NOTE: this will probably not work; will need to adjust Skymap to read the time from the header
    time_gws = [Time(sm["DATE_OBS"], format="isot", scale="utc") for sm in gw_skymaps]
    ind_gw_time = np.argsort(time_gws)

    # Iterate over GW events in time order
    ti_actives = []
    time_actives = []
    skymap_actives = []
    agn_flares = []
    assoc_matrix = []
    for ti, ti_next in zip(ind_gw_time, np.roll(ind_gw_time, -1)):

        ##############################
        ###  Add activated skymap  ###
        ##############################

        # Get time, skymap
        time_ti = time_gws[ti]
        skymap_ti = gw_skymaps[ti]

        # Flatten skymap
        skymap_ti_flat = skymap_ti.flatten(background_skymap_level)

        # Add mask for hpxs in ci_followup
        hpxs = skymap_ti_flat.get_hpxs_for_ci_areas(ci_followup)[0]
        skymap_ti_flat["INCIFOLLOWUP"] = False
        skymap_ti_flat["INCIFOLLOWUP"][hpxs] = True

        ## Add columns for minimum/maximum redshift
        # Iterate over healpixs
        zmins = []
        zmaxs = []
        for i in np.arange(len(skymap_ti_flat)):
            # Get healpix
            hpx = skymap_ti_flat[i]

            # Get dp_ddL
            dp_dz = skymap_ti.dp_dz(background_z_grid, hpx)

            # Ensure probability sums to at least background_z_frac
            if np.trapezoid(dp_dz, background_z_grid) < background_z_frac:
                raise ValueError(
                    "dp_dz does not sum to at least background_z_frac; perhaps expand background_z_grid?"
                )

            # Get background_z_frac bounds
            argsort_dp_dz = np.argsort(dp_dz)[::-1]
            dp_dz_sorted = dp_dz[argsort_dp_dz]
            dp_dz_cumsum = np.cumsum(dp_dz_sorted)
            ind_ci_cutoff = np.searchsorted(dp_dz_cumsum, background_z_frac)
            z_grid_ci = background_z_grid[argsort_dp_dz[:ind_ci_cutoff]]
            zmin = z_grid_ci.min()
            zmax = z_grid_ci.max()

            # Append to lists
            zmins.append(zmin)
            zmaxs.append(zmax)

        # Add to skymap
        skymap_ti_flat["ZMIN"] = zmins
        skymap_ti_flat["ZMAX"] = zmaxs

        # Add to active lists
        ti_actives.append(ti)
        time_actives.append(time_ti)
        skymap_actives.append(skymap_ti_flat)

        # Recalculate active skymap
        active_skymap = _calc_active_skymap(skymap_actives)

        ##############################
        ###Process terminating GWs ###
        ##############################

        # Get time for next GW event; if ti_next has cycled to the end, set to far future
        if ti_next == ind_gw_time[0]:
            time_next = Time("9999-12-31T23:59:59", format="isot", scale="utc")
        else:
            time_next = time_gws[ti_next]

        # Iterate through times (assumes the actives have remained in time order)
        time_start = time_ti
        i_terminate = 0
        time_terminate = time_actives[i_terminate]
        while time_terminate < time_next:

            ##############################
            ###  Get number of flares  ###
            ##############################

            # Calculate n_agn in active volume
            # TODO: Broaden this to allow for redshift-dependent AGN distributions and other flare models
            # Calculate prism volumes for each hpx by dividing comoving volume difference by number of pixels
            # prism volume = (4/3 pi d_comoving(zmax)^3 - 4/3 pi d_comoving(zmax)^3) / n_pix
            active_volume_prisms = cosmo.comoving_volume(
                active_skymap["ZMAX"],
            ) - cosmo.comoving_volume(
                active_skymap["ZMIN"],
            ) / hp.order2npix(
                background_skymap_level
            )
            # Sum to get total active volume
            active_volume = active_volume_prisms.sum()

            # Calculate n_flares by multiplying by flat rates
            n_flares = (
                active_volume
                * (10**-4.75 * u.Mpc**-3)
                * flare_model.flare_rate()
                * (time_terminate - time_start)
            )

            ##############################
            ###Sample flare coordinates###
            ##############################

            # Sample flare times
            time_flare_backgrounds = (
                rng_np.uniform(time_start.mjd, time_terminate.mjd, n_flares) * u.day
            )

            # Sample hpx; get ra, dec
            hpx_flare_backgrounds = rng_np.choice(
                np.arange(len(active_skymap)),
                n_flares,
                p=active_skymap["INCIFOLLOWUP"],
                replace=True,
            )
            ra_flare_backgrounds, dec_flare_backgrounds = (
                hp.pix2ang(
                    hp.order2nside(background_skymap_level),
                    hpx_flare_backgrounds,
                    lonlat=True,
                )
                * u.deg
            )

            # Sample redshifts
            z_flare_backgrounds = []
            for hpx in hpx_flare_backgrounds:
                # Choose redshift
                z = agn_distribution.sample_z(
                    1,
                    z_grid=np.linspace(
                        active_skymap["ZMIN"][hpx],
                        active_skymap["ZMAX"][hpx],
                        100,
                    ),
                    rng_np=rng_np,
                )[0]

                # Append to list
                z_flare_backgrounds.append(z)

            # Compile flare coordinates
            coord_flares = np.array(
                [
                    time_flare_backgrounds,
                    ra_flare_backgrounds,
                    dec_flare_backgrounds,
                    z_flare_backgrounds,
                ]
            ).T

            ##############################
            ###  Update assoc_matrix   ###
            ##############################

            # Add gw flares in time interval
            for cfgw in coord_flare_gws:
                if time_start <= cfgw[0] < time_terminate:
                    coord_flares = np.vstack([coord_flares, cfgw])

            # Iterate over flares
            for cf in coord_flares:
                # Initialize row
                assoc_row = np.zeros(n_gw, dtype=bool)

                # Get flare hpx
                hpx = hp.ang2pix(
                    hp.order2nside(background_skymap_level),
                    cf[1].deg,
                    cf[2].deg,
                    lonlat=True,
                    nest=True,
                )

                # Iterate over active GWs
                for tia in ti_actives:
                    if skymap_actives[tia]["INCIFOLLOWUP"][hpx]:
                        assoc_row[tia] = True

                # Append to agn_flares, assoc_matrix
                agn_flares.append(Flare(*cf))
                assoc_matrix.append(assoc_row)

            ##############################
            ###      Cleanup loop      ###
            ##############################

            # Update time_start
            time_start = time_terminate

            # Update i_terminate, time_terminate
            i_terminate += 1
            time_terminate = time_actives[i_terminate]

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
