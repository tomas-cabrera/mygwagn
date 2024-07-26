import glob
import multiprocessing as mp
import os.path as pa
from copy import copy

import astropy.units as u
import astropy_healpix as ah
import emcee
import healpy as hp
import ligo.skymap.moc as lsm_moc
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import QTable
from astropy.time import Time
from ligo.skymap.io.fits import write_sky_map
from ligo.skymap.postprocess.crossmatch import crossmatch
from myagn.flares.flares import Flare
from mygw.io.skymaps import Skymap

from mygwagn.io import paths

# Define auxiliary directory
aux_dir = pa.join(paths.mygwagn_dir, "examples", "palmese21", "aux")


def _calc_active_skymap(active_skymaps):
    ### HPXs

    # Get all HEALPix indices
    hpx_arr = np.array([asm["INCIFOLLOWUP"] for asm in active_skymaps])

    # Highlight all active hpxs
    hpxs = np.any(hpx_arr, axis=0)

    ### ZMIN, ZMAX
    # Get zmins, zmaxs
    zmin_arr = np.array([asm["ZMIN"] for asm in active_skymaps])
    zmax_arr = np.array([asm["ZMAX"] for asm in active_skymaps])

    # Set inactive hpx zs to nan to avoid affecting min, max
    zmin_arr[~hpx_arr] = np.nan
    zmax_arr[~hpx_arr] = np.nan

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


def _calc_n_flares(
    t0,
    t1,
    active_skymap,
    flare_model,
    cosmo,
    use_exact_rates,
    rng,
):

    # Calculate n_agn in active volume
    # TODO: Broaden this to allow for redshift-dependent AGN distributions and other flare models
    # Calculate prism volumes for each hpx by dividing comoving volume difference by number of pixels
    # prism volume = (4/3 pi d_comoving(zmax)^3 - 4/3 pi d_comoving(zmax)^3) / n_pix
    active_volume_prisms = (
        cosmo.comoving_volume(
            active_skymap["ZMAX"],
        )
        - cosmo.comoving_volume(
            active_skymap["ZMIN"],
        )
    ) / len(active_skymap)
    # Sum to get total active volume
    active_volume = active_volume_prisms[active_skymap["INCIFOLLOWUP"]].sum()

    # Calculate expected n_flares by multiplying by flat rates
    n_flares = (
        active_volume * (10**-4.75 * u.Mpc**-3) * flare_model.flare_rate() * (t1 - t0)
    )
    print("n_flares")
    print(n_flares)
    n_flares = n_flares.to_value(u.dimensionless_unscaled)
    print(n_flares)

    # Set/draw n_flares
    if use_exact_rates:
        n_flares = round(n_flares.to_value(u.dimensionless_unscaled))
        pass
    else:
        n_flares = rng.poisson(n_flares)
    print(n_flares)

    return n_flares


def _sample_flare_coords(
    n_flares,
    t0,
    t1,
    active_skymap,
    agn_distribution,
    rng,
):
    # Sample flare times
    time_flare_backgrounds = rng.uniform(t0.mjd, t1.mjd, n_flares) * u.day

    # Sample hpx
    hpx_flare_backgrounds = rng.choice(
        np.arange(len(active_skymap)),
        n_flares,
        p=active_skymap["INCIFOLLOWUP"] / active_skymap["INCIFOLLOWUP"].sum(),
        replace=True,
    )

    # Choose random ra, dec in hpxs
    dx, dy = rng.uniform(0, 1, (2, n_flares))
    ra_flare_backgrounds, dec_flare_backgrounds = (
        ah.healpix_to_lonlat(
            hpx_flare_backgrounds,
            ah.npix_to_nside(len(active_skymap)),
            order="nested",
            dx=dx,
            dy=dy,
        )
        # hp.pix2ang(
        #     ah.npix_to_nside(len(active_skymap)),
        #     hpx_flare_backgrounds,
        #     lonlat=True,
        # )
        # * u.deg
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
                101,
            ),
            rng_np=rng,
        )[0]

        # Append to list
        z_flare_backgrounds.append(z)

    # Compile flare coordinates
    coord_flares = np.array(
        [
            time_flare_backgrounds,
            ra_flare_backgrounds.to(u.deg).value,
            dec_flare_backgrounds.to(u.deg).value,
            z_flare_backgrounds,
        ]
    ).T

    return coord_flares


def _update_flares(
    t0,
    t1,
    coord_flares,
    coord_flare_gws,
    gw_skymaps_flat,
    active_mask,
    agn_flares,
    assoc_matrix,
):
    # Copy agn_flares, assoc_matrix
    agn_flares_temp = copy(agn_flares)
    assoc_matrix_temp = copy(assoc_matrix)

    # Add gw flares in time interval
    for cfgw in coord_flare_gws:
        if t0 <= cfgw[0] < t1:
            cfgw_temp = copy(cfgw)
            cfgw_temp[1] = cfgw_temp[1].to(u.deg).value
            cfgw_temp[2] = cfgw_temp[2].to(u.deg).value
            coord_flares = np.vstack([coord_flares, cfgw_temp])

    # Iterate over flares
    for cf in coord_flares:
        # Initialize row
        assoc_row = np.zeros(len(gw_skymaps_flat), dtype=bool)

        # Get flare hpx
        hpx = ah.lonlat_to_healpix(
            cf[1] * u.deg,
            cf[2] * u.deg,
            nside=ah.npix_to_nside(len(gw_skymaps_flat[0].skymap)),
            order="nested",
        )

        # Iterate over active GWs
        for tia in np.arange(len(gw_skymaps_flat)):
            if active_mask[tia]:
                if gw_skymaps_flat[tia].skymap["INCIFOLLOWUP"][hpx]:
                    assoc_row[tia] = True

        # Append to agn_flares, assoc_matrix
        agn_flares_temp.append(Flare(*cf))
        assoc_matrix_temp.append(assoc_row)

    return agn_flares_temp, assoc_matrix_temp


def mock_data(
    skymap_dir,
    lambda_,
    agn_distribution,
    flare_model,
    n_gw=50,
    ci_followup=0.9,
    Dt_followup=200 * u.day,
    background_skymap_level=5,
    background_z_grid=np.linspace(0, 1, 101),
    background_z_frac=0.9,
    cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
    rng=np.random.default_rng(12345),
    use_exact_rates=False,
    independent_gws=False,
):
    """Generates a mock observation dataset for this analysis."""
    ##############################
    ###        GW events       ###
    ##############################

    # Get skymap filenames
    skymap_filenames = glob.glob(pa.join(skymap_dir, "*IMRPhenomXPHM.fits"))

    # Choose n_gw skymaps
    skymap_filenames = rng.choice(skymap_filenames, n_gw, replace=False)

    # Initialize skymaps
    gw_skymaps = [Skymap(f, moc=True) for f in skymap_filenames]

    ##############################
    ###        AGN flares      ###
    ##############################

    ### GW counterpart flares

    # Draw n_flares_gw from Poisson distribution
    n_flares_gw_expected = lambda_ * n_gw
    if use_exact_rates:
        n_flares_gw = round(n_flares_gw_expected)
    else:
        n_flares_gw = rng.poisson(n_flares_gw_expected)
        n_flares_gw = max(n_flares_gw, n_gw)

    # Select gw events for flares
    i_gws = np.arange(n_gw)
    i_flare_gws = rng.choice(i_gws, n_flares_gw, replace=False)
    skymap_flare_gws = [gw_skymaps[i] for i in i_flare_gws]

    # Select flare locations
    position_flare_gws = [
        sm.draw_random_location(np.linspace(0, 1, 101)[:1], ci_followup)
        for sm in skymap_flare_gws
    ]

    # Set times using distribution after GW event
    if use_exact_rates:
        times_flare_gws = [Dt_followup / 2] * n_flares_gw
    else:
        times_flare_gws = (
            rng.uniform(0, Dt_followup.to(u.day).value, n_flares_gw) * u.day
        )
    times_flare_gws = [
        Time(sm.skymap.meta["gps_time"], format="gps") + t
        for sm, t in zip(skymap_flare_gws, times_flare_gws)
    ]

    # Combine coordinates
    coord_flare_gws = [[t, *x] for t, x in zip(times_flare_gws, position_flare_gws)]

    ### Background flares

    ## Rasterize GW skymaps for easier volume calculations
    gw_skymaps_flat = []
    for sm in gw_skymaps:
        # Flatten skymap
        sm_flat = sm.flatten(background_skymap_level)

        ## 2D CI masking
        # Add mask for hpxs in ci_followup
        hpxs = sm_flat.get_hpxs_for_ci_areas(ci_followup)[0]
        sm_flat.skymap["INCIFOLLOWUP"] = False
        sm_flat.skymap["INCIFOLLOWUP"][hpxs] = True

        ## Add columns for minimum/maximum redshift
        # Iterate over healpixs
        zmins = []
        zmaxs = []
        for i in np.arange(len(sm_flat.skymap)):
            # Set to background max/min if not in followup
            if not sm_flat.skymap[i]["INCIFOLLOWUP"]:
                zmins.append(background_z_grid.min())
                zmaxs.append(background_z_grid.max())
                continue

            # Get dp_dz
            dp_dz = sm_flat.dp_dz(background_z_grid, i)

            # Ensure probability sums to at least background_z_frac
            if np.trapz(dp_dz, background_z_grid) < background_z_frac:
                print(
                    "WARNING: dp_dz does not sum to at least background_z_frac; setting zmin and zmax to min and max of z_grid"
                )
                zmins.append(background_z_grid.min())
                zmaxs.append(background_z_grid.max())
                continue
                # raise ValueError(
                #     "dp_dz does not sum to at least background_z_frac; perhaps expand background_z_grid?"
                # )
            if np.trapz(dp_dz, background_z_grid) > 10:
                print(
                    "WARNING: dp_dz sums to > 1000%; setting zmin and zmax to min and max of z_grid"
                )
                zmins.append(background_z_grid.min())
                zmaxs.append(background_z_grid.max())
                continue

            # Get background_z_frac bounds
            argsort_dp_dz = np.argsort(dp_dz)[::-1]
            prob_sorted = dp_dz[argsort_dp_dz] * (
                background_z_grid[1] - background_z_grid[0]
            )
            prob_cumsum = np.cumsum(prob_sorted)
            ind_ci_cutoff = np.searchsorted(prob_cumsum, background_z_frac)
            z_grid_ci = background_z_grid[argsort_dp_dz[:ind_ci_cutoff]]
            zmin = z_grid_ci.min()
            zmax = z_grid_ci.max()

            # Append to lists
            zmins.append(zmin)
            zmaxs.append(zmax)

        # Add to skymap
        sm_flat.skymap["ZMIN"] = zmins
        sm_flat.skymap["ZMAX"] = zmaxs

        ## Append to list
        gw_skymaps_flat.append(sm_flat)

    # Get GW event times, order by time
    time_gws = [
        Time(sm.skymap.meta["gps_time"], format="gps") for sm in gw_skymaps_flat
    ]
    ind_gw_time = np.argsort(time_gws)

    # Iterate over GW events in time order
    active_mask = [False] * n_gw
    agn_flares = []
    assoc_matrix = []
    for ti, ti_next in zip(ind_gw_time, np.roll(ind_gw_time, -1)):
        print(f"Processing GW event #{ti}/{n_gw}")

        ##############################
        ###  Add activated skymap  ###
        ##############################

        # Mark as active
        print(f"Setting GW event {ti} as active (Time {time_gws[ti]})")
        active_mask[ti] = True

        ##############################
        ###Process terminating GWs ###
        ##############################

        # Get time for next GW event; if ti_next has cycled to the end, set to far future
        if ti_next == ind_gw_time[0]:
            time_next = Time("9999-12-31T23:59:59", format="isot", scale="utc")
        else:
            time_next = time_gws[ti_next]
        print(f"Next GW event is at {time_next}")

        # Check active skymaps for termination (in time order)
        t0 = time_gws[ti]
        for tti in ind_gw_time:
            # Skip those that are not active
            if not active_mask[tti]:
                print(f"GW event {tti} is not active; skipping")
                continue

            # Set termination time
            t1 = time_gws[tti] + Dt_followup

            # Skip those that have already terminated
            if t1 < t0:
                print(f"GW event {tti} has already terminated; skipping")
                continue

            # Skip those that do not terminate before the next GW event
            if t1 >= time_next:
                print(f"GW event {tti} does not terminate before next event; skipping")
                continue

            print(f"GW event {tti} terminates (Time {t1})")

            ##############################
            ###  Get number of flares  ###
            ##############################

            # Recalculate active skymap from active skymaps
            active_skymap = _calc_active_skymap(
                [gsf.skymap for (am, gsf) in zip(active_mask, gw_skymaps_flat) if am]
            )

            n_flares = _calc_n_flares(
                t0,
                t1,
                active_skymap,
                flare_model,
                cosmo,
                use_exact_rates,
                rng,
            )

            # Draw coords if n_flares > 0
            if n_flares > 0:

                ##############################
                ###Sample flare coordinates###
                ##############################

                coord_flares = _sample_flare_coords(
                    n_flares,
                    t0,
                    t1,
                    active_skymap,
                    agn_distribution,
                    rng,
                )

                ##############################
                ###  Update assoc_matrix   ###
                ##############################

                agn_flares, assoc_matrix = _update_flares(
                    t0,
                    t1,
                    coord_flares,
                    coord_flare_gws,
                    gw_skymaps_flat,
                    active_mask,
                    agn_flares,
                    assoc_matrix,
                )

            ##############################
            ###      Cleanup loop      ###
            ##############################

            # Update time_start
            t0 = t1

            # Mark terminated flare as inactive
            print(f"Setting GW event {tti} as inactive (Time {time_gws[tti]})")
            active_mask[tti] = False

        ### Cover time block before next GW event begins
        t1 = time_next

        # If any GWs are active, get flares
        if np.any(active_mask):

            ##############################
            ###  Get number of flares  ###
            ##############################

            # Recalculate active skymap from active skymaps
            active_skymap = _calc_active_skymap(
                [gsf.skymap for (am, gsf) in zip(active_mask, gw_skymaps_flat) if am]
            )

            n_flares = _calc_n_flares(
                t0,
                t1,
                active_skymap,
                flare_model,
                cosmo,
                use_exact_rates,
                rng,
            )

            # Draw coords if n_flares > 0
            if n_flares > 0:

                ##############################
                ###Sample flare coordinates###
                ##############################

                coord_flares = _sample_flare_coords(
                    n_flares,
                    t0,
                    t1,
                    active_skymap,
                    agn_distribution,
                    rng,
                )

                ##############################
                ###  Update assoc_matrix   ###
                ##############################

                agn_flares, assoc_matrix = _update_flares(
                    t0,
                    t1,
                    coord_flares,
                    coord_flare_gws,
                    gw_skymaps_flat,
                    active_mask,
                    agn_flares,
                    assoc_matrix,
                )

    ##############################
    ###        Return          ###
    ##############################

    # Transpose assoc_matrix to comply with n_gw x n_flares convention
    assoc_matrix = np.transpose(assoc_matrix)

    return gw_skymaps, agn_flares, assoc_matrix


class Framework:
    """Base class for Palmese21-like analyses."""

    def __init__(
        self,
        gw_skymaps,
        agn_flares,
        assoc_matrix,
        agn_distribution,
        flare_model,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        z_grid=np.linspace(0, 1, 1001),
        ci_prob=0.9,
        Dt_followup=200 * u.day,
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
        self.Dt_followup = Dt_followup

        # Set gw/flare indices
        self.ind_gw, self.ind_flare = np.where(self.assoc_matrix)

        # Set flare hpxs (same dimensions as assoc_matrix)
        self.hpx_flares = np.array(
            [
                [
                    sm.skycoord2pix(SkyCoord(f.ra, f.dec, unit=u.deg), moc_level_max=12)
                    for f in self.agn_flares
                ]
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
        skymaps = [self.gw_skymaps[i] for i in self.ind_gw]
        flares = [self.agn_flares[i] for i in self.ind_flare]

        # Get hpx indices of flare locations
        hpx_flares = self.hpx_flares[self.ind_gw, self.ind_flare]

        # Calculate dp_dOmega
        dp_dOmega = (
            u.Quantity([sm.dp_dOmega(i) for sm, i in zip(skymaps, hpx_flares)])
            / self.ci_prob
        )

        # Calculate dp_dz
        dp_dz = u.Quantity(
            [
                sm.dp_dz(f.z, i, cosmo=cosmo)
                for sm, f, i in zip(skymaps, flares, hpx_flares)
            ]
        )

        # Combine probability densities
        s_values = dp_dOmega * dp_dz

        # Set values in s_arr
        s_arr[self.assoc_matrix] = s_values.flatten()

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
        flares = [self.agn_flares[i] for i in self.ind_flare]
        z_flares = [f.z for f in flares]

        # Calculate values
        b_values = (
            self.agn_distribution.dp_dOmega_dz(
                z_grid=self.z_grid,
                z_evaluate=z_flares,
                cosmo=cosmo,
                brightness_limits=brightness_limits,
            )
            * self.Dt_followup
            * self.flare_model.flare_rate(flares)
        )

        # Set values in b_arr
        b_arr[self.assoc_matrix] = b_values

        return b_arr  # Shape: (n_skymaps, n_flares)

    def calc_mu_arr(self, lambda_, cosmo=None):
        return lambda_ * np.ones(len(self.gw_skymaps))


class Lambda:
    def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        pass

    def set_arrs(self, inference, lambda_, cosmo):
        # Calculate signal terms
        self.s_arr = inference.calc_s_arr(cosmo=cosmo)

        # Calculate background terms
        self.b_arr = inference.calc_b_arr(cosmo=cosmo)

        # Calculate mu terms
        self.mu_arr = inference.calc_mu_arr(lambda_, cosmo=cosmo)

    def ln_prior(self, lambda_):
        # If in ok domain, return 0; otheriwse return -inf
        if 0.0 <= lambda_ <= 1.0:
            return 0.0
        return -np.inf

    def ln_likelihood(self, lambda_):
        # Calculate likelihood
        ln_likelihood = (
            np.sum(np.log(lambda_ * self.s_arr + self.b_arr))
            # - self.calc_mu_arr(lambda_).sum()
            - lambda_ * self.s_arr.shape[0]
        )

        return ln_likelihood

    def ln_posterior(self, lambda_):
        # Evaluate prior
        ln_prior = self.ln_prior(lambda_)

        # Return -inf if prior is inf
        if not np.isfinite(ln_prior):
            return -np.inf

        # Evaluate likelihood, after updating mu_arr
        ln_likelihood = self.ln_likelihood(lambda_)

        # Combine into posterior
        ln_posterior = ln_prior + ln_likelihood

        return ln_posterior

    def run_mcmc(
        self,
        inference=None,
        lambda_0=0.5,
        cosmo=None,
        n_walkers=32,
        n_steps=5000,
        n_proc=32,
    ):
        # # Get cosmo
        # cosmo = self.get_cosmo(cosmo)

        # Set arrays
        self.set_arrs(inference, lambda_0, cosmo)

        # Initial lambdas; clip to 0, 1
        initial_state = lambda_0 + 1e-4 * np.random.randn(n_walkers, 1)
        initial_state = np.clip(initial_state, 0.0, 1.0)

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
            "nsteps": n_steps,
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


# class Lambda(Palmese21):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def set_arrs(self, lambda_, cosmo=None):
#         # Get cosmo
#         cosmo = self.get_cosmo(cosmo)

#         # Calculate signal terms
#         self.s_arr = self.calc_s_arr(cosmo=cosmo)

#         # Calculate background terms
#         self.b_arr = self.calc_b_arr(cosmo=cosmo)

#         # Calculate mu terms
#         self.mu_arr = self.calc_mu_arr(lambda_, cosmo=cosmo)

#     def ln_prior(self, lambda_):
#         # If in ok domain, return 0; otheriwse return -inf
#         if 0.0 <= lambda_ <= 1.0:
#             return 0.0
#         return -np.inf

#     def ln_likelihood(self, lambda_):
#         # Calculate likelihood
#         ln_likelihood = (
#             np.sum(np.log(lambda_ * self.s_arr + self.b_arr))
#             - self.calc_mu_arr(lambda_).sum()
#         )

#         return ln_likelihood

#     def ln_posterior(self, lambda_):
#         # Evaluate prior
#         ln_prior = self.ln_prior(lambda_)

#         # Return -inf if prior is inf
#         if not np.isfinite(ln_prior):
#             return -np.inf

#         # Evaluate likelihood, after updating mu_arr
#         ln_likelihood = self.ln_likelihood(lambda_)

#         # Combine into posterior
#         ln_posterior = ln_prior + ln_likelihood

#         return ln_posterior

#     def run_mcmc(self, lambda_0=0.5, cosmo=None, n_walkers=32, n_steps=5000, n_proc=32):
#         # Get cosmo
#         cosmo = self.get_cosmo(cosmo)

#         # Set arrays
#         self.set_arrs(lambda_0, cosmo=cosmo)

#         # Initial lambdas; clip to 0, 1
#         initial_state = lambda_0 + 1e-4 * np.random.randn(n_walkers, 1)
#         initial_state = np.clip(initial_state, 0.0, 1.0)

#         # Define sampler args
#         args_sampler = {
#             "nwalkers": n_walkers,
#             "ndim": 1,
#             "log_prob_fn": self.ln_posterior,
#         }

#         # Define sampler kwargs
#         kwargs_sampler = {}
#         # Define run_mcmc args
#         args_run_mcmc = {
#             "initial_state": initial_state,
#             "nsteps": n_steps,
#         }
#         # Define run_mcmc kwargs
#         kwargs_run_mcmc = {
#             "progress": True,
#         }

#         # If n_proc=1, run without multiprocessing
#         if n_proc == 1:
#             # Initialize sampler
#             sampler = emcee.EnsembleSampler(
#                 **args_sampler,
#                 **kwargs_sampler,
#             )

#             # Run sampler
#             sampler.run_mcmc(**args_run_mcmc, **kwargs_run_mcmc)
#         # Else run with multiprocessing
#         else:
#             with mp.Pool(n_proc) as pool:
#                 # Initialize sampler
#                 sampler = emcee.EnsembleSampler(
#                     **args_sampler,
#                     **kwargs_sampler,
#                     pool=pool,
#                 )

#                 # Run sampler
#                 sampler.run_mcmc(**args_run_mcmc, **kwargs_run_mcmc)

#         return sampler


class LambdaH0(Palmese21):
    pass


class LambdaH0Om0(Palmese21):
    pass
