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
    # BUG: if there is a near and far GW event, then the follow-up volume will include the space between them as well
    zmin = np.nanmin(zmin_arr, axis=0)
    zmax = np.nanmax(zmax_arr, axis=0)

    # Set inactive hpx zs to 0.5  (this causes the volume to be 0 for these pixels)
    zmin[~hpxs] = 0.5
    zmax[~hpxs] = 0.5

    ### Wrap-up

    # Compile results
    active_skymap = QTable([hpxs, zmin, zmax], names=["INCIFOLLOWUP", "ZMIN", "ZMAX"])

    # ### DEBUG: check volumes
    # cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    # outstr = "\n"
    # ## volumes for active skymaps
    # for asm in active_skymaps:
    #     # ligo.skymap crossmatch volume:
    #     asm["UNIQ"] = ah.level_ipix_to_uniq(
    #         ah.nside_to_level(ah.npix_to_nside(len(asm))),
    #         np.arange(len(asm)),
    #     )
    #     cmresult = crossmatch(asm, contours=[0.9]).contour_vols[0] * u.Mpc**3
    #     outstr += f"\nligo.skymap volume={cmresult:.2f}, "

    #     # Calculate prism volumes for each hpx by dividing comoving volume difference by number of pixels
    #     # prism volume = (4/3 pi d_comoving(zmax)^3 - 4/3 pi d_comoving(zmax)^3) / n_pix
    #     active_volume_prisms = (
    #         cosmo.comoving_volume(
    #             asm["ZMAX"],
    #         )
    #         - cosmo.comoving_volume(
    #             asm["ZMIN"],
    #         )
    #     ) / len(asm)
    #     # Sum to get total active volume
    #     active_volume = active_volume_prisms[asm["INCIFOLLOWUP"]].sum()
    #     outstr += f"prism volume={active_volume.to(u.Mpc**3):.2f}"
    # ## active_skymap volume
    # active_volume_prisms = (
    #     cosmo.comoving_volume(
    #         active_skymap["ZMAX"],
    #     )
    #     - cosmo.comoving_volume(
    #         active_skymap["ZMIN"],
    #     )
    # ) / len(active_skymap)
    # # Sum to get total active volume
    # active_volume = active_volume_prisms[active_skymap["INCIFOLLOWUP"]].sum()
    # outstr += f"\ntotal prism volume={active_volume.to(u.Mpc**3):.2f}"
    # outstr += "\n"
    # print(outstr)

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
    n_flares = n_flares.to_value(u.dimensionless_unscaled)

    # Set/draw n_flares
    if use_exact_rates:
        n_flares = round(n_flares)
        pass
    else:
        n_flares = rng.poisson(n_flares)

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
    time_flare_backgrounds = rng.uniform(t0.mjd, t1.mjd, n_flares)
    time_flare_backgrounds = [Time(t, format="mjd") for t in time_flare_backgrounds]

    # Sample hpx
    hpx_flare_backgrounds = rng.choice(
        np.arange(len(active_skymap)),
        n_flares,
        p=active_skymap["INCIFOLLOWUP"] / active_skymap["INCIFOLLOWUP"].sum(),
        replace=True,
    )

    # Choose random ra, dec in hpxs
    dx, dy = rng.uniform(0, 1, (2, n_flares))
    ra_flare_backgrounds, dec_flare_backgrounds = ah.healpix_to_lonlat(
        hpx_flare_backgrounds,
        ah.npix_to_nside(len(active_skymap)),
        order="nested",
        dx=dx,
        dy=dy,
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
    coord_flares = [
        [t, ra, dec, z]
        for t, ra, dec, z in zip(
            time_flare_backgrounds,
            ra_flare_backgrounds,
            dec_flare_backgrounds,
            z_flare_backgrounds,
        )
    ]

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
            coord_flares.append(cfgw)

    # Iterate over flares
    for cf in coord_flares:
        # Initialize row
        assoc_row = np.zeros(len(gw_skymaps_flat), dtype=bool)

        # Get flare hpx
        hpx = ah.lonlat_to_healpix(
            cf[1],
            cf[2],
            nside=ah.npix_to_nside(len(gw_skymaps_flat[0].skymap)),
            order="nested",
        )

        # Iterate over active GWs
        for tia in np.arange(len(gw_skymaps_flat)):
            if active_mask[tia]:
                # If in follow-up volume
                if gw_skymaps_flat[tia].skymap[hpx]["INCIFOLLOWUP"] and (
                    gw_skymaps_flat[tia].skymap[hpx]["ZMIN"]
                    <= cf[3]
                    <= gw_skymaps_flat[tia].skymap[hpx]["ZMAX"]
                ):
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
    background_z_grid=np.linspace(0, 2, 201),
    background_z_frac=0.9,
    cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
    rng=np.random.default_rng(12345),
    use_exact_rates=False,
    independent_gws=False,
    verbose=1,
):
    """Generates a mock observation dataset for this analysis.
    This version includes the first attempt to calculate a total follow-up volume for all active skymaps, accounting for overlaps.
    """

    ##################################################
    ##################################################
    ####                                          ####
    ####                GW events                 ####
    ####                                          ####
    ##################################################
    ##################################################

    print("*" * 60)
    print("Selecting GW events for mock follow-up...")

    # Get skymap filenames
    skymap_filenames = glob.glob(pa.join(skymap_dir, "*IMRPhenomXPHM.fits"))

    # Choose n_gw skymaps
    skymap_filenames = rng.choice(skymap_filenames, n_gw, replace=False)

    # Initialize skymaps
    gw_skymaps = [Skymap(f, moc=True) for f in skymap_filenames]

    print(f"Selected {len(gw_skymaps)} events")

    ##################################################
    ##################################################
    ####                                          ####
    ####                AGN flares                ####
    ####                                          ####
    ##################################################
    ##################################################

    print("*" * 60)
    print("Generating AGN flares...")

    ########################################
    ########################################
    ####     GW counterpart flares      ####
    ########################################
    ########################################

    print("*" * 50)
    print("Generating GW counterpart flares...")

    # Draw n_flares_gw from Poisson distribution
    n_flares_gw_expected = lambda_ * n_gw
    if use_exact_rates:
        n_flares_gw = round(n_flares_gw_expected)
    else:
        n_flares_gw = rng.poisson(n_flares_gw_expected)
        n_flares_gw = min(n_flares_gw, n_gw)

    # Select gw events for flares
    i_gws = np.arange(n_gw)
    i_flare_gws = rng.choice(i_gws, n_flares_gw, replace=False)
    skymap_flare_gws = [gw_skymaps[i] for i in i_flare_gws]

    # Select flare locations
    position_flare_gws = [
        sm.draw_random_location(np.linspace(0, 1, 101), ci_followup)
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

    # Summarize step
    print(f"Generated {n_flares_gw} GW counterpart flares")
    print(f"(lambda={lambda_} * n_gw={n_gw} ~ {lambda_ * n_gw})")
    if verbose > 0:
        for sfg, cfg in zip(skymap_flare_gws, coord_flare_gws):
            print(
                pa.basename(sfg.filename).split("-v2-")[1].split("_PED")[0],
                cfg[0].iso,
                cfg[1].to(u.hourangle),
                cfg[2].to(u.deg),
                f"{cfg[3]:.2f}",
            )

    ########################################
    ########################################
    ####        Background flares       ####
    ########################################
    ########################################

    print("*" * 50)
    print("Generating background flares...")

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

            # Set to background max/min for bad distmu
            if not np.isfinite(sm_flat.skymap[i]["DISTMU"]):
                zmins.append(background_z_grid.min())
                zmaxs.append(background_z_grid.max())
                continue

            # Get dp_dz
            dp_dz = sm_flat.dp_dz(background_z_grid, i)

            # Ensure probability sums to at least background_z_frac
            warn_message = None
            prob_sum = np.trapz(dp_dz, background_z_grid)
            if prob_sum < background_z_frac:
                warn_message = (
                    f"dp_dz sums to {prob_sum} < background_z_frac={background_z_frac}"
                )
                print(
                    "WARNING:",
                    warn_message,
                    f"(DISTMU={sm_flat.skymap[i]['DISTMU']},",
                    f"DISTSIGMA={sm_flat.skymap[i]['DISTSIGMA']},",
                    f"DISTNORM={sm_flat.skymap[i]['DISTNORM']});",
                    "setting zmin and zmax to min and max of z_grid",
                )
                zmins.append(background_z_grid.min())
                zmaxs.append(background_z_grid.max())
                continue
            if prob_sum > 1.01:
                warn_message = f"dp_dz sums to {prob_sum} > 101%"
                print(
                    "WARNING:",
                    warn_message,
                    f"(DISTMU={sm_flat.skymap[i]['DISTMU']},",
                    f"DISTSIGMA={sm_flat.skymap[i]['DISTSIGMA']},",
                    f"DISTNORM={sm_flat.skymap[i]['DISTNORM']});",
                    "substituting values for maximum probability hpx",
                )
                if sm_flat.moc:
                    probkey = "PROBDENSITY"
                else:
                    probkey = "PROB"
                dp_dz = sm_flat.dp_dz(
                    background_z_grid,
                    np.argmax(sm_flat.skymap[probkey]),
                )
            if warn_message is not None:
                pass
                # raise ValueError(
                #     "dp_dz does not sum to at least background_z_frac; perhaps expand background_z_grid?"
                # )

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
    coord_flare_backgrounds = []
    for ti, ti_next in zip(ind_gw_time, np.roll(ind_gw_time, -1)):

        ##############################
        ###  Add activated skymap  ###
        ##############################

        # Mark as active
        active_mask[ti] = True

        ##############################
        ###Process terminating GWs ###
        ##############################

        # Get time for next GW event; if ti_next has cycled to the end, set to far future
        if ti_next == ind_gw_time[0]:
            time_next = Time("9999-12-31T23:59:59", format="isot", scale="utc")
        else:
            time_next = time_gws[ti_next]

        # Check active skymaps for termination (in time order)
        t0 = time_gws[ti]
        for tti in ind_gw_time:
            # Skip those that are not active
            if not active_mask[tti]:
                continue

            # Set termination time
            t1 = time_gws[tti] + Dt_followup

            # Skip those that have already terminated
            if t1 < t0:
                continue

            # Skip those that do not terminate before the next GW event
            if t1 >= time_next:
                continue

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
            coord_flares = []
            if n_flares > 0:

                ##############################
                ###Sample flare coordinates###
                ##############################

                # Sample coordinates for flares
                coord_flares = _sample_flare_coords(
                    n_flares,
                    t0,
                    t1,
                    active_skymap,
                    agn_distribution,
                    rng,
                )

                # Update running list
                coord_flare_backgrounds = [*coord_flare_backgrounds, *coord_flares]

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
            coord_flares = []
            if n_flares > 0:

                ##############################
                ###Sample flare coordinates###
                ##############################

                # Sample coordinates for flares
                coord_flares = _sample_flare_coords(
                    n_flares,
                    t0,
                    t1,
                    active_skymap,
                    agn_distribution,
                    rng,
                )

                # Update running list
                coord_flare_backgrounds = [*coord_flare_backgrounds, *coord_flares]

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

    # Report background flares
    print(f"Generated {len(coord_flare_backgrounds)} background flares")
    if verbose > 0:
        for cfb in coord_flare_backgrounds:
            print(
                cfb[0].iso,
                cfb[1].to(u.hourangle),
                cfb[2].to(u.deg),
                f"{cfb[3]:.2f}",
            )

    ##################################################
    ##################################################
    ####                                          ####
    ####                  Return                  ####
    ####                                          ####
    ##################################################
    ##################################################

    # Transpose assoc_matrix to comply with n_gw x n_flares convention
    assoc_matrix = np.transpose(assoc_matrix)

    # Print summary
    lambda_actual = n_flares_gw / n_gw
    if verbose > 0:
        print("*" * 50)
        print("Summary:")
        print(f"lambda={lambda_}, lambda_actual={lambda_actual}")
        print(f"n_gw: {n_gw}")
        print(f"n_flares_gw: {n_flares_gw}")
        print(f"n_flares_background: {len(coord_flare_backgrounds)}")
        print(f"n_flares_total: {len(agn_flares)}")

    return gw_skymaps, agn_flares, assoc_matrix, lambda_actual


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

        # Calculate number density of gw event
        # Because each probability distribution corresponds to 1 event, dn_dOmega_dz = dp_dOmega_dz
        dn_dOmega_dz = dp_dOmega * dp_dz

        # Save as s_values
        # (not rate of flares like for b_arr, since lambda_ is multiplied later)
        s_values = dn_dOmega_dz.flatten()

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

        # # Initialize b_arr
        # b_arr = np.ones((len(self.gw_skymaps), len(self.agn_flares)))

        # Fetch flares, flare redshifts
        z_flares = [f.z for f in self.agn_flares]

        # Calculate number density of AGNi
        dn_dOmega_dz = self.agn_distribution.dn_dOmega_dz(
            zs=z_flares,
            cosmo=cosmo,
            brightness_limits=brightness_limits,
        )

        # Calculate number density of flares
        b_values = (
            dn_dOmega_dz
            * self.Dt_followup
            * self.flare_model.flare_rate(self.agn_flares)
        ).flatten()

        # Set values in b_arr
        b_arr = np.repeat([b_values], len(self.gw_skymaps), axis=0)
        # b_arr[self.assoc_matrix] = b_values

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

        # Set association matrix
        self.assoc_matrix = inference.assoc_matrix

    def ln_prior(self, lambda_):
        # If in ok domain, return 0; otheriwse return -inf
        if 0.0 <= lambda_ <= 1.0:
            return 0.0
        return -np.inf

    def ln_likelihood_palmese21(self, lambda_):
        # Calculate likelihood
        ln_likelihood = (
            np.sum(np.log((lambda_ * self.s_arr + self.b_arr)[self.assoc_matrix]))
            # - self.calc_mu_arr(lambda_).sum()
            - lambda_ * self.s_arr.shape[0]
        )

        return ln_likelihood

    def ln_likelihood_onegwflare(self, lambda_):
        # Calculate likelihood
        # Lowball version; assumes 0-1 gw flares per dataset, rest are background
        # (i.e. assumes at most one flare across all follow-ups is gw-sourced)
        ln_likelihood = (
            np.log(lambda_ * np.sum((self.s_arr / self.b_arr)[self.assoc_matrix]) + 1)
            + np.sum(np.log(self.b_arr[self.assoc_matrix]))
            # - self.calc_mu_arr(lambda_).sum()
            - lambda_ * self.s_arr.shape[0]
        )

        return ln_likelihood

    def ln_likelihood_multiflare(self, lambda_):
        # Calculate likelihood
        # multiflare version: still contains terms associating multiple flares to a single GW event
        ln_likelihood = (
            np.sum(
                np.log(
                    lambda_
                    * np.sum((self.s_arr / self.b_arr) * self.assoc_matrix, axis=1)
                    + 1
                )
            )
            + np.sum(np.log(self.b_arr[self.assoc_matrix]))
            # - self.calc_mu_arr(lambda_).sum()
            - lambda_ * self.s_arr.shape[0]
        )

        return ln_likelihood

    def ln_likelihood_multigw(self, lambda_):
        # Calculate likelihood
        # multigw version: still contains terms where a single flare is associated with multiple GW events
        ln_likelihood = (
            np.sum(
                np.log(
                    lambda_
                    * np.sum((self.s_arr / self.b_arr) * self.assoc_matrix, axis=0)
                    + 1
                )
            )
            + np.sum(np.log(self.b_arr[self.assoc_matrix]))
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
        ln_likelihood = self._ln_likelihood(lambda_)

        # Combine into posterior
        ln_posterior = ln_prior + ln_likelihood

        return ln_posterior

    def run_mcmc(
        self,
        inference=None,
        lambda_0=0.5,
        likelihood="palmese21",
        cosmo=None,
        n_walkers=32,
        n_steps=5000,
        n_proc=32,
        rng=np.random.default_rng(12345),
    ):
        # # Get cosmo
        # cosmo = self.get_cosmo(cosmo)

        # Set arrays
        self.set_arrs(inference, lambda_0, cosmo)
        print("s_arr:")
        print(self.s_arr)
        print("b_arr:")
        print(self.b_arr)
        print("np.sum(assoc_matrix, axis=0):", np.sum(self.assoc_matrix, axis=0))
        print("np.sum(assoc_matrix, axis=1):", np.sum(self.assoc_matrix, axis=1))

        # Set likelihood function
        self._ln_likelihood = getattr(self, f"ln_likelihood_{likelihood}")

        # Initial lambdas; clip to 0, 1
        initial_state = lambda_0 + 1e-4 * rng.standard_normal((n_walkers, 1))
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


class LambdaH0:
    pass


class LambdaH0Om0:
    pass
