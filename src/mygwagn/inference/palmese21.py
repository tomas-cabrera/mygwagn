from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import numpy as np


class Palmese21:

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
        self.id_hpx_flares = np.array(
            [
                [sm.skycoord_to_healpix(f.skycoord()) for f in self.agn_flares]
                for sm in self.gw_skymaps
            ]
        )

    def get_cosmo(self, cosmo):
        """Returns default cosmo if None, else returns input."""
        if cosmo is None:
            return self.cosmo
        return cosmo

    def calc_s_arr(self):

        # Set cosmo
        cosmo = self.get_cosmo(cosmo)

        # Initialize s_arr
        s_arr = np.zeros((len(self.gw_skymaps), len(self.agn_flares)))

        # Get skymaps, flares
        skymaps = self.gw_skymaps[self.ind_gw]
        flares = self.agn_flares[self.ind_flare]

        # Get hpx indices of flare locations
        id_hpx_flares = self.id_hpx_flares[self.ind_gw, self.ind_flare]

        # Calculate dp_dOmega
        dp_dOmega = (
            u.quantity([sm.dp_dOmega(i) for sm, i in zip(skymaps, id_hpx_flares)])
            / self.ci_prob
        )

        # Calculate dp_dz
        dp_dz = u.quantity(
            [
                sm.dp_dz(self.z_grid, i, f.z())
                for sm, f, i in zip(skymaps, flares, id_hpx_flares)
            ]
        )

        # Combine probability densities
        s_values = dp_dOmega * dp_dz

        # Set values in s_arr
        s_arr[self.assoc_matrix] = s_values

        return s_arr  # Shape: (n_skymaps, n_flares)

    def calc_b_arr(self, cosmo=None):

        # Set cosmo
        cosmo = self.get_cosmo(cosmo)

        # Initialize b_arr
        b_arr = np.zeros((len(self.gw_skymaps), len(self.agn_flares)))

        # Fetch flares, flare redshifts
        flares = self.agn_flares[self.ind_flare]
        z_flares = [f.z() for f in flares]

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


class Lambda(Palmese21):
    pass


class LambdaH0(Palmese21):
    pass


class LambdaH0Om0(Palmese21):
    pass
