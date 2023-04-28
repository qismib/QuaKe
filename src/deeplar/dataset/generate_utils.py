""" This module contains the utility functions for the data generation process. """
import logging
from typing import Tuple
from pathlib import Path
import math
import numpy as np
import uproot
import awkward as ak
from scipy import sparse
from scipy.stats import poisson
import matplotlib as mpl
import matplotlib.pyplot as plt
from deeplar import PACKAGE

# DUNE-like LArTPC properties
AVG_IONIZATION_ENERGY = 23.6 * 1e-6  # MeV
RECOMBINATION_FACTOR = 0.75
TRANSVERSE_DIFFUSION_COEFFICIENT = 12.346370738748105  # cm^2 / s
LONGITUDINAL_DIFFUSION_COEFFICIENT = 6.601275345391748  # cm^2 / s
DRIFT_VELOCITY = 158254.38543213354  # cm / s
MAX_DRIFT_LENGTH = 3500  # mm
E_LIFETIME = 30e-3 # s (a conservative hypothesis on electron lifetime in decontaminated LAr)

# 136-Xe double beta decay Q value
Q_VALUE = 2.45783  # MeV

logger = logging.getLogger(PACKAGE + ".datagen")


class Geometry:
    """Utility class describing detector geometry."""

    def __init__(self, detector: dict):
        """
        Parameters
        ----------
        detector: dict
            The detector geometry settings.
        """
        # geometry imputs
        self.xmin, self.xmax = self.xlim = detector["xlim"]
        self.ymin, self.ymax = self.ylim = detector["ylim"]
        self.zmin, self.zmax = self.zlim = detector["zlim"]
        (
            self.xbin_w,
            self.ybin_w,
            self.zbin_w,
        ) = self.bin_w = detector["resolution"]

        # number of bins
        self.nb_xbins = math.ceil((self.xmax - self.xmin) / self.xbin_w)
        self.nb_ybins = math.ceil((self.ymax - self.ymin) / self.ybin_w)
        self.nb_zbins = math.ceil((self.zmax - self.zmin) / self.zbin_w)

        # if detector["should_crop_planes"]:
        # reduced number of bins
        xmin_reduced, xmax_reduced = detector["xlim_reduced"]
        ymin_reduced, ymax_reduced = detector["ylim_reduced"]
        zmin_reduced, zmax_reduced = detector["zlim_reduced"]
        self.nb_xbins_reduced = math.ceil((xmax_reduced - xmin_reduced) / self.xbin_w)
        self.nb_ybins_reduced = math.ceil((ymax_reduced - ymin_reduced) / self.ybin_w)
        self.nb_zbins_reduced = math.ceil((zmax_reduced - zmin_reduced) / self.zbin_w)

        # bin edeges
        self.xbins = np.linspace(self.xmin, self.xmax, self.nb_xbins + 1)
        self.ybins = np.linspace(self.ymin, self.ymax, self.nb_ybins + 1)
        self.zbins = np.linspace(self.zmin, self.zmax, self.nb_zbins + 1)

        self.min_energy = detector["min_energy"]

        # TODO [enhancement]: think about using @property as setter and getter editable
        # geometry attributes


def get_image(
    fname: Path,
    x: np.ndarray,
    y: np.ndarray,
    energy: np.ndarray,
    xbin_w: float,
    ybin_w: float,
):
    """Plots track in the [-20,20]x[-20,20] mm box, with the histogram cell
    grids.

    Parameters
    ----------
    fname: Path
        The name dataset file name.
    x: np.ndarray
        x coordinate array of shape=(nb_hits,).
    y: np.ndarray
        y coordinate array of shape=(nb_hits,).
    xbin_w: float
        Bin width in the x coordinate.
    ybin_w: float
        Bin width in the y coordinate.
    """
    n_xbins = int(np.ceil(40 / xbin_w)) + 1
    n_ybins = int(np.ceil(40 / ybin_w)) + 1
    xticks = np.linspace(-20, 20, n_xbins)
    yticks = np.linspace(-20, 20, n_ybins)
    mpl.rcParams.update({"font.size": 12})
    ax = plt.subplot()
    plt.suptitle(f"File name: {fname}")
    ax.title.set_text(
        f"Tracks and histogram cells. Resolution: {xbin_w} x {ybin_w}",
    )
    ax.set_xlabel("First coordinate")
    ax.set_ylabel("Second coordinate")
    ax.scatter(x, y, c=energy, s=3)
    plt.xlim([-20, 20])
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xticks))
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(yticks))
    plt.ylim([-20, 20])
    plt.grid()
    plt.show()


def load_tracks(
    name: Path, geo: Geometry, is_signal: bool = False, seed: int = 42
) -> Tuple[ak.Array, ak.Array, ak.Array, ak.Array]:
    """Loads events from file.

    Returns tracks starting from origin plus a random
    shift sampled uniformly in the box [-xw,xw]x[-yw,yw]x[-zw,zw]. Where `<axis>w`
    is the detector resolution in mm on that specific axis.
    If `is_signal` is True: subsequent row couples refer to two b tracks and
    they are merged together. Jagged arrays are treated with awkward module,
    regular ones with numpy instead.

    Features are:

        - TrackPostX: float, x Geant4 step position
        - TrackPostY: float, y Geant4 step position
        - TrackPostZ: float, z Geant4 step position
        - TrackEnergy: float, Geant4 step energy
        - NTrack: int, number of Geant4 step in track
        - DepositedEnergy: float, energy integrated over track

    Parameters
    ----------
    name: Path
        The name of the file to read the tracks features.
    geo: Geometry
        Detector geometry object, to get the axis resolution.
    is_signal: bool
        Wether to concatenate subsequent rows for signal tracks.
    seed: int
        Random generator seed for code reproducibility.

    Returns
    -------
    xs: ak.Array
        x Geant4 step position of shape=(tracks, [steps]).
    ys: ak.Array
        y Geant4 step position of shape=(tracks, [steps]).
    zs: ak.Array
        z Geant4 step position of shape=(tracks, [steps]).
    Es: ak.Array
        Geant4 step energy of shape=(tracks, [steps]).
    """
    rng = np.random.default_rng(seed=seed)
    with uproot.open(name) as sig_root:
        qtree = sig_root["qtree"]

        xs = qtree["TrackPostX"].array()
        ys = qtree["TrackPostY"].array()
        zs = qtree["TrackPostZ"].array()
        Es = qtree["TrackEnergy"].array()
        tid = qtree["TrackID"].array()

    if is_signal:
        # concatenate the two b tracks (from two consecutive rows)
        cat_fn = lambda arr: ak.concatenate([arr[::2], arr[1::2]], axis=1)
        xs = cat_fn(xs)
        ys = cat_fn(ys)
        zs = cat_fn(zs)
        Es = cat_fn(Es)
        tid = cat_fn(tid)

    Xs = ak.sum(xs[tid == 1] * Es[tid == 1], axis=1) / ak.sum(Es[tid == 1], axis=1)
    Ys = ak.sum(ys[tid == 1] * Es[tid == 1], axis=1) / ak.sum(Es[tid == 1], axis=1)
    Zs = ak.sum(zs[tid == 1] * Es[tid == 1], axis=1) / ak.sum(Es[tid == 1], axis=1)

    normalize = (
        lambda arr, shift, bw: arr - shift + rng.uniform(low=-bw, high=bw, size=1000)
    )

    xs = normalize(xs, Xs, geo.xbin_w)
    ys = normalize(ys, Ys, geo.ybin_w)
    zs = normalize(zs, Zs, geo.zbin_w)

    xs, ys, zs, Es = diffuse_electrons(xs, ys, zs, Es)

    return xs, ys, zs, Es


def diffuse_electrons(
    xs: ak.Array, ys: ak.Array, zs: ak.Array, Es:ak.Array,
) -> Tuple[ak.Array, ak.Array, ak.Array]:
    """Simulates electron diffusion in an electric field
    using data from https://lar.bnl.gov/properties/trans.html and electron lifetime in purified LAr.

    Parameters
    ----------
    xs: ak.Array
        x coordinates pre-diffusion.
    ys: ak.Array
        y coordinates pre-diffusion.
    zs: ak.Array
        z coordinates pre-diffusion.
    Es: ak.Array
        energy depositions pre-diffusion.

    Returns
    -------
    xs: ak.Array
        x coordinates post-diffusion.
    ys: ak.Array
        y coordinates post-diffusion.
    zs: ak.Array
        z coordinates post-diffusion.
    Es: ak.Array
        energy depositions post-diffusion.
    """
    nt = len(xs)
    readout_distance = np.random.uniform(0, MAX_DRIFT_LENGTH, nt)

    drift_times = 0.1*readout_distance/DRIFT_VELOCITY
    survival_rates = np.exp(-drift_times/E_LIFETIME)
    survival_hitmiss = np.random.uniform(0, 1, nt)
    survival_mask = survival_hitmiss < survival_rates

    xs = xs[survival_mask]
    ys = ys[survival_mask]
    zs = zs[survival_mask]
    Es = Es[survival_mask]
    drift_times = drift_times[survival_mask]
    readout_distance = readout_distance[survival_mask]

    nt = len(xs)

    rdn_orientation = np.random.uniform(0, 2 * np.pi, nt)

    sigma_l = np.sqrt(2 * drift_times * LONGITUDINAL_DIFFUSION_COEFFICIENT)*10
    sigma_t = np.sqrt(2 * drift_times * TRANSVERSE_DIFFUSION_COEFFICIENT)*10

    z_shift = np.random.normal(0, sigma_l, nt)
    transverse_shift = np.random.normal(0, sigma_t, nt)
    x_shift = transverse_shift * np.cos(rdn_orientation)
    y_shift = transverse_shift * np.sin(rdn_orientation)

    xs = xs + x_shift
    ys = ys + y_shift
    zs = zs + z_shift

    return xs, ys, zs, Es


def tracks2histograms(
    xs: ak.Array,
    ys: ak.Array,
    zs: ak.Array,
    Es: ak.Array,
    geo: Geometry,
    seed: int = 42,
) -> ak.Array:
    """Compute energy histogram from track hit positions.

    This function converts the simulated hit energy depositions to pixel images.
    The Geometry object, passed as a parameter, controls the binning resolution
    and other histogramming settings.

    Parameters
    ----------
    xs: ak.Array
        x hit position of shape=(tracks, [hits]).
    ys: ak.Array
        y hit position of shape=(tracks, [hits]).
    zs: ak.Array
        z hit position of shape=(tracks, [hits]).
    Es: ak.Array
        Hit energy of shape=(tracks, [hits]).
    geo: Geometry
        Detector geometry.
    seed: int
        Random generator seed for code reproducibility.

    Returns
    -------
    hists: ak.Array
        Sparse energy histogram.
    """
    logger.debug("Converting to histogram ...")
    hists = []

    # filter out of bin range hits
    mx = np.logical_and(xs >= geo.xbins[0], xs < geo.xbins[-1])
    my = np.logical_and(ys >= geo.ybins[0], ys < geo.ybins[-1])
    mz = np.logical_and(zs >= geo.zbins[0], zs < geo.zbins[-1])
    m = np.logical_and(np.logical_and(mx, my), mz)

    for x, y, z, energy in zip(xs[m], ys[m], zs[m], Es[m]):
        # digits start from 1
        get_digit = lambda p1, p2: np.digitize(p1, p2) - 1
        x_digits = get_digit(x.to_numpy(), geo.xbins)
        y_digits = get_digit(y.to_numpy(), geo.ybins)
        z_digits = get_digit(z.to_numpy(), geo.zbins)

        yz_digits = y_digits * geo.nb_zbins + z_digits
        shape = (geo.nb_xbins, geo.nb_ybins * geo.nb_zbins)

        # the csr_matrix automatically sums all the entries that fall in the
        # same pixel
        hist = sparse.csr_matrix(
            (energy.to_numpy(), (x_digits, yz_digits)), shape=shape
        )

        # applying charge pairs fluctuations and thresholding
        rows, cols = hist.nonzero()
        values = np.array(hist[rows, cols])[0]

        values = np.array(
            AVG_IONIZATION_ENERGY
            / RECOMBINATION_FACTOR
            * poisson.rvs(
                mu=values / AVG_IONIZATION_ENERGY * RECOMBINATION_FACTOR,
                random_state=seed,
            )
        )

        underflow = values < geo.min_energy

        # removing empty histograms
        if not np.count_nonzero(~underflow):
            continue

        # threshold histograms
        if np.count_nonzero(underflow):
            values[underflow] = 0
            hist = sparse.csr_matrix((values, (rows, cols)), shape=shape)

        # Normalizinge energy
        hist = hist / hist.sum() * Q_VALUE
        hists.append(hist.reshape(1, -1))

    hists = sparse.vstack(hists)
    return hists
