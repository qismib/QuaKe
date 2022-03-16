""" This module contains the utility functions for the data generation process. """
import logging
from typing import Tuple
from pathlib import Path
import math
import numpy as np
import uproot
import awkward as ak
from scipy import sparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from quake import PACKAGE

logger = logging.getLogger(PACKAGE + ".datagen")


class Geometry:
    """Utility class describing detector geometry."""

    def __init__(self, detector: dict):
        """
        Parameters
        ----------
            - detector: the detector geometry settings
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

        # bin edeges
        self.xbins = np.linspace(self.xmin, self.xmax, self.nb_xbins + 1)
        self.ybins = np.linspace(self.ymin, self.ymax, self.nb_ybins + 1)
        self.zbins = np.linspace(self.zmin, self.zmax, self.nb_zbins + 1)

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
    """
    Plots track in the [-20,20]x[-20,20] mm box, with the histogram cell grids.
    Parameters
    ----------
        - fname: the name dataset file name
        - x: x coordinate array of shape=(nb_hits,)
        - y: y coordinate array of shape=(nb_hits,)
        - xbin_w: bin width in the x coordinate
        - ybin_w: bin width in the y coordinate
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
    name: Path, geo: Geometry, is_signal: bool = False, seed: int = None
) -> Tuple[ak.Array, ak.Array, ak.Array, ak.Array]:
    """
    Loads events from file. Returns tracks starting from origin plus a random
    shift sampled uniformly in the box [-xw,xw]x[-yw,yw]x[-zw,zw]. Where `<axis>w`
    is the detector resolution in mm on that specific axis.
    If `is_signal` is True: subsequent row couples refer to two b tracks and
    they are merged together. Jagged arrays are treated with awkward module,
    regular ones with numpy instead.
    Features are:
        - TrackPostX: float, x hit position
        - TrackPostY: float, y hit position
        - TrackPostZ: float, z hit position
        - TrackEnergy: float, hit energy
        - NTrack: int, number of hits in track
        - DepositedEnergy: float, energy integrated over track

    Parameters
    ----------
        - name: the name of the file to read the tracks features
        - geo: detector geometry object, to get the axis resolution
        - is_signal: wether to concatenate subsequent rows for signal tracks
        - seed: random generator seed for code reproducibility

    Returns
    -------
        - x hit position of sh-ape=(tracks, [hits])
        - y hit position of shape=(tracks, [hits])
        - z hit position of shape=(tracks, [hits])
        - hit energy of shape=(tracks, [hits])
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
        # Get the index of the track starting point
        idx = np.sum(tid[:, ::2] == 1, axis=1) - 1 
    else:
        idx = np.sum(tid == 1, axis=1) - 1
    idx = idx.to_numpy()
    nev = idx.shape[0]
    s_ = np.s_[np.arange(nev), idx]

    normalize = lambda arr, shift: arr + rng.uniform(low=-shift, high=shift, size=1000)

    xs = normalize(xs - xs[s_], geo.xbin_w)
    ys = normalize(ys - ys[s_], geo.ybin_w)
    zs = normalize(zs - zs[s_], geo.zbin_w)
    return xs, ys, zs, Es


def tracks2histograms(
    xs: ak.Array, ys: ak.Array, zs: ak.Array, Es: ak.Array, geo: Geometry
) -> ak.Array:
    """
    Compute energy histogram from track hit positions. This function converts
    the simulated hit energy depositions to pixel images. The Geometry object,
    passed as a parameter, controls the binning resolution and other
    histogramming settings.

    Parameters
    ----------
        - xs: x hit position of shape=(tracks, [hits])
        - ys: y hit position of shape=(tracks, [hits])
        - zs: z hit position of shape=(tracks, [hits])
        - Es: hit energy of shape=(tracks, [hits])
        - geo: detector geometry

    Returns
    -------
        - sparse energy histogram of shape=()
    """
    logger.debug("Converting to histogram ...")
    hists = []

    # filter out of bin range hits
    mx = np.logical_and(xs >= geo.xbins[0], xs < geo.xbins[-1])
    my = np.logical_and(ys >= geo.xbins[0], ys < geo.xbins[-1])
    mz = np.logical_and(zs >= geo.xbins[0], zs < geo.xbins[-1])
    m = np.logical_and(np.logical_and(mx, my), mz)
    for x, y, z, energy in zip(xs[m], ys[m], zs[m], Es[m]):
        # digits start from 1
        get_digit = lambda p1, p2: np.digitize(p1, p2) - 1
        x_digits = get_digit(x.to_numpy(), geo.xbins)
        y_digits = get_digit(y.to_numpy(), geo.ybins)
        z_digits = get_digit(z.to_numpy(), geo.zbins)

        yz_digits = y_digits * geo.nb_zbins + z_digits
        shape = (geo.nb_xbins, geo.nb_ybins * geo.nb_zbins)
        hist = sparse.csr_matrix(
            (energy.to_numpy(), (x_digits, yz_digits)), shape=shape
        )

        try:
            assert all(energy > 0)
        except:
            ValueError(f"Found non positive energies: {np.count_nonzero(energy > 0)}")

        hists.append(hist.reshape(1, -1))
    hists = sparse.vstack(hists)
    return hists
