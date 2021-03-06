"""
    This module reads events from files and stores histograms.

    Usage:
    
    ```
    python storedata <input> -o <output> [--show]
    ```

    The `--show` optional flag allows to show a plot taken from each file in the
    dataset folder.
"""
import argparse
from pathlib import Path
from time import time as tm
from tqdm import tqdm
import uproot
import math
import numpy as np
from scipy import sparse
import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt


class Geometry:
    """Utility class describing detector geometry."""

    def __init__(
        self, xlim=(-250, 250), ylim=(-250, 250), zlim=(-250, 250), bin_w=(5, 5, 1)
    ):
        """
        Parameters
        ----------
            - xlim: tuple, x axis min and max values
            - ylim: tuple, y axis min and max values
            - zlim: tuple, z axis min and max values
            - bin_w: tuple, bin width resolution for x, y, z axis
        """
        # geometry imputs
        self.xmin, self.xmax = self.xlim = xlim
        self.ymin, self.ymax = self.ylim = ylim
        self.zmin, self.zmax = self.zlim = zlim
        (
            self.xbin_w,
            self.ybin_w,
            self.zbin_w,
        ) = self.bin_w = bin_w

        # number of bins
        self.nb_xbins = math.ceil((self.xmax - self.xmin) / self.xbin_w)
        self.nb_ybins = math.ceil((self.ymax - self.ymin) / self.ybin_w)
        self.nb_zbins = math.ceil((self.zmax - self.zmin) / self.zbin_w)

        # bin edeges
        self.xbins = np.linspace(self.xmin, self.xmax, self.nb_xbins + 1)
        self.ybins = np.linspace(self.ymin, self.ymax, self.nb_ybins + 1)
        self.zbins = np.linspace(self.zmin, self.zmax, self.nb_zbins + 1)

        # TODO: think about using @property as setter and getter editable
        # geometry attributes


def get_image(fname, x, y, energy, xbin_w, ybin_w):
    """
    Plots track in the [-20,20]x[-20,20] mm box, with the histogram cell grids.

    Parameters
    ----------
        - fname: Path, the name dataset file name
        - x: np.array, x coordinate array of shape=(nb_hits,)
        - y: np.array, y coordinate array of shape=(nb_hits,)
        - xbin_w: float, bin width in the x coordinate
        - ybin_w: float, bin width in the y coordinate
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


def load_tracks(name, is_signal=False):
    """
    Loads events from file. Returns hit positions with zero mean. If `is_signal`
    is True: subsequent row couples refer to two b tracks and they are merged
    together. Jagged arrays are treated with awkward module, regular ones with
    numpy instead.

    Features are:
        - TrackPostX: float, x hit position
        - TrackPostY: float, y hit position
        - TrackPostZ: float, z hit position
        - TrackEnergy: float, hit energy
        - NTrack: int, number of hits in track
        - DepositedEnergy: float, energy integrated over track

    Parameters
    ----------
        - name: Path, the name of the file to read the tracks features
        - is_signal: bool, wether to concatenate subsequent rows for signal
                     tracks. Default: False.

    Returns
    -------
        - ak.Array, x hit position of sh-ape=(tracks, [hits])
        - ak.Array, y hit position of shape=(tracks, [hits])
        - ak.Array, z hit position of shape=(tracks, [hits])
        - ak.Array, hit energy of shape=(tracks, [hits])

    Optional
    --------
        - np.array, number of hits in track position of shape=(tracks,) (commented)
        - np.array, energy integrated over track of shape=(tracks,) (commented)
    """
    with uproot.open(name) as sig_root:
        qtree = sig_root["qtree"]

        # (track, [hits]) nested branches
        normalize = lambda arr: arr - ak.mean(arr, axis=-1)
        xs = normalize(qtree["TrackPostX"].array())
        ys = normalize(qtree["TrackPostY"].array())
        zs = normalize(qtree["TrackPostZ"].array())
        Es = qtree["TrackEnergy"].array()

        # (track,) non-nested branches
        # these shoudn't be needed, uncomment the lines to load them
        # track_energy = qtree_sig["DepositedEnergy"].array(library="np")
        # nb_hits = qtree_sig["NTrack"].array(library="np")

    if is_signal:
        # concatenate the two b tracks (from two consecutive rows)
        cat_fn = lambda arr: ak.concatenate([arr[::2], arr[1::2]], axis=1)
        xs = cat_fn(xs)
        ys = cat_fn(ys)
        zs = cat_fn(zs)
        Es = cat_fn(Es)

        # merge_bb_fn = lambda x: x[::2] + x[1::2]
        # track_energy = merge_bb_fn(track_energy)
        # nb_hits = merge_bb_fn(nb_hits)
    return xs, ys, zs, Es


def tracks2histograms(xs, ys, zs, Es, geo):
    """
    Compute energy histogram from track hit positions. The image is

    Parameters
    ----------
        - xs: ak.Arrays, x hit position of shape=(tracks, [hits])
        - ys: ak.Arrays, y hit position of shape=(tracks, [hits])
        - zs: ak.Arrays, z hit position of shape=(tracks, [hits])
        - Es: ak.Arrays, hit energy of shape=(tracks, [hits])
        - geo: Geometry, detector geometry

    Returns
    -------
        - ak.Arrays, sparse energy histogram of shape=()
    """
    print("Converting to histogram...")
    hists = []

    # filter out of bin range hits
    mx = np.logical_and(xs < geo.xbins[-1], xs >= geo.xbins[0])
    my = np.logical_and(ys < geo.xbins[-1], ys >= geo.xbins[0])
    mz = np.logical_and(zs < geo.xbins[-1], zs >= geo.xbins[0])
    m = np.logical_and(np.logical_and(mx, my), mz)
    for x, y, z, energy in tqdm(zip(xs[m], ys[m], zs[m], Es[m])):
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
        hists.append(hist.reshape(1, -1))
    hists = sparse.vstack(hists)
    return hists


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="the input folder")
    parser.add_argument(
        "--output", "-o", type=Path, help="the output folder", default=Path("./data")
    )
    parser.add_argument(
        "--show", action="store_true", help="show a track visual example "
    )
    args = parser.parse_args()

    start = tm()
    for file in args.input.iterdir():
        if file.suffix == ".root":
            print("Opening ", file.name)
            # just check that file is just signal or background
            assert file.name[0] in ["b", "e"]
            is_signal = file.name[0] == "b"
            xs, ys, zs, Es = load_tracks(file, is_signal)
            track_fn = lambda t, i: t[i].to_numpy()
            if args.show:
                xresolution = 5
                yresolution = 1
                get_image(
                    file.name,
                    track_fn(xs, 0),
                    track_fn(zs, 0),
                    track_fn(Es, 0),
                    xresolution,
                    yresolution,
                )
            lims = (-50, 50)  # cube edge
            geo = Geometry(xlim=lims, ylim=lims, zlim=lims)
            data_sparse = tracks2histograms(xs, ys, zs, Es, geo)
            sparse.save_npz(args.output / file.name, data_sparse)
    print(f"Program done in {tm()-start}s")

# TODO: think about dropping Etot and nb_hits as they seem redundant information
