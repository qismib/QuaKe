"""
    This module contains the main data generation subcommand functions: it opens
    root files with simulated energy depositions  from <input folder> and dumps
    converted images in `.npz` format to <output folder>.

    Usage example:

    ```
    quake datagen ../root_folder -o ../output_folder
    ```
"""
import logging
from pathlib import Path
from scipy import sparse
from quake import PACKAGE
from .generate_utils import load_tracks, get_image, Geometry, tracks2histograms

logger = logging.getLogger(PACKAGE + ".datagen")


def add_arguments_datagen(parser):
    """
    Adds datagen subparser arguments.

    Parameters
    ----------
        - parser: ArgumentParser, datagen subparser object
    """
    parser.add_argument(
        "input", type=Path, help="the input folder", default=Path("./root files")
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="the output folder", default=Path("./data")
    )
    parser.add_argument(
        "--xresolution", type=int, help="the x axis binning resolution in mm", default=5
    )
    parser.add_argument(
        "--yresolution", type=int, help="the y axis binning resolution in mm", default=5
    )
    parser.add_argument(
        "--zresolution", type=int, help="the z axis binning resolution in mm", default=1
    )
    parser.add_argument(
        "--show", action="store_true", help="show a track visual example "
    )
    # TODO: if the algorithm introduces some random function, we must add a
    # `--seed` argument for reproducibility
    parser.set_defaults(func=datagen)


def datagen(args):
    """
    Data generation wrapper function: calls the data generation main function.

    Parameters
    ----------
        - args: NameSpace object, command line parsed arguments.
    """
    # TODO [enhancement]: maybe transform CMD line parsed arguments into a runcard
    # no need to do that as long as there are just a few settings
    datagen_main(
        args.input,
        args.output,
        args.xresolution,
        args.yresolution,
        args.zresolution,
        args.show,
    )


def datagen_main(
    in_folder, out_folder, xresolution, yresolution, zresolution, should_show=False
):
    """
    Data generation main function: extracts a dataset from a folder containing
    root files.

    Parameters
    ----------
        - in_folder: Path, the input folder path
        - out_folder: Path, the output folder path
        - xresolution: int, the x axis binning resolution in mm
        - yresolution: int, the y axis binning resolution in mm
        - zresolution: int, the z axis binning resolution in mm
        - should_show: bool, wether to show a visual example or not
    """
    for file in in_folder.iterdir():
        if file.suffix == ".root":
            logger.info(f"Opening {file}")
            # check that file is just signal or background
            assert file.name[0] in ["b", "e"]
            is_signal = file.name[0] == "b"
            xs, ys, zs, Es = load_tracks(file, is_signal)
            track_fn = lambda t, i: t[i].to_numpy()
            if should_show:
                # a plot for debugging purposes
                get_image(
                    file.name,
                    track_fn(xs, 0),
                    track_fn(zs, 0),
                    track_fn(Es, 0),
                    xresolution,
                    zresolution,
                )
            lims = (-20, 20)  # assume cubic geometry
            bin_w = (xresolution, yresolution, zresolution)
            geo = Geometry(xlim=lims, ylim=lims, zlim=lims, bin_w=bin_w)
            data_sparse = tracks2histograms(xs, ys, zs, Es, geo)
            fname = out_folder / file.name
            sparse.save_npz(fname, data_sparse)
            logger.info(f"Histogram saved at {fname}.npz")
