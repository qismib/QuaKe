"""
    This module contains the main data generation subcommand functions: it opens
    root files with simulated energy depositions  from <input folder> and dumps
    converted images in `.npz` format to <output folder>.

    Usage example:

    ```
    deeplar datagen ../root_folder -o ../output_folder
    ```
"""
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from scipy import sparse
from deeplar import PACKAGE
from .generate_utils import load_tracks, get_image, Geometry, tracks2histograms
from deeplar.utils.utils import load_runcard, save_runcard, initialize_output_folder

logger = logging.getLogger(PACKAGE + ".datagen")


def add_arguments_datagen(parser: ArgumentParser):
    """Adds datagen subparser arguments.

    Parameters
    ----------
    parser: ArgumentParser
        Datagen subparser object.
    """
    parser.add_argument(
        "runcard",
        type=Path,
        help="the input folder",
        default=Path("./cards/runcard.yaml"),
    )

    parser.add_argument(
        "--output", "-o", type=Path, help="the output folder", default=Path("./data")
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing files if present"
    )
    parser.add_argument(
        "--show", action="store_true", help="show a track visual example"
    )
    parser.add_argument("--energy", type=float)
    parser.add_argument("--res", type=float)
    parser.set_defaults(func=datagen)


def datagen(args: Namespace):
    """Data generation wrapper function.

    Calls the data generation main function.

    Parameters
    ----------
    args: Namespace
        Command line parsed arguments.
    """
    # load runcard and setup output folder structure
    setup = load_runcard(args.runcard)
    setup.update({"output": args.output})
    initialize_output_folder(args.output, args.force)
    save_runcard(args.output / "cards/runcard.yaml", setup)
    # save a default runcard in folder to allow default resoration
    save_runcard(args.output / "cards/runcard_default.yaml", setup)

    # launch main datagen function
    datagen_main(
        setup["dataset_dir"],
        setup["output"] / "data",
        setup["detector"],
        args.show,
    )


def datagen_main(
    in_folder: Path,
    out_folder: Path,
    detector: dict,
    should_show: bool = False,
):
    """Data generation main function.

    Extracts a dataset from a folder containing root files.

    Parameters
    ----------
    in_folder: Path
        The input folder path.
    out_folder: Path
        The output folder path.
    detector: dict
        The detector geometry settings.
    should_show: bool
        Wether to show a visual example or not.
    """
    logger.info(f"Generate data to {out_folder}/data")
    xresolution, _, zresolution = detector["resolution"]
    for file in in_folder.iterdir():
        if file.suffix == ".root":
            logger.info(f"Opening {file}")
            # check that file is just signal or background
            assert file.name[0] in ["b", "e"]
            is_signal = file.name[0] == "b"
            geo = Geometry(detector)
            xs, ys, zs, Es = load_tracks(file, geo, is_signal)
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
            data_sparse = tracks2histograms(xs, ys, zs, Es, geo)
            fname = out_folder / file.name

            sparse.save_npz(fname, data_sparse)
            logger.info(f"Histogram saved at {fname}.npz")
