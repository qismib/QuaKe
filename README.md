# DeepLAr

DeepLAr: Deep Learning classifier for low-energy events in Liquid Argon TPC

## Installation

The package can be installed with Python's `pip` package manager.

```bash
git clone https://github.com/CERN-IT-INNOVATION/DeepLAr.git
cd DeepLAr
pip install .[MODE]
```

The last command allows to install the `deeplar` program into the environment
python path.  

:warning: **Note: install the appropriate TensorFlow distribution**

`deeplar` assumes that the user has already installed the most optimized version
of TensorFlow for his platform. As such, by default, `pip` will not check it as
a requirement.

However, the user can also install it specifying a `MODE` option in the
`pip` command. The list below summarizes the valid choices for the `MODE` flag:

- `tf`: installs the `tensorflow` package
- `tf-cpu`: installs the `tensorflow-cpu` package
- `tf-gpu`: installs the `tensorflow-gpu` package
- `tf-amd`: installs the `tensorflow-rocm` package

## Running the code

In order to launch the code

```bash
deeplar <subcommand> [options]
```

Valid subcommands are: `datagen` | `train`.  
Use `deeplar <subcommand> --help` to print the correspondent help message.  
For example, the help message for `datagen` subcommand is:

```bash
$ deeplar datagen --help
usage: deeplar datagen [-h] [--output OUTPUT] [--force] [--show] runcard

generate voxelized dataset from root files

positional arguments:
  runcard               the input folder

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        the output folder
  --force               overwrite existing files if present
  --show                show a track visual example
```

### Configuration cards

Models' parameter settings are stored in `yaml` files. The [cards](cards) folder
contains some examples.

## Pipeline (TODO)

Implement a Makefile to run the entire pipeline given a runcard. A default
subcommand with the entire pipeline can also be implemented.

### Data generation

Extracts histograms from 3D simulated energy depositions.

```bash
deeplar datagen <runcard.yaml> --output <output folder> [--force]
```

The `.yaml` runcard should store the path to dataset folder containing `.root`
files and the bin widths setting the histogram resolution.

An equivalent runcard is copied to the output folder in order to pass settings
to subsequent steps of the pipeline.

### Model training

Train a model on data extracted in folder.

```bash
deeplar train <folder> --model <modeltype>
```


