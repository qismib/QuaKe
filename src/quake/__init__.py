"""Package source root directory. """
from importlib.metadata import metadata
from quake.utils.configlog import PACKAGE

__version__ = metadata(PACKAGE)["version"]
