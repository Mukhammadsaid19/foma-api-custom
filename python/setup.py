import re
from pathlib import Path
from setuptools import setup

__doc__ = """\
This package contains a modified version of the foma.py Python API file
from https://github.com/mhulden/foma/tree/master/foma/python, allowing it
to be installed via pip as
    pip install 'git+https://github.com/andrewdotn/foma#egg=foma&subdirectory=foma/python'
This package requires that libfoma be installed; this package does not
include foma itself.
"""

def get_version():
    "Parse the current foma version out of fomalib.h"

    version = []

    header_file = Path(__file__).parent / ".." / "fomalib.h"
    header_text = header_file.read_text()
    for key in [
        "MAJOR_VERSION",
        "MINOR_VERSION",
        "BUILD_VERSION",
        "STATUS_VERSION",
    ]:
        match = re.search(
            f'^#define {key} (?:([0-9]+)|"([^"]*)")$', header_text, re.MULTILINE
        )

        if not match:
            raise Exception(
                f"Unable to find or parse ‘#define {key}’ line from {header_file}"
            )
        if match.group(1):
            version.append(str(match.group(1)))
        elif match.group(2):
            version.append(match.group(2))

    return ".".join(version)


setup(name="foma", version=get_version(), py_modules=['foma'])