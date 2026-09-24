"""Check installed magic-strikes imports outside the checkout."""

import subprocess
import sys


def test_installed_package_imports_from_another_directory(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
from magic_strikes import magic_strike, swap, utils
from magic_strikes.heston import HestonModel
from magic_strikes.rough_bergomi import RoughBergomiModel
from magic_strikes.model import ForwardVarianceModel
assert issubclass(HestonModel, ForwardVarianceModel)
assert issubclass(RoughBergomiModel, ForwardVarianceModel)
""",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
