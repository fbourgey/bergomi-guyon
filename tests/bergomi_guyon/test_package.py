"""Check installed Bergomi--Guyon imports and CLI entry points."""

import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest


def test_installed_package_imports_from_another_directory(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
from bergomi_guyon import generate_coefficients, verify
import sys
assert 'numpy' not in sys.modules
assert 'sympy' not in sys.modules
verify(generate_coefficients(2))
from bergomi_guyon.bg_coefficients_order_6 import a
assert len(a) == 7
""",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.mark.parametrize("entry_point", ["module", "console"])
def test_installed_cli_from_another_directory(tmp_path, entry_point):
    if entry_point == "module":
        command = [sys.executable, "-I", "-m", "bergomi_guyon"]
    else:
        suffix = ".exe" if sys.platform == "win32" else ""
        command = [
            str(
                Path(sysconfig.get_path("scripts"))
                / ("generate-bg-coefficients" + suffix)
            )
        ]
    process = subprocess.run(
        [*command, "--order", "2", "--verify", "--quiet"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert process.stdout == ""
    assert "verified orders 1 through 2" in process.stderr
