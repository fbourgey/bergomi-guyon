"""CLI validation, output destinations, and status streams."""

import sys

import pytest

from bergomi_guyon import generate_coefficients
from bergomi_guyon.cli import main
from bergomi_guyon.render import render_coefficients, render_python_module


@pytest.mark.parametrize("order", ["0", "-1", "1.5", "hello"])
def test_cli_invalid_order(monkeypatch, capsys, order):
    monkeypatch.setattr(sys, "argv", ["generate-bg-coefficients", "--order", order])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err


@pytest.mark.parametrize("output_format", ["python", "latex", "text"])
def test_cli_output_file(monkeypatch, capsys, tmp_path, output_format):
    output = tmp_path / "coefficients"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate-bg-coefficients",
            "--order",
            "1",
            "--verify",
            "--quiet",
            "--format",
            output_format,
            "--output",
            str(output),
        ],
    )
    main()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "verified orders 1 through 1" in captured.err
    assert "wrote orders 1 through 1" in captured.err
    coefficients = generate_coefficients(1).coefficients
    expected = (
        render_python_module(coefficients)
        if output_format == "python"
        else render_coefficients(coefficients, output_format)
    )
    assert output.read_text() == expected


@pytest.mark.parametrize("quiet", [False, True])
def test_cli_stdout_without_verification(monkeypatch, capsys, quiet):
    arguments = ["generate-bg-coefficients", "--order", "1", "--format", "text"]
    monkeypatch.setattr(sys, "argv", arguments + (["--quiet"] if quiet else []))
    main()
    captured = capsys.readouterr()
    expected = render_coefficients(generate_coefficients(1).coefficients, "text")
    assert captured.out == ("" if quiet else expected)
    assert captured.err == ""
