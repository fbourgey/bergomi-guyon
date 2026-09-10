"""Generate exact Bergomi--Guyon coefficients.

Examples:
    uv run generate-bg-coefficients --order 6 --format text
    uv run generate-bg-coefficients --order 6 --verify --quiet
    uv run generate-bg-coefficients --order 6 --format python --output src/bergomi_guyon/bg_coefficients_order_6.py
"""

import argparse
import sys
from pathlib import Path

from . import generate_coefficients, verify
from .render import render_coefficients, render_python_module


def main() -> None:
    """Generate coefficients using command-line arguments.

    Notes
    -----
    Coefficients go to stdout or the requested file. Verification and file
    status messages go to stderr. Invalid arguments raise SystemExit through
    argparse; explicit output files are written even when --quiet is set.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--order", type=int, default=4, help="highest order to generate"
    )
    parser.add_argument(
        "--format",
        choices=("text", "latex", "python"),
        default="text",
        help="output format (default: text; latex requires paper tree macros not bundled here)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify the PDE, tree boundaries, grading, and original matching identity",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress coefficient output (useful with --verify)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write generated output to this path",
    )
    args = parser.parse_args()
    if args.order < 1:
        parser.error("--order must be at least one")

    result = generate_coefficients(args.order)
    if args.verify:
        verify(result)
        print(f"verified orders 1 through {args.order}", file=sys.stderr)
    if args.output is None and args.quiet:
        return
    rendered = (
        render_python_module(result.coefficients)
        if args.format == "python"
        else render_coefficients(result.coefficients, args.format)
    )
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote orders 1 through {args.order} to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
