from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from csdata.prepare import prepare
from csdata.registry import list_specs, load_spec
from csdata.validate import validate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="csdata")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    subparsers.add_parser("list")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("dataset")
    prepare_parser.add_argument("--out", required=True)
    prepare_parser.add_argument("--schema", choices=["name", "idx"], default="name")
    prepare_parser.add_argument("--naming", choices=["real", "anonymized"], default=None)
    prepare_parser.add_argument("--cache-dir", default=None)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("dataset")
    validate_parser.add_argument("--dir", required=True)
    validate_parser.add_argument("--schema", choices=["name", "idx"], default="name")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "list":
            print("\n".join(list_specs()))
            return 0

        if args.cmd == "prepare":
            out = prepare(
                args.dataset,
                args.out,
                schema=args.schema,
                naming=args.naming,
                cache_dir=args.cache_dir,
            )
            print(f"wrote {out}")
            return 0

        if args.cmd == "validate":
            import pandas as pd

            spec = load_spec(args.dataset)
            out_dir = Path(args.dir)
            info = json.loads((out_dir / "info.json").read_text(encoding="utf-8"))
            full_csv = out_dir / "full.csv"
            df = pd.read_csv(full_csv) if full_csv.exists() else None
            msgs = validate(info, spec=spec, df=df, schema=args.schema)
            for msg in msgs:
                print(msg)
            return 1 if msgs else 0
    except (KeyError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
