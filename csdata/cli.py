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

    csv_parser = subparsers.add_parser("prepare-csv")
    csv_parser.add_argument("csv", help="path to a custom CSV file")
    csv_parser.add_argument("--target", required=True, help="target column name")
    csv_parser.add_argument("--out", required=True)
    csv_parser.add_argument("--schema", choices=["name", "idx"], default="name")
    csv_parser.add_argument("--name", default="custom", help="dataset name for artifacts")
    csv_parser.add_argument(
        "--task-type",
        choices=["binclass", "multiclass", "regression"],
        default=None,
        help="task type (inferred from the target when omitted)",
    )
    csv_parser.add_argument(
        "--drop-ids",
        action="store_true",
        help="drop columns that look like row identifiers instead of modelling them",
    )
    csv_parser.add_argument(
        "--parse-dates",
        action="store_true",
        help="auto-detect date columns (read as strings) and treat them as datetimes",
    )
    csv_parser.add_argument(
        "--date-threshold",
        type=float,
        default=0.9,
        help="fraction of values that must parse as dates for --parse-dates (default 0.9)",
    )

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("dataset")
    validate_parser.add_argument("--dir", required=True)
    validate_parser.add_argument("--schema", choices=["name", "idx"], default="name")

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("dataset", nargs="?", help="dataset name, or omit with --all")
    audit_parser.add_argument("--all", action="store_true", help="audit every UCI-mappable dataset")

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

        if args.cmd == "prepare-csv":
            import pandas as pd

            from csdata.infer import infer_spec, parse_date_columns

            df = pd.read_csv(args.csv)
            if args.parse_dates:
                df = parse_date_columns(df, threshold=args.date_threshold, skip={args.target})
            spec = infer_spec(
                df,
                target=args.target,
                name=args.name,
                task_type=args.task_type,
                drop_ids=args.drop_ids,
            )
            out = prepare(
                args.name,
                args.out,
                schema=args.schema,
                raw_df=df,
                spec=spec,
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

        if args.cmd == "audit":
            from csdata.audit import audit_types, auditable

            if not args.all and not args.dataset:
                print("error: provide a dataset name or --all", file=sys.stderr)
                return 2
            targets = auditable() if args.all else [args.dataset]
            try:
                for name in targets:
                    findings = audit_types(name)
                    if findings:
                        print(f"# {name}")
                        for msg in findings:
                            print(f"  - {msg}")
                    else:
                        print(f"# {name}: consistent with UCI")
            except ImportError:
                print("error: audit requires ucimlrepo (pip install 'csdata[audit]')", file=sys.stderr)
                return 2
            except ConnectionError:
                print("error: could not reach the UCI server (network required for audit)", file=sys.stderr)
                return 2
            return 0  # advisory only
    except (KeyError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
