#!/usr/bin/env python

from __future__ import annotations

import argparse

from lerobot.cloud.materializer import MaterializerHTTPServer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve edge episode materialization over HTTP.")
    parser.add_argument("--ingestion-root", required=True, help="Directory containing committed episode uploads.")
    parser.add_argument("--output-root", required=True, help="Directory for materialized JSON outputs.")
    parser.add_argument(
        "--dataset-root",
        help="Optional directory for exported local LeRobotDataset outputs.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    server = MaterializerHTTPServer(
        (args.host, args.port),
        ingestion_root=args.ingestion_root,
        output_root=args.output_root,
        dataset_root=args.dataset_root,
    )
    print(f"materializer server listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
