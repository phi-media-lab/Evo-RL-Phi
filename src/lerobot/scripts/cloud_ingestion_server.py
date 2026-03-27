#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.cloud.ingestion import EpisodeIngestionHTTPServer, FilesystemEpisodeIngestionStore


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal HTTP ingestion service.")
    parser.add_argument("--root", required=True, help="Filesystem durability root for committed episodes.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    server = EpisodeIngestionHTTPServer(
        (args.host, args.port),
        FilesystemEpisodeIngestionStore(Path(args.root)),
    )
    print(f"ingestion-server listening on http://{args.host}:{args.port} root={args.root}")
    server.serve_forever()


if __name__ == "__main__":
    main()
