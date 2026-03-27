#!/usr/bin/env python

from __future__ import annotations

import argparse

from lerobot.cloud.materializer import FilesystemEpisodeMaterializer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize committed edge episodes into local datasets.")
    parser.add_argument("--ingestion-root", required=True, help="Directory containing committed episode uploads.")
    parser.add_argument("--output-root", required=True, help="Directory for materialized JSON outputs.")
    parser.add_argument(
        "--lerobot-root",
        help="Optional output directory for a local LeRobotDataset export.",
    )
    parser.add_argument(
        "--repo-id",
        default="local/edge-materialized",
        help="Repo id metadata to embed when exporting a LeRobotDataset.",
    )
    parser.add_argument("--fps", type=int, default=20, help="Dataset fps when exporting a LeRobotDataset.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    materializer = FilesystemEpisodeMaterializer(
        ingestion_root=args.ingestion_root,
        output_root=args.output_root,
    )
    if args.lerobot_root:
        manifest = materializer.materialize_to_lerobot_dataset(
            repo_id=args.repo_id,
            dataset_root=args.lerobot_root,
            fps=args.fps,
        )
    else:
        manifest = materializer.materialize_all()
    print(
        f"materialized episodes={manifest.episode_count} steps={manifest.total_step_count} "
        f"repo_id={manifest.lerobot_repo_id or '-'}"
    )


if __name__ == "__main__":
    main()
