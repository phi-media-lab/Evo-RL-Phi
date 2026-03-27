#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_policy_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a one-step training smoke test on a local LeRobotDataset.")
    parser.add_argument("--dataset-root", required=True, help="Local LeRobotDataset root.")
    parser.add_argument("--repo-id", required=True, help="Repo id metadata of the local LeRobotDataset.")
    parser.add_argument("--output-dir", required=True, help="Scratch output dir for the smoke run.")
    parser.add_argument("--policy-type", default="act", help="Policy type to instantiate.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the smoke step.")
    return parser


def run_training_smoke(
    *,
    dataset_root: str,
    repo_id: str,
    output_dir: str,
    policy_type: str = "act",
    batch_size: int = 1,
    save_policy: bool = False,
) -> dict[str, float | int | str]:
    output_dir = str(output_dir)
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=repo_id, root=dataset_root),
        policy=make_policy_config(policy_type, push_to_hub=False, device="cpu"),
        output_dir=output_dir,
        steps=1,
        batch_size=batch_size,
        num_workers=0,
        eval_freq=999999,
        save_freq=999999,
        log_freq=1,
    )
    cfg.validate()

    dataset = make_dataset(cfg)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta)
    optimizer, _ = make_optimizer_and_scheduler(cfg, policy)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=True)

    batch = next(iter(dataloader))
    policy.train()
    optimizer.zero_grad()
    loss, _ = policy.forward(batch)
    loss.backward()
    optimizer.step()

    result: dict[str, float | int | str] = {
        "dataset_num_frames": len(dataset),
        "dataset_num_episodes": dataset.num_episodes,
        "policy_type": policy_type,
        "loss": float(loss.item()),
        "dataset_root": str(dataset_root),
        "output_dir": output_dir,
        "stats_path": str(Path(dataset_root) / "meta" / "stats.json"),
    }
    if save_policy:
        policy_dir = Path(output_dir) / "pretrained_policy"
        policy.save_pretrained(policy_dir)
        result["policy_dir"] = str(policy_dir)
    return result


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_training_smoke(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        policy_type=args.policy_type,
        batch_size=args.batch_size,
    )
    print(
        f"training-smoke policy={result['policy_type']} episodes={result['dataset_num_episodes']} "
        f"frames={result['dataset_num_frames']} loss={result['loss']:.6f}"
    )


if __name__ == "__main__":
    main()
