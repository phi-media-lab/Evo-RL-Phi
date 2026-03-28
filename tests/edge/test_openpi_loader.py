#!/usr/bin/env python

from __future__ import annotations

from types import SimpleNamespace

from lerobot.edge.openpi_loader import (
    OpenPICheckpointPolicyRef,
    parse_openpi_policy_ref,
    resolve_openpi_policy_ref,
)


def test_parse_openpi_checkpoint_policy_ref():
    spec = parse_openpi_policy_ref(
        "openpi://checkpoint?config=pi0_aloha_sim&dir=/tmp/ckpt&default_prompt=pick&pytorch_device=cpu"
    )

    assert spec == OpenPICheckpointPolicyRef(
        config="pi0_aloha_sim",
        checkpoint_dir="/tmp/ckpt",
        default_prompt="pick",
        pytorch_device="cpu",
        openpi_root=spec.openpi_root,
        openpi_client_root=spec.openpi_client_root,
    )


def test_resolve_openpi_policy_ref_uses_training_config_and_checkpoint(monkeypatch):
    calls: dict[str, object] = {}

    class FakePolicyConfigModule:
        @staticmethod
        def create_trained_policy(train_config, checkpoint_dir, *, default_prompt=None, pytorch_device=None):
            calls["train_config"] = train_config
            calls["checkpoint_dir"] = checkpoint_dir
            calls["default_prompt"] = default_prompt
            calls["pytorch_device"] = pytorch_device
            return {"policy": "fake-openpi-policy"}

    class FakeTrainingConfigModule:
        @staticmethod
        def get_config(config_name):
            calls["config_name"] = config_name
            return SimpleNamespace(name=config_name)

    monkeypatch.setattr(
        "lerobot.edge.openpi_loader._import_openpi_modules",
        lambda spec: (FakePolicyConfigModule, FakeTrainingConfigModule),
    )

    policy = resolve_openpi_policy_ref(
        "openpi://checkpoint?config=pi0_aloha_sim&dir=/tmp/ckpt&default_prompt=pick&pytorch_device=cpu"
    )

    assert policy == {"policy": "fake-openpi-policy"}
    assert calls["config_name"] == "pi0_aloha_sim"
    assert calls["checkpoint_dir"] == "/tmp/ckpt"
    assert calls["default_prompt"] == "pick"
    assert calls["pytorch_device"] == "cpu"
