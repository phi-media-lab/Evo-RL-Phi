#!/usr/bin/env python

from __future__ import annotations

import subprocess
from pathlib import Path


def test_control_plane_auto_release_tmux_script_has_valid_bash_syntax():
    script = Path("scripts/control_plane_auto_release_tmux.sh")
    result = subprocess.run(["bash", "-n", str(script)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_control_plane_auto_release_tmux_script_exposes_core_actions():
    script = Path("scripts/control_plane_auto_release_tmux.sh").read_text(encoding="utf-8")
    assert "start)" in script
    assert "stop)" in script
    assert "status)" in script
    assert "logs)" in script
    assert "python -m lerobot.scripts.control_plane_auto_release_daemon" in script


def test_control_plane_systemd_template_points_to_auto_release_daemon():
    service = Path("scripts/control_plane_auto_release.service").read_text(encoding="utf-8")
    assert "ExecStart=" in service
    assert "lerobot.scripts.control_plane_auto_release_daemon" in service
    assert "Restart=always" in service


def test_cloud_stack_tmux_script_has_valid_bash_syntax():
    script = Path("scripts/cloud_stack_tmux.sh")
    result = subprocess.run(["bash", "-n", str(script)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cloud_stack_tmux_script_exposes_core_actions():
    script = Path("scripts/cloud_stack_tmux.sh").read_text(encoding="utf-8")
    assert "start)" in script
    assert "stop)" in script
    assert "status)" in script
    assert "logs)" in script
    assert "python -m lerobot.scripts.cloud_stack" in script
    assert "--status-port" in script


def test_cloud_stack_systemd_template_points_to_cloud_stack_runner():
    service = Path("scripts/cloud_stack.service").read_text(encoding="utf-8")
    assert "ExecStart=" in service
    assert "lerobot.scripts.cloud_stack" in service
    assert "STATUS_PORT" in service
    assert "Restart=always" in service
