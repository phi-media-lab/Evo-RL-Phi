# Evo-RL

This branch is a working delivery branch for one concrete target:

- single-repo deployment of the ROCm + MI300X + `pi05` Evo-RL ACP workflow

The goal is that a new MI300X machine can clone this repository, bootstrap the environment, run validation, and start the staged workflow without depending on a separate `AMD_Hackathon` repo.

## Branch Status

Branch:

- `pi05-rocm-acp`

Validated on this branch:

- fresh clone of this repository
- bootstrap into `.venvs/pi05-openpi-ssp`
- OpenPI-patched `transformers==4.53.2`
- real `pi05` inference with `google/paligemma-3b-pt-224` and `lerobot/pi05_base`
- key tests:
  - `pytest -q tests/training/test_acp_pi05_prompt_pipeline.py tests/policies/pi0_pi05/test_pi05.py`
  - result: `4 passed, 1 warning`
- staged workflow:
  - `smoke`
  - `pilot`
  - `stage1`
  - `stage2`
  - `stage3`
- staged validation confirmed through `500` training steps with checkpointing

## Fastest Path

On a fresh MI300X instance:

```bash
git clone https://github.com/phi-media-lab/Evo-RL-Phi.git
cd Evo-RL-Phi
git checkout pi05-rocm-acp
bash scripts/setup/bootstrap_runpod_mi300x.sh
source .venvs/pi05-openpi-ssp/bin/activate
hf auth login
pytest -q tests/training/test_acp_pi05_prompt_pipeline.py tests/policies/pi0_pi05/test_pi05.py
bash scripts/experiments/pi05_acp/run_pi05_acp_smoke.sh
```

For the full staged rerun:

```bash
bash scripts/experiments/pi05_acp/run_pi05_acp_full_rerun.sh rerun_YYYYMMDD
```

## Main Entrypoints

- [`scripts/setup/bootstrap_runpod_mi300x.sh`](./scripts/setup/bootstrap_runpod_mi300x.sh)
- [`scripts/setup/setup_rocm_pi05.sh`](./scripts/setup/setup_rocm_pi05.sh)
- [`scripts/experiments/pi05_acp/`](./scripts/experiments/pi05_acp)
- [`docs/source/pi05_acp_mi300x.mdx`](./docs/source/pi05_acp_mi300x.mdx)
- [`docs/source/pi05_acp_runpod_quickstart.mdx`](./docs/source/pi05_acp_runpod_quickstart.mdx)
- [`docs/source/pi05_acp/runbook.md`](./docs/source/pi05_acp/runbook.md)
- [`docs/source/pi05_acp/execution_plan.md`](./docs/source/pi05_acp/execution_plan.md)
- [`docs/source/pi05_acp/experiment_report.md`](./docs/source/pi05_acp/experiment_report.md)

## Branch-Specific Changes

The branch carries the compatibility and workflow changes needed to make `pi05` usable on ROCm in this environment, including:

- `pi05` tokenizer/config loading cleanup
- OpenPI-patched transformers compatibility fixes in `pi05`
- no-KV-cache fallback for `select_action`
- `value-infer` support for explicit `dataset.video_backend`
- repository-local setup, bootstrap, smoke, and staged workflow scripts
- branch-focused MI300X / RunPod documentation

## Known Constraints

- requires ROCm-capable MI300X hardware
- requires Hugging Face access to `google/paligemma-3b-pt-224`
- current validated data path uses `pyav`
- current validated dataset is `maxbeau/XLeRobot`
- dataset assumptions still include:
  - no `episode_success`
  - no state quantile stats
  - local cache or first-download access required

## Current Recommendation

Use this branch when you want to:

- reproduce the validated MI300X `pi05` ACP environment
- verify that the single-repo bootstrap path works
- continue branch-local work on scaling the staged workflow beyond `stage3`

Do not use this README as a generic overview of the whole project. It is intentionally scoped to the current branch progress.

## License

Apache-2.0. See [LICENSE](./LICENSE).
