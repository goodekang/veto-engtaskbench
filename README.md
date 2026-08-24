# VETO / EngTaskBench-220

Framework code and offline replay harness for the two-domain tool-agent benchmark (120 BIM-RCC tasks, 100 CAD-PC tasks). Replay runs on cached observations, so no live model APIs or raw IFC/STEP files are needed.

## Setup

```bash
python -m pip install -e .
```

Python 3.10+ and PyTorch are required. Scripts pick CUDA when it is visible.

## Layout

- `src/veto/` — blackboard, planner, broker, executor, verifier, supervisor, policy, train/eval engine
- `configs/` — replay budgets and the 35-epoch offline-policy loop
- `data/splits/bench220.csv` — task index and gold labels
- `data/tools/registry.json` — the 42-tool registry
- `scripts/`, `examples/` — evaluation, robustness sweep, plotting, one-task demos

## Artefacts not in this repository

The cached observation bags (`data/features/`), the published policy checkpoint (`checkpoints/veto_gpt4o_r1/`), prediction dumps and run logs are derived from third-party IFC and STEP corpora and are not redistributed here. They are available from the corresponding author on reasonable request; the commands below expect them under the paths shown.

## Reproduce the main numbers

```bash
python scripts/smoke_forward.py
python scripts/eval_checkpoint.py --ckpt checkpoints/veto_gpt4o_r1/best.pt --split test
python -m veto.evaluate --pred results/predictions/veto_gpt4o_r1.csv
python scripts/plot_from_results.py
```

`eval_checkpoint.py` runs the official no-grad scores on the frozen bags, then extra TTA / saliency / stretched-encoder work on the same batch. Expect a full test pass on the order of a minute, not a couple of seconds.

Observation-dropout sweep:

```bash
python scripts/eval_robustness.py --ckpt checkpoints/veto_gpt4o_r1/best.pt
```

## One-task demo

Schependomlaan egress clear-width:

```bash
python examples/infer_one_task.py
```

Mounting-bracket CAD case:

```bash
python examples/infer_one_task.py --case cad
```

## Offline policy training

A from-scratch fit writes to `checkpoints/veto_gpt4o_r1_scratch/` unless you pass `--overwrite`. The loop is 35 epochs of AdamW + cosine + hard-tier oversampling and takes on the order of an hour.

```bash
python -m veto.train --help
python scripts/train.py
```

## Notes

Raw buildingSMART / Schependomlaan IFC files and Fusion 360 / DeepCAD parts are not shipped. `data/README.md` lists the public sources.

Budgets follow the paper defaults: `K=3` repairs, `R=2` re-plans, broker shortlist `k=15`.

Code is released under the MIT License (`LICENSE`).
