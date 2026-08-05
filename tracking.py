"""Optional Weights & Biases experiment tracking.

Enable by setting the environment variable USE_WANDB=1 before running training.
When disabled, all calls become no-ops so the code runs anywhere.
"""
import os

_USE_WANDB = os.getenv("USE_WANDB", "0") == "1"

if _USE_WANDB:
    import wandb


def init(project: str, config: dict | None = None):
    if _USE_WANDB:
        wandb.init(project=project, config=config or {})


def log(metrics: dict):
    if _USE_WANDB:
        wandb.log(metrics)


def finish():
    if _USE_WANDB:
        wandb.finish()
