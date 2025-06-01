import torch
import os
from pathlib import Path


def save_weights(model, model_name, dir="data/weights"):
    save_path = Path(dir) / f"{model_name}.pt"
    torch.save(model.state_dict(), save_path)


def create_checkpoint(
    dir,
    model,
    epoch,
    optimizer=None,
    scaler=None,
    scheduler=None,
):
    save_dict = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
    }

    if optimizer:
        save_dict["optimizer_state_dict"] = optimizer.state_dict()

    if scaler:
        save_dict["scaler_state_dict"] = scaler.state_dict()

    if scheduler:
        save_dict["scheduler_state_dict"] = scheduler.state_dict()

    # checkpoint dir
    dir_path = Path.cwd() / dir
    try:
        dir_path.mkdir(parents=True)
    except:
        pass

    save_path = dir_path / f"checkpoint_{epoch}.pt"
    link_path = dir_path / "checkpoint.pt"
    torch.save(save_dict, save_path)

    # link last checkpoint
    try:
        os.remove(link_path)
    except:
        pass
    os.symlink(save_path, link_path)


def load_checkpoint(
    dir,
    model,
    optimizer=None,
    scaler=None,
    scheduler=None,
    name="checkpoint.pt",
):
    checkpoint = torch.load(Path(dir) / name)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"]
