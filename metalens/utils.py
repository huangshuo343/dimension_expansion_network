import os
import torch


def save_checkpoint(model, optimizer, epoch, save_dir,
                    model_name="latest_checkpoint"):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        epoch: Current epoch.
        save_dir: Save directory.
        model_name: Checkpoint name.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    save_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, save_dir,
                    model_name="latest_checkpoint"):
    """
    Load model checkpoint.

    Returns:
        epoch (int):
            Loaded epoch.

        optimizer_loaded (bool):
            Whether optimizer state was successfully loaded.
    """
    load_path = os.path.join(save_dir, f"{model_name}.pth")

    checkpoint = torch.load(
        load_path,
        map_location="cpu"
    )

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except KeyError:
        model.load_state_dict(checkpoint)

    optimizer_loaded = True

    try:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
    except Exception:
        optimizer_loaded = False
        print(
            "Optimizer state not found. "
            "Using a newly initialized optimizer."
        )

    epoch = checkpoint.get("epoch", 0)

    return epoch, optimizer_loaded


