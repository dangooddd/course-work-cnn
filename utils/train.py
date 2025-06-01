import torch
from sklearn.metrics import f1_score
from rich.progress import track
from .checkpoint import create_checkpoint


@torch.compile
def train_epoch(model, criterion, loader, device, optimizer, scaler):
    model.train()
    loss_epoch = 0.0
    total_samples = 0
    total_true = []
    total_pred = []

    for images, labels in loader:
        # setup
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        # forward
        optimizer.zero_grad()
        with torch.autocast(device, dtype=torch.float16):
            outs = model(images)
            loss = criterion(outs, labels)

        # backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # stats
        with torch.no_grad():
            preds = torch.argmax(outs, dim=1)
            loss_epoch += loss.item() * batch_size
            total_samples += batch_size
            total_true.append(labels.cpu())
            total_pred.append(preds.cpu())

    total_true = torch.cat(total_true).numpy()
    total_pred = torch.cat(total_pred).numpy()

    loss_epoch /= total_samples
    f1_epoch = f1_score(total_true, total_pred, average="micro")
    return loss_epoch, f1_epoch


@torch.compile
@torch.no_grad()
def test_epoch(model, criterion, loader, device):
    model.eval()
    loss_epoch = 0.0
    total_samples = 0
    total_true = []
    total_pred = []

    for images, labels in loader:
        # setup test data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        with torch.autocast(device, dtype=torch.float16):
            outs = model(images)
            loss = criterion(outs, labels)

        # stats
        preds = torch.argmax(outs, dim=1)
        loss_epoch += loss.item() * batch_size
        total_samples += batch_size
        total_true.append(labels.cpu())
        total_pred.append(preds.cpu())

    total_true = torch.cat(total_true).numpy()
    total_pred = torch.cat(total_pred).numpy()

    loss_epoch /= total_samples
    f1_epoch = f1_score(total_true, total_pred, average="micro")
    return loss_epoch, f1_epoch


def train(
    model,
    train_loader,
    test_loader,
    device,
    criterion,
    optimizer,
    scaler,
    scheduler=None,
    writer=None,
    epochs=10,
    start_epoch=0,
    checkpoint_dir="data/checkpoints",
    checkpoint_step=10,
):
    for epoch in track(
        range(start_epoch, start_epoch + epochs), description="Training..."
    ):
        # train
        loss, f1 = train_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
        )

        # test
        test_loss, test_f1 = test_epoch(
            model=model,
            criterion=criterion,
            loader=test_loader,
            device=device,
        )

        if writer:
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("F1-score/train", f1, epoch)
            writer.add_scalar("F1-score/test", test_f1, epoch)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

        # checkpoint
        if (epoch + 1) % checkpoint_step == 0:
            create_checkpoint(
                dir=checkpoint_dir,
                model=model,
                epoch=epoch + 1,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
            )

    return start_epoch + epochs
