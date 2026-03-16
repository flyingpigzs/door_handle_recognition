import torch


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    avg_loss = total_loss / max(len(dataloader), 1)
    print(f"Epoch [{epoch}] Training Loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def validate_one_epoch(model, dataloader, device, epoch):
    model.train()
    total_loss = 0.0

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

    avg_loss = total_loss / max(len(dataloader), 1)
    print(f"Epoch [{epoch}] Validation Loss: {avg_loss:.4f}")
    return avg_loss