import time
from typing import Tuple

import torch

import utils


def train_one_epoch(model, loss_fn, optimizer, data_loader, device) -> Tuple[float, int]:
    model.train()
    optimizer.zero_grad()

    t1 = time.time()
    train_loss = utils.MovingAverage(name="train-loss")

    for batch in data_loader:
        img = batch["image"].to(device)
        target = batch["label"].to(device)

        out, _ = model(img)
        loss = loss_fn(out, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss.update(loss.item())

    t2 = time.time()
    timetaken = round(t2 - t1)

    return train_loss.value(), timetaken


def evaluate(model, loss_fn, data_loader, device):
    model.eval()

    t1 = time.time()
    test_loss = utils.MovingAverage(name="test-loss")
    test_acc = utils.MovingAverage(name="test-acc")

    with torch.no_grad():
        for batch in data_loader:
            img = batch["image"].to(device).float()
            label = batch["label"].to(device).long()

            out, _ = model(img)
            pred = torch.argmax(out, dim=1)

            loss = loss_fn(out, label)
            acc = torch.mean(pred == label, dtype=torch.float32)

            test_loss.update(loss.item())
            test_acc.update(acc.item())

    t2 = time.time()
    timetaken = round(t2 - t1)

    return test_loss.value(), test_acc.value(), timetaken
