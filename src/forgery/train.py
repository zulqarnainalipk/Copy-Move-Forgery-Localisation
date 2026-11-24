"""Two-stage training: decoder warmup with a frozen encoder, then joint fine-tuning."""
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from forgery.config import SEED, Paths
from forgery.data import load_split_table, make_loader
from forgery.losses import build_loss
from forgery.model import ForgerySegmenter, count_parameters


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None, accumulation=1, desc=""):
    training = optimizer is not None
    model.train(training)
    total, steps = 0.0, 0
    bar = tqdm(loader, desc=desc, leave=False)
    if training:
        optimizer.zero_grad()
    for index, (images, masks) in enumerate(bar):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=torch.float16, enabled=scaler is not None):
            loss = loss_fn(model(images).float(), masks)
        if training:
            scaler.scale(loss / accumulation).backward() if scaler else (loss / accumulation).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
        total += loss.item()
        steps += 1
        bar.set_postfix(loss=f"{total / steps:.4f}")
    return total / max(steps, 1)


def run_stage(model, loaders, optimizer, epochs, patience, loss_fn, tag, device, accumulation, amp, checkpoint,
              history, best=float("inf"), eta_min=1e-6):
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    scaler = torch.cuda.amp.GradScaler() if amp and device.type == "cuda" else None
    stalled = 0
    for epoch in range(1, epochs + 1):
        started = time.time()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss = run_epoch(model, loaders["train"], loss_fn, device, optimizer, scaler, accumulation,
                               f"{tag} epoch {epoch}/{epochs} train")
        val_loss = run_epoch(model, loaders["val"], loss_fn, device, desc=f"{tag} epoch {epoch}/{epochs} val")
        scheduler.step()
        saved = val_loss < best
        if saved:
            best, stalled = val_loss, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stalled += 1
        history.append({"stage": tag, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "lr": learning_rate, "seconds": time.time() - started, "saved": saved})
        print(f"[{tag}] epoch {epoch:02d}/{epochs} train {train_loss:.4f} val {val_loss:.4f} "
              f"lr {learning_rate:.2e} {'saved' if saved else f'patience {stalled}/{patience}'} "
              f"({time.time() - started:.0f}s)", flush=True)
        if stalled >= patience:
            print(f"[{tag}] early stop")
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return best


def train(args, paths=None):
    """Two-stage training: decoder warmup, then joint fine-tuning of the encoder and decoder."""
    paths = paths or Paths.from_env()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = paths.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoint.pt"

    table = load_split_table(run_dir.parent / "splits.csv" if args.splits is None else args.splits)
    train_table = table[table.split == "train"]
    val_table = table[table.split == "val"]
    if args.max_images:
        train_table, val_table = train_table.head(args.max_images), val_table.head(args.max_images)
    print(f"[data] device {device} | train {len(train_table)} | val {len(val_table)} | "
          f"seed {args.seed} | augment {args.augment} | encoder input {args.encoder_input} | head {args.head} | loss {args.loss}")
    loaders = {
        "train": make_loader(train_table, paths, args.batch_size, True, args.augment, args.img_size, args.workers, args.seed),
        "val": make_loader(val_table, paths, args.batch_size, False, "none", args.img_size, args.workers),
    }

    model = ForgerySegmenter(paths.dinov2_dir, args.head, args.encoder_input, args.img_size).to(device)
    print(f"[model] parameters {count_parameters(model) / 1e6:.2f}M | decoder {count_parameters(model.head) / 1e6:.2f}M")
    loss_fn = build_loss(args.loss)
    history = []

    print("[stage 1] decoder warmup")
    optimizer = AdamW(model.head.parameters(), lr=args.lr_head, weight_decay=1e-4)
    best = run_stage(model, loaders, optimizer, args.epochs1, args.patience1, loss_fn, "S1", device,
                     args.accumulation, args.amp, checkpoint, history)

    print("[stage 2] joint fine-tuning")
    model.unfreeze_encoder()
    print(f"[model] trainable parameters {count_parameters(model, True) / 1e6:.2f}M")
    optimizer = AdamW([{"params": model.head.parameters(), "lr": args.lr_head},
                       {"params": model.encoder.parameters(), "lr": args.lr_encoder}], weight_decay=1e-4)
    best = run_stage(model, loaders, optimizer, args.epochs2, args.patience2, loss_fn, "S2", device,
                     args.accumulation, args.amp, checkpoint, history, best, eta_min=args.lr_encoder)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    print(f"[done] best validation loss {best:.4f} | checkpoint {checkpoint}")
    return checkpoint
