# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import lightning as L
import torch
from torch.utils.data import DataLoader
from functools import partial
from lit_gpt.model import GPT, Block, MBlock, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops
from lit_gpt.utils import chunked_cross_entropy, num_parameters
from pytorch_lightning.loggers import WandbLogger
from lit_gpt import FusedCrossEntropyLoss
import random
import os
os.environ["WANDB_MODE"] = "dryrun"

# Suppressing Warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='torch')

model_name = "Samba_421M"  # change to "Samba_1.3B" for 1.3B model
train_config = "tsz512x4k_20B"  # change to "tsz512x4k_100B" for 1.3B model
name = train_config + "_" + model_name

out_dir = Path(os.getenv("LIGHTNING_ARTIFACTS_DIR", "out")) / name

# Hyperparameters
if "20B" in name:
    nodes = 1
    max_tokens = int(1e11) // 5
elif "100B" in name:
    nodes = 8
    max_tokens = int(1e11)

if "512x4k" in name:
    global_batch_size = 512 // nodes
    micro_batch_size = 8
elif "256x8k" in name:
    global_batch_size = 256 // nodes
    micro_batch_size = 4
elif "128x16k" in name:
    global_batch_size = 128 // nodes
    micro_batch_size = 2
elif "64x32k" in name:
    global_batch_size = 64 // nodes
    micro_batch_size = 1
elif "1024x2k" in name:
    global_batch_size = 1024 // nodes
    micro_batch_size = 16

learning_rate = 4e-4

total_evals = 400
warmup_tokens = int(max_tokens * 0.01)
log_step_interval = 10
eval_iters = total_evals // micro_batch_size
save_step_interval = 1000
eval_step_interval = 1000

num_extrapol = 4

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = learning_rate / 10

batch_size = global_batch_size
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0

log_iter_interval = log_step_interval * gradient_accumulation_steps

train_data_config = [
    ("train_slim", 1.0),
]

val_data_config = [
    ("validation", 1.0),
]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}

wandb_logger = WandbLogger(project="pretrain-LLM")


def setup(
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    resume: Union[bool, Path] = False,
) -> None:
    
    # Use CPU for training
    cpu_cores = 4
    fabric = L.Fabric(devices=cpu_cores, precision="bf16-mixed", loggers=[wandb_logger])
    fabric.launch()
    fabric.print(hparams)
    fabric.logger.log_hyperparams(hparams)

    main(fabric, train_data_dir, val_data_dir, resume,)


def main(fabric, train_data_dir, val_data_dir, resume, **overrides):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name, **overrides)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=3407,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(partial(model._init_weights, n_layer=config.n_layer))
 
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")
    fabric.print(model)
    
    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), fused=True
    )
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        resume = sorted(out_dir.glob("*.pth"))[-1]
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, resume)
    fabric.print(f"Training time: {(time.perf_counter() - train_time):.2f}s")


def train(fabric, state, train_dataloader, val_dataloader, monitor, resume):
    model = state["model"]
    optimizer = state["optimizer"]

    if val_dataloader is not None:
        validate(fabric, model, val_dataloader)  # sanity check

    total_lengths = 0
    total_t0 = time.perf_counter()

    max_tokens_per_device = max_tokens
    tokens_per_iter = micro_batch_size * model.config.block_size
    max_iters = max_tokens_per_device // tokens_per_iter
    warmup_iters = warmup_tokens // tokens_per_iter
    initial_iter = state["iter_num"]
    curr_iter = 0
            
    loss_func = FusedCrossEntropyLoss()
    for train_data in train_dataloader:
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
        if state["iter_num"] >= max_iters:
            break
        
        lr = get_lr(state["iter_num"], warmup_iters, max_iters) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0: model.config.block_size].contiguous()
        targets = train_data[:, 1: model.config.block_size + 1].contiguous()
        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = loss_func(logits, targets)
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        state["iter_num"] += 1
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. " 
                f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']):.2f} seconds"
        )

        if state["iter_num"] % save_step_interval == 0:
            fabric.save(out_dir / f"checkpoint_{state['iter_num']}.pth", state)

        if state["iter_num"] % eval_step_interval == 0 and val_dataloader:
            validate(fabric, model, val_dataloader)
    
    fabric.print(f"Training took {(time.perf_counter() - total_t0) / 3600:.2f} hours")


def validate(fabric, model, val_dataloader):
    model.eval()
    val_loss = 0.0
    val_iters = len(val_dataloader)
    with torch.no_grad():
        for i, val_data in enumerate(val_dataloader):
            input_ids = val_data[:, 0: model.config.block_size].contiguous()
            targets = val_data[:, 1: model.config.block_size + 1].contiguous()
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets)
            val_loss += loss.item()
            if (i + 1) % log_step_interval == 0:
                fabric.print(f"Validation iter {i + 1}/{val_iters}: loss {loss.item():.4f}")

    model.train()
    fabric.print(f"Validation loss: {val_loss / val_iters:.4f}")


def get_lr(iter_num: int, warmup_iters: int, max_iters: int) -> float:
    if iter_num < warmup_iters:
        return learning_rate * (iter_num / warmup_iters)
    return max(
        learning_rate * 0.5 * (1 + math.cos(math.pi * (iter_num - warmup_iters) / (max_iters - warmup_iters))),
        min_lr,
    )


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric: L.Fabric,
    train_data_dir: Path,
    val_data_dir: Optional[Path],
    seed: int,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    random.seed(seed)
    n_chunks = 10  # Replace with appropriate value
    block_size = 1024  # Replace with appropriate value

    train_datasets = [
        PackedDataset(Path(train_data_dir) / f"{dataset}.bin", n_chunks=n_chunks, block_size=block_size)
        for dataset, _ in train_data_config
    ]
    train_dataset = CombinedDataset(train_datasets, seed=69)
    
    # Determine if the dataset is an IterableDataset
    is_iterable = isinstance(train_dataset, torch.utils.data.IterableDataset)
    
    # Set shuffle only if the dataset is not an IterableDataset
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=not is_iterable, drop_last=True, num_workers=4
    )
    
    val_dataloader = None
    if val_data_dir:
        val_datasets = [PackedDataset(Path(val_data_dir) / f"{dataset}.bin", n_chunks=n_chunks, block_size=block_size) for dataset, _ in val_data_config]
        val_dataset = CombinedDataset(val_datasets, seed=69)
        
        # Determine if the dataset is an IterableDataset
        is_iterable = isinstance(val_dataset, torch.utils.data.IterableDataset)
        
        # Set shuffle only if the dataset is not an IterableDataset
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=not is_iterable, drop_last=False, num_workers=4
        )

    return train_dataloader, val_dataloader


if __name__ == "__main__":
    train_data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    val_data_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    resume = sys.argv[3] if len(sys.argv) > 3 else False
    setup(train_data_dir, val_data_dir, resume)
