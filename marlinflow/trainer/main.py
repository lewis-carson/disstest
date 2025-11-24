from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random

from dataloader import BatchLoader
from model import (
    NnBoard768Cuda,
    NnBoard768,
    NnHalfKA,
    NnHalfKACuda,
    NnHalfKP,
    NnHalfKPCuda,
)
from time import time

import torch
import wandb
from replay import ReplayBuffer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOG_ITERS = 10_000_000


class WeightClipper:
    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "weight"):
            w = module.weight.data
            w = w.clamp(-1.98, 1.98)
            module.weight.data = w


def acpl_to_elo(acpl: float, total_time_sec: float, ref_time_sec: float = 300, alpha: float = 0.28) -> float:
    """
    Estimate engine Elo from ACPL and total clock time per player using CCRL-style scaling.
    
    Parameters:
        acpl (float): Average centipawn loss at reference time.
        total_time_sec (float): Total clock time per player (seconds).
        ref_time_sec (float): Reference time for which ACPL was measured (default 300s).
        alpha (float): Time-scaling exponent (default 0.28 for CCRL population).
        
    Returns:
        float: Estimated Elo (CCRL-style).
    """
    # Scale ACPL based on available time
    acpl_effective = acpl * (ref_time_sec / total_time_sec) ** alpha
    
    # Power-law mapping from ACPL to Elo (CCRL adjustments)
    elo = 1250 + 9000 * acpl_effective ** -0.5
    return elo

def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: BatchLoader,
    wdl: float,
    scale: float,
    epochs: int,
    save_epochs: int,
    train_id: str,
    lr_drop: int | None = None,
    use_wandb: bool = False,
    replay_buffer: ReplayBuffer | None = None,
    replay_prob: float = 0.0,
    replay_min_batches: int = 1,
) -> None:
    clipper = WeightClipper()
    running_loss = torch.zeros((1,), device=DEVICE)
    start_time = time()
    iterations = 0

    loss_since_log = torch.zeros((1,), device=DEVICE)
    iter_since_log = 0

    fens = 0
    epoch = 0

    replay_steps = 0

    while epoch < epochs:
        new_epoch, fresh_batch = dataloader.read_batch(DEVICE)
        if new_epoch:
            epoch += 1
            if epoch == lr_drop:
                optimizer.param_groups[0]["lr"] *= 0.1
            print(
                f"epoch {epoch}",
                f"epoch train loss: {running_loss.item() / iterations}",
                f"epoch pos/s: {fens / (time() - start_time)}",
                sep=os.linesep,
            )

            if use_wandb:
                epoch_loss = running_loss.item() / iterations
                epoch_acpl = 4 * scale * (epoch_loss ** 0.5)
                wandb.log(
                    {
                        "epoch": epoch,
                        "epoch_loss": epoch_loss,
                        "epoch_acpl": epoch_acpl,
                        "epoch_elo_3m": acpl_to_elo(epoch_acpl, 180),
                        "epoch_elo_15m": acpl_to_elo(epoch_acpl, 900),
                        "epoch_elo_40m": acpl_to_elo(epoch_acpl, 2400),
                        "pos_per_s": fens / (time() - start_time),
                    }
                )

            running_loss = torch.zeros((1,), device=DEVICE)
            start_time = time()
            iterations = 0
            fens = 0
            replay_steps = 0

            if epoch % save_epochs == 0:
                torch.save(model.state_dict(), f"nn/{train_id}_{epoch}")
                param_map = {
                    name: param.detach().cpu().numpy().tolist()
                    for name, param in model.named_parameters()
                }
                with open(f"nn/{train_id}.json", "w") as json_file:
                    json.dump(param_map, json_file)

        batch = fresh_batch
        if replay_buffer is not None:
            replay_buffer.add(fresh_batch)
            if (
                replay_prob > 0.0
                and replay_buffer.can_sample(replay_min_batches)
                and random.random() < replay_prob
            ):
                batch = replay_buffer.sample_to_device(DEVICE)
                replay_steps += 1

        optimizer.zero_grad()
        prediction = model(batch)
        expected = torch.sigmoid(batch.cp / scale) * (1 - wdl) + batch.wdl * wdl

        loss = torch.mean((prediction - expected) ** 2)
        loss.backward()
        optimizer.step()
        model.apply(clipper)

        with torch.no_grad():
            running_loss += loss
            loss_since_log += loss
        iterations += 1
        iter_since_log += 1
        fens += batch.size

        if iter_since_log * batch.size > LOG_ITERS:
            loss = loss_since_log.item() / iter_since_log
            replay_ratio = replay_steps / iterations if iterations else 0.0
            print(
                f"At {iterations * batch.size} positions",
                f"Running Loss: {loss}",
                f"Replay ratio: {replay_ratio:.2f}",
                sep=os.linesep,
            )
            if use_wandb:
                acpl = 4 * scale * (loss ** 0.5)
                wandb.log(
                    {
                        "loss": loss,
                        "acpl": acpl,
                        "elo_3m": acpl_to_elo(acpl, 180),
                        "elo_15m": acpl_to_elo(acpl, 900),
                        "elo_40m": acpl_to_elo(acpl, 2400),
                        "global_step": iterations * batch.size,
                    }
                )

            iter_since_log = 0
            loss_since_log = torch.zeros((1,), device=DEVICE)


def main():
    print(f"Training on {DEVICE}")

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--data-root", type=str, help="Root directory of the data files"
    )
    parser.add_argument("--train-id", type=str, help="ID to save train logs with")
    parser.add_argument("--lr", type=float, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, help="Epochs to train for")
    parser.add_argument("--batch-size", type=int, default=16384, help="Batch size")
    parser.add_argument("--wdl", type=float, default=0.0, help="WDL weight to be used")
    parser.add_argument("--scale", type=float, help="WDL weight to be used")
    parser.add_argument(
        "--save-epochs",
        type=int,
        default=100,
        help="How often the program will save the network",
    )
    parser.add_argument(
        "--lr-drop",
        type=int,
        default=None,
        help="The epoch learning rate will be dropped",
    )
    parser.add_argument("--wandb-project", type=str, help="Wandb project name")
    parser.add_argument("--wandb-entity", type=str, help="Wandb entity name")
    parser.add_argument(
        "--replay-buffer-size",
        type=int,
        default=0,
        help="How many past batches to keep for replay",
    )
    parser.add_argument(
        "--replay-prob",
        type=float,
        default=0.0,
        help="Probability of training on a replayed batch instead of a fresh one",
    )
    parser.add_argument(
        "--replay-min-batches",
        type=int,
        default=1,
        help="Minimum stored batches before replay can start",
    )
    parser.add_argument(
        "--replay-priority-metric",
        type=str,
        choices=["cp_norm", "cp_abs_mean", "cp_var"],
        default=None,
        help="Optional metric for prioritized replay sampling",
    )
    parser.add_argument(
        "--replay-priority-eps",
        type=float,
        default=1e-6,
        help="Floor added to computed priorities to keep them positive",
    )

    args = parser.parse_args()

    assert args.scale is not None

    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
        )
        if args.train_id is None:
            args.train_id = wandb.run.name

    if args.train_id is None:
        args.train_id = str(int(time()))

    model = NnHalfKPCuda(256).to(DEVICE)

    data_path = pathlib.Path(args.data_root)
    paths = list(map(str, data_path.rglob("*.binpack")))
    dataloader = BatchLoader(paths, model.input_feature_set(), args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    replay_buffer = None
    if args.replay_buffer_size > 0:
        replay_buffer = ReplayBuffer(
            args.replay_buffer_size,
            DEVICE,
            priority_metric=args.replay_priority_metric,
            priority_eps=args.replay_priority_eps,
        )

    train(
        model,
        optimizer,
        dataloader,
        args.wdl,
        args.scale,
        args.epochs,
        args.save_epochs,
        args.train_id,
        lr_drop=args.lr_drop,
        use_wandb=args.wandb_project is not None,
        replay_buffer=replay_buffer,
        replay_prob=args.replay_prob,
        replay_min_batches=args.replay_min_batches,
    )


if __name__ == "__main__":
    main()
