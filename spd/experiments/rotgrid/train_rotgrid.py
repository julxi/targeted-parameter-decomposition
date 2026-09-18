"""Train a RotGridTransformer on uniform random walks over the rotating grid-world."""

from functools import partial
from pathlib import Path

import fire
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm, trange

# The learning-rate schedules are generic.
from spd.experiments.ih.train_ih import constant_lr, cosine_decay_lr, linear_lr, warmup_lr
from spd.experiments.rotgrid.configs import RotGridModelConfig, RotGridTrainConfig
from spd.experiments.rotgrid.dataset import RotGridDataset
from spd.experiments.rotgrid.model import RotGridTransformer
from spd.log import logger
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import set_seed
from spd.utils.run_utils import ExecutionStamp, read_noneable_str, save_file


def evaluate(
    model: RotGridTransformer, dataset: RotGridDataset, batch_size: int
) -> dict[str, float]:
    """Score the model against the exact next-token distribution.

    The truth is uniform over the legal tokens, so `kl_to_optimal` bottoms out at exactly zero,
    unlike `ce_nats` whose floor is the task's irreducible entropy of ~0.99 nats.
    """
    was_training = model.training
    model.eval()

    with torch.no_grad():
        tokens, legal = dataset.generate_batch(batch_size)
        logits = model(tokens)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        n_legal = legal.sum(dim=-1).float()
        mean_log_prob_of_legal = (log_probs * legal).sum(dim=-1) / n_legal
        kl_to_optimal = (-n_legal.log() - mean_log_prob_of_legal).mean()

        illegal = ~legal
        argmax_is_illegal = illegal.gather(-1, logits.argmax(dim=-1, keepdim=True)).squeeze(-1)

        ce_nats = F.cross_entropy(logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten())

    model.train(was_training)
    return {
        "ce_nats": ce_nats.item(),
        "kl_to_optimal": kl_to_optimal.item(),
        "illegal_prob_mass": (probs * illegal).sum(dim=-1).mean().item(),
        "illegal_argmax_rate": argmax_is_illegal.float().mean().item(),
        "entropy_floor_nats": n_legal[:, :-1].log().mean().item(),
    }


def train(
    model: RotGridTransformer,
    dataloader: DatasetGeneratedDataLoader[tuple[torch.Tensor, torch.Tensor]],
    dataset: RotGridDataset,
    config: RotGridTrainConfig,
    log_wandb: bool,
) -> tuple[list[float], list[int]]:
    match config.lr_schedule:
        case "linear":
            lr_schedule_fn = linear_lr
        case "cosine":
            lr_schedule_fn = cosine_decay_lr
        case "constant":
            lr_schedule_fn = constant_lr

    if config.lr_warmup > 0:
        warmup_steps = (
            config.lr_warmup
            if isinstance(config.lr_warmup, int)
            else int(config.lr_warmup * config.steps)
        )
        lr_schedule_fn = partial(warmup_lr, warmup_steps=warmup_steps, lr_fn=lr_schedule_fn)

    opt = torch.optim.AdamW(
        list(model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    losses: list[float] = []
    loss_steps: list[int] = []
    max_grad_norm_since_eval = 0.0

    data_iter = iter(dataloader)
    with trange(config.steps, ncols=0) as t:
        for step in t:
            step_lr = config.lr * lr_schedule_fn(step, config.steps)
            for group in opt.param_groups:
                group["lr"] = step_lr
            opt.zero_grad(set_to_none=True)

            tokens, _ = next(data_iter)
            logits = model(tokens)
            loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten())
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), config.grad_clip).item()
            opt.step()

            max_grad_norm_since_eval = max(max_grad_norm_since_eval, grad_norm)
            t.set_postfix(loss=loss.item(), lr=step_lr, grad_norm=grad_norm)

            if step % config.eval_freq == 0 or step + 1 == config.steps:
                metrics = evaluate(model, dataset, config.eval_batch_size)
                loss_steps.append(step)
                losses.append(loss.item())
                tqdm.write(
                    f"Step {step} loss {loss.item():.4f} "
                    f"kl {metrics['kl_to_optimal']:.4f} "
                    f"illegal {metrics['illegal_prob_mass']:.5f}"
                )
                if log_wandb:
                    wandb.log(
                        {
                            "loss": loss.item(),
                            "lr": step_lr,
                            "max_grad_norm": max_grad_norm_since_eval,
                            **metrics,
                        },
                        step=step,
                    )
                max_grad_norm_since_eval = 0.0

    return losses, loss_steps


def get_model_and_dataloader(
    config: RotGridTrainConfig, device: str
) -> tuple[
    RotGridTransformer,
    DatasetGeneratedDataLoader[tuple[torch.Tensor, torch.Tensor]],
    RotGridDataset,
]:
    model = RotGridTransformer(config.rotgrid_model_config).to(device)
    dataset = RotGridDataset(seq_len=config.rotgrid_model_config.seq_len, device=device)
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    return model, dataloader, dataset


def plot_loss_curve(
    losses: list[float], steps: list[int], entropy_floor: float, out_dir: Path
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, label="Training loss", color="blue")
    plt.axhline(
        entropy_floor,
        color="red",
        linestyle="--",
        label=f"Entropy floor ({entropy_floor:.3f} nats)",
    )
    plt.xlabel("Steps")
    plt.ylabel("Cross-entropy (nats)")
    plt.title("RotGrid training loss")
    plt.legend()
    plt.grid()
    plt.savefig(out_dir / "train_loss_curve.png")
    plt.close()


def plot_attention_maps(
    model: RotGridTransformer,
    dataset: RotGridDataset,
    batch_size: int,
    steps: int,
    out_dir: Path,
) -> None:
    model.eval()
    with torch.no_grad():
        weights = torch.cat(
            [
                model.get_attention_weights(dataset.generate_batch(batch_size)[0])
                for _ in range(steps)
            ],
            dim=0,
        )
        avg_weights = weights.mean(dim=0)
        max_weights = weights.max(dim=0).values

    for layer in range(model.config.n_layers):
        for head in range(model.config.n_heads):
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            assert isinstance(ax, np.ndarray)
            ax[0].imshow(avg_weights[layer, head].cpu().numpy(), cmap="viridis", aspect="auto")
            ax[0].set_title(f"Layer {layer + 1}, head {head + 1} - avg attention")
            ax[1].imshow(max_weights[layer, head].cpu().numpy(), cmap="viridis", aspect="auto")
            ax[1].set_title(f"Layer {layer + 1}, head {head + 1} - max attention")
            plt.colorbar(ax[0].images[0], ax=ax[0])
            plt.colorbar(ax[1].images[0], ax=ax[1])
            plt.tight_layout()
            fig.savefig(out_dir / f"attention_layer{layer + 1}_head{head + 1}.png")
            plt.close(fig)


def log_figures_to_wandb(out_dir: Path) -> None:
    """Upload the figures written at the end of training, which are otherwise lost with the run."""
    figures = sorted(out_dir.glob("*.png"))
    assert figures, f"No figures were written to {out_dir}"
    wandb.log({f"figures/{path.stem}": wandb.Image(str(path)) for path in figures})


def run_train(config: RotGridTrainConfig, device: str) -> Path:
    model, dataloader, dataset = get_model_and_dataloader(config, device)
    model_config = config.rotgrid_model_config

    execution_stamp = ExecutionStamp.create(run_type="train", create_snapshot=False)
    out_dir = execution_stamp.out_dir
    logger.info(f"Run ID: {execution_stamp.run_id}")
    logger.info(f"Output directory: {out_dir}")

    log_wandb = config.wandb_project is not None
    if config.wandb_project:
        run_name = (
            f"rotgrid_layers{model_config.n_layers}_dmodel{model_config.d_model}"
            f"_heads{model_config.n_heads}_seq{model_config.seq_len}"
            f"_steps{config.steps}_batch{config.batch_size}_lr{config.lr}"
        )
        wandb.init(
            id=execution_stamp.run_id,
            project=config.wandb_project,
            name=run_name,
            tags=[f"rotgrid_{model_config.n_layers}L"],
        )

    config_path = out_dir / "rotgrid_train_config.yaml"
    save_file(config.model_dump(mode="json"), config_path)
    if config.wandb_project:
        wandb.save(str(config_path), base_path=out_dir, policy="now")
    logger.info(f"Saved config to {config_path}")

    losses, loss_steps = train(
        model=model, dataloader=dataloader, dataset=dataset, config=config, log_wandb=log_wandb
    )

    final_metrics = evaluate(model, dataset, config.eval_batch_size)
    logger.info(f"Final metrics: {final_metrics}")

    plot_loss_curve(
        losses=losses,
        steps=loss_steps,
        entropy_floor=final_metrics["entropy_floor_nats"],
        out_dir=out_dir,
    )
    plot_attention_maps(
        model=model,
        dataset=dataset,
        batch_size=config.eval_batch_size,
        steps=config.attention_maps_n_steps,
        out_dir=out_dir,
    )
    if log_wandb:
        log_figures_to_wandb(out_dir)

    model_path = out_dir / "rotgrid.pth"
    save_file(model.state_dict(), model_path)
    if config.wandb_project:
        wandb.save(str(model_path), base_path=out_dir, policy="now")
    logger.info(f"Saved model to {model_path}")

    return out_dir


def main(
    n_layers: int,
    d_model: int,
    n_heads: int,
    steps: int,
    wandb_project: str,
    seq_len: int = 256,
    batch_size: int = 256,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    seed: int = 0,
) -> None:
    config = RotGridTrainConfig(
        wandb_project=read_noneable_str(wandb_project),
        rotgrid_model_config=RotGridModelConfig(
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_fanout=4,
            use_ff=True,
            use_pos_encoding=True,
            use_layer_norm=True,
        ),
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        lr_warmup=500,
        weight_decay=0.01,
        grad_clip=grad_clip,
        lr_schedule="cosine",
        seed=seed,
        eval_freq=100,
        eval_batch_size=64,
        attention_maps_n_steps=4,
    )
    set_seed(config.seed)
    run_train(config=config, device=get_device())


if __name__ == "__main__":
    fire.Fire(main)
