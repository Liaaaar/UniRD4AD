import os
import warnings
from argparse import ArgumentParser

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.optim.lr_scheduler import StepLR

from datasets.datasets import build_dataset, normalize_dataset_name
from models.decoders.de_resnet import (
    de_resnet18,
    de_resnet34,
    de_resnet50,
    de_wide_resnet50_2,
    de_wide_resnet101_2,
)
from models.encoders.resnet import (
    resnet18,
    resnet34,
    resnet50,
    wide_resnet50_2,
    wide_resnet101_2,
)
from models.losses.losses import ClusterLoss, RDLoss
from utils.utils import evaluation, setup_seed

BACKBONES = {
    "resnet18": (resnet18, de_resnet18, 512),
    "resnet34": (resnet34, de_resnet34, 512),
    "resnet50": (resnet50, de_resnet50, 2048),
    "wide_resnet50_2": (wide_resnet50_2, de_wide_resnet50_2, 2048),
    "wide_resnet101_2": (wide_resnet101_2, de_wide_resnet101_2, 2048),
}


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--dataset",
        default="mvtec",
        type=str.lower,
        choices=["mvtec", "visa", "realiad"],
    )
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument(
        "--backbone",
        default="wide_resnet50_2",
        type=str.lower,
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "wide_resnet50_2",
            "wide_resnet101_2",
        ],
    )
    parser.add_argument(
        "--rd_loss",
        default="cosine",
        type=str.lower,
        choices=["cosine", "arc", "ssim", "mse", "mae"],
    )
    parser.add_argument(
        "--cluster_loss",
        default="cosine",
        type=str.lower,
        choices=["cosine", "arc", "ssim", "mse", "mae", "ssot"],
        # ssot is best for mvtec, cosine is best for visa and realiad
    )
    parser.add_argument("--cluster_scale", default=None, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--eval_interval", default=5, type=int)
    parser.add_argument("--scheduler_step", default=20, type=int)
    parser.add_argument("--scheduler_gamma", default=0.5, type=float)
    parser.add_argument("--cluster_weight", default=1.0, type=float)
    parser.add_argument("--data_root", default="D:/documents/datasets/mvtec", type=str)
    parser.add_argument("--save_path", default="checkpoints", type=str)
    parser.add_argument("--resume", default="", type=str)
    return parser.parse_args()


def build_dataloaders(args):
    train_data = build_dataset(
        args.dataset, args.img_size, args.data_root, mode="train"
    )
    test_data = build_dataset(args.dataset, args.img_size, args.data_root, mode="test")

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_data, train_loader, test_loader


def build_models(args):
    if args.backbone not in BACKBONES:
        raise ValueError(f"Unsupported backbone: {args.backbone}")

    encoder_fn, decoder_fn, latent_channels = BACKBONES[args.backbone]
    spatial_size = max(1, args.img_size // 32)

    encoder, bn = encoder_fn(pretrained=True)
    decoder = decoder_fn(pretrained=False)
    cluster_loss = ClusterLoss(
        channels=latent_channels,
        spatial_size=spatial_size,
        mode=args.cluster_loss,
        scale=args.cluster_scale,
    )
    return encoder, bn, decoder, cluster_loss


def load_cluster_loss_state(cluster_loss, checkpoint):
    for key in ("cluster_loss", "svdd"):
        if key not in checkpoint:
            continue

        try:
            cluster_loss.load_state_dict(checkpoint[key], strict=True)
            print(f"loaded {key} state from checkpoint")
        except RuntimeError as exc:
            print(f"skip loading {key} state: {exc}")
        return


def maybe_resume(args, device, bn, decoder, cluster_loss, optimizer, scheduler):
    best = {
        "epoch": 0,
        "score": 0.0,
        "auroc_sp": None,
        "auroc_px": None,
        "aupro_px": None,
    }
    history = {
        "train_epochs": [],
        "eval_epochs": [],
        "loss_rd": [],
        "loss_cluster": [],
        "loss_total": [],
        "auroc_sp": [],
        "auroc_px": [],
        "aupro_px": [],
    }

    if not args.resume:
        return 1, best, history

    checkpoint = torch.load(args.resume, map_location=device)
    bn.load_state_dict(checkpoint["bn"])
    decoder.load_state_dict(checkpoint["decoder"])
    load_cluster_loss_state(cluster_loss, checkpoint)

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    best.update(checkpoint.get("best", {}))
    history.update(checkpoint.get("history", {}))
    start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"resume from checkpoint: {args.resume}")
    return start_epoch, best, history


def save_history_plot(history, save_path):
    fig, axes = plt.subplots(3, 2, figsize=(8, 12))

    axes[0][0].plot(history["eval_epochs"], history["auroc_sp"])
    axes[0][0].set_title("auroc_sp")
    axes[0][1].plot(history["eval_epochs"], history["auroc_px"])
    axes[0][1].set_title("auroc_px")
    axes[1][0].plot(history["eval_epochs"], history["aupro_px"])
    axes[1][0].set_title("aupro_px")
    axes[1][1].plot(history["train_epochs"], history["loss_rd"])
    axes[1][1].set_title("loss_rd")
    axes[2][0].plot(history["train_epochs"], history["loss_cluster"])
    axes[2][0].set_title("loss_cluster")
    axes[2][1].plot(history["train_epochs"], history["loss_total"])
    axes[2][1].set_title("loss_total")

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, "monitor_traning.png"), dpi=100)
    plt.close(fig)


def train():
    args = get_args()
    args.dataset = normalize_dataset_name(args.dataset)
    setup_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    print(f"dataset = {args.dataset}")
    print(f"backbone = {args.backbone}")
    print(f"rd_loss = {args.rd_loss}")
    print(f"cluster_loss = {args.cluster_loss}")

    train_data, train_loader, test_loader = build_dataloaders(args)
    encoder, bn, decoder, cluster_loss = build_models(args)
    rd_loss = RDLoss(args.rd_loss)

    encoder.to(device)
    encoder.eval()
    encoder.requires_grad_(False)
    bn.to(device)
    decoder.to(device)
    cluster_loss.to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": decoder.parameters(), "lr": args.lr},
            {"params": bn.parameters(), "lr": args.lr * 0.1},
            {"params": cluster_loss.parameters(), "lr": args.lr * 0.1},
        ],
        betas=(0.5, 0.999),
    )
    scheduler = StepLR(
        optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
    )

    start_epoch, best, history = maybe_resume(
        args, device, bn, decoder, cluster_loss, optimizer, scheduler
    )

    for epoch in range(start_epoch, args.epochs + 1):
        bn.train()
        decoder.train()
        cluster_loss.train()

        loss_rd_sum = 0.0
        loss_cluster_sum = 0.0

        for img, _, _ in train_loader:
            img = img.to(device)

            with torch.no_grad():
                teacher_features = encoder(img)

            embedding = bn(teacher_features)
            student_features = decoder(embedding)

            loss_rd_value = rd_loss(teacher_features, student_features)
            loss_cluster_value = cluster_loss(embedding)
            loss = loss_rd_value + args.cluster_weight * loss_cluster_value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_rd_sum += loss_rd_value.item()
            loss_cluster_sum += loss_cluster_value.item()

        scheduler.step()

        mean_loss_rd = loss_rd_sum / len(train_loader)
        mean_loss_cluster = loss_cluster_sum / len(train_loader)
        mean_loss_total = mean_loss_rd + args.cluster_weight * mean_loss_cluster

        history["train_epochs"].append(epoch)
        history["loss_rd"].append(mean_loss_rd)
        history["loss_cluster"].append(mean_loss_cluster)
        history["loss_total"].append(mean_loss_total)

        print(
            f"epoch [{epoch}/{args.epochs}], "
            f"loss_total:{mean_loss_total:.4f}, "
            f"loss_rd:{mean_loss_rd:.4f}, "
            f"loss_cluster:{mean_loss_cluster:.4f}"
        )

        if epoch % args.eval_interval != 0:
            continue

        auroc_px, auroc_sp, aupro_px = evaluation(
            encoder, bn, decoder, test_loader, device, train_data.classes
        )

        history["eval_epochs"].append(epoch)
        history["auroc_sp"].append(auroc_sp["mean"])
        history["auroc_px"].append(auroc_px["mean"])
        history["aupro_px"].append(aupro_px["mean"])

        print(f"Sample Auroc{auroc_sp}\nPixel Auroc:{auroc_px}\nPixel Aupro{aupro_px}")

        score = (auroc_sp["mean"] + auroc_px["mean"] + aupro_px["mean"]) / 3
        if score <= best["score"]:
            continue

        best["epoch"] = epoch
        best["score"] = score
        best["auroc_sp"] = auroc_sp
        best["auroc_px"] = auroc_px
        best["aupro_px"] = aupro_px

        cluster_state = cluster_loss.state_dict()
        torch.save(
            {
                "epoch": epoch,
                "args": vars(args),
                "best": best,
                "history": history,
                "bn": bn.state_dict(),
                "decoder": decoder.state_dict(),
                "cluster_loss": cluster_state,
                "svdd": cluster_state,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            os.path.join(args.save_path, "best_model.pth"),
        )

    if best["auroc_sp"] is None:
        raise RuntimeError(
            "No evaluation was run. Please check eval_interval and epochs."
        )

    metrics = {"class": [], "AUROC_sample": [], "AUROC_pixel": [], "AUPRO_pixel": []}
    for cls_name in best["auroc_sp"]:
        metrics["class"].append(cls_name)
        metrics["AUROC_sample"].append(best["auroc_sp"][cls_name])
        metrics["AUROC_pixel"].append(best["auroc_px"][cls_name])
        metrics["AUPRO_pixel"].append(best["aupro_px"][cls_name])

    pd.DataFrame(metrics).to_csv(
        os.path.join(args.save_path, "best_results.csv"), index=False
    )

    print(
        "Epoch {} is the Best !\nSample Auroc: {:.4f}, Pixel Auroc: {:.4f}, Pixel Aupro: {:.4f}".format(
            best["epoch"],
            best["auroc_sp"]["mean"],
            best["auroc_px"]["mean"],
            best["aupro_px"]["mean"],
        )
    )

    save_history_plot(history, args.save_path)


if __name__ == "__main__":
    train()
