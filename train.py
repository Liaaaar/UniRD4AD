import os
import torch
import torch.nn.functional as F
import warnings
from utils.train import setup_seed, Simplex_CLASS
from utils.test import evaluation
from datasets.datasets import Uni_MVTecDataset
from models.encoders.resnet import (
    resnet50,
    wide_resnet50_2,
    resnet18,
    resnet34,
    wide_resnet101_2,
)
from models.decoders.de_resnet_v2 import (
    de_resnet50,
    de_wide_resnet50_2,
    de_wide_resnet101_2,
    de_resnet18,
    de_resnet34,
)
from models.losses.losses import SVDD_loss, RD_loss, SSIM_loss
from torch.optim.lr_scheduler import StepLR
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--backbone", default="wide_resnet50_2", type=str)
    parser.add_argument("--rd_loss", default="cosine", type=str)  # ssim/cosine
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", default=0.01, type=float)  # 0.01
    parser.add_argument("--scheduler", default=(20, 0.5), type=tuple)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--data_root", default="D:/documents/datasets/mvtec", type=str)
    parser.add_argument("--save_path", default="checkpoints", type=str)
    args = parser.parse_args()
    return args


args = get_args()


def train():
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("device =", device)

    train_data = Uni_MVTecDataset(
        img_size=args.img_size, root=args.data_root, mode="train"
    )
    test_data = Uni_MVTecDataset(
        img_size=args.img_size, root=args.data_root, mode="test"
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=8
    )

    svdd = SVDD_loss()

    if args.rd_loss == "cosine":
        rd_loss = RD_loss()
    elif args.rd_loss == "ssim":
        rd_loss = SSIM_loss()

    def get_encoder(backbone):
        if backbone == "wide_resnet50_2":
            return wide_resnet50_2(pretrained=True)
        if backbone == "wide_resnet101_2":
            return wide_resnet101_2(pretrained=True)
        elif backbone == "resnet50":
            return resnet50(pretrained=True)
        elif backbone == "resnet18":
            return resnet18(pretrained=True)
        elif backbone == "resnet34":
            return resnet34(pretrained=True)

    def get_decoder(backbone):
        if backbone == "wide_resnet50_2":
            return de_wide_resnet50_2(pretrained=False)
        if backbone == "wide_resnet101_2":
            return de_wide_resnet101_2(pretrained=False)
        elif backbone == "resnet50":
            return de_resnet50(pretrained=False)
        elif backbone == "resnet18":
            return de_resnet18(pretrained=False)
        elif backbone == "resnet34":
            return de_resnet34(pretrained=False)

    encoder, bn = get_encoder(args.backbone)
    decoder = get_decoder(args.backbone)
    encoder.to(device).eval()

    # encoder.load_state_dict(
    #     torch.load(
    #         "/home/fkw/.cache/torch/hub/checkpoints/wide_resnet50_2-9ba9bcbe.pth"
    #     )
    # )

    #! checkpoint = torch.load("20240226/baseline_cosine_withnorm/best_model.pth")["svdd"]
    #! svdd.load_state_dict(checkpoint)

    bn.to(device)
    decoder.to(device)
    svdd.to(device)

    # optimizer = torch.optim.Adam(
    #     #list(decoder.parameters()) + list(bn.parameters()) + list(svdd.parameters()),

    #     lr=args.lr,
    #     betas=(0.5, 0.999),
    #     # weight_decay=1e-6,
    # )
    optimizer = torch.optim.Adam(
        [
            {"params": decoder.parameters(), "lr": args.lr},
            {"params": bn.parameters(), "lr": args.lr * 0.1},
            {"params": svdd.parameters(), "lr": args.lr * 0.1},
        ],
        betas=(0.5, 0.999),
        # weight_decay=1e-6,
    )
    scheduler = StepLR(optimizer, step_size=args.scheduler[0], gamma=args.scheduler[1])

    best = {
        "best_epoch": 0,
        "best_score": 0,
        "best_auroc_sp": 0,
        "best_auroc_px": 0,
        "best_aupro_px": 0,
    }
    history = {
        "auroc_sp": [],
        "auroc_px": [],
        "aupro_px": [],
        "loss_rd": [],
        "loss_svdd": [],
        "loss_total": [],
    }
    # gen_noise = Simplex_CLASS()
    for epoch in range(args.epochs):
        bn.train()
        decoder.train()
        loss_rd_sum = 0
        loss_svdd_sum = 0
        # loss_rd_norm_sum = 0
        # loss_rd_noise_sum = 0
        # loss_svdd_norm_sum = 0
        # loss_svdd_noise_sum = 0
        # loss_cos_sum = 0
        embedding_center = 0
        for img, cls, cls_id in train_dataloader:
            img = img.to(device)
            # noise = torch.randn_like(img, device=device)
            # img_noisy = torch.zeros_like(img, device=device)
            # noise = torch.Tensor(
            #     gen_noise.rand_3d_octaves((3, args.img_size, args.img_size), 6, 0.6)
            # ).to(device)

            # img_noisy = 0.8 * img + 0.2 * noise
            # cls = cls.to(device)
            cls_id = cls_id.to(device)
            inputs = encoder(img)
            embedding = bn(inputs)
            # embedding = F.layer_norm(embedding, embedding.shape[1:])
            # embedding = F.normalize(embedding, dim=1)
            outputs = decoder(embedding)  # bn(inputs))

            #! ema center
            # embedding_center = 0.99 * embedding_center + 0.01 * embedding.mean(
            #     dim=0, keepdim=True
            # )
            # loss_svdd = torch.cdist(embedding, embedding_center, p=2).mean()

            # inputs_noisy = encoder(img_noisy)
            # embedding_noisy = bn(inputs_noisy)
            # outputs_noisy = decoder(embedding_noisy)

            # rd_norm = rd_loss(inputs, outputs)
            loss_rd = rd_loss(inputs, outputs)
            # rd_noise = rd_loss(inputs, outputs_noisy)
            # loss_rd = rd_norm + rd_noise
            #! center loss
            # loss_svdd = (
            #     1
            #     - F.cosine_similarity(
            #         embedding, embedding.mean(dim=0, keepdim=True), dim=1
            #     ).mean()
            # )
            # loss_svdd = torch.acos(
            #     torch.clip(
            #         F.cosine_similarity(
            #             embedding, embedding.mean(dim=0, keepdim=True), dim=1
            #         ),
            #         -1 + 1e-7,
            #         1 - 1e-7,
            #     )
            # ).mean()
            #! 使用cdist计算l2是错误的
            # loss_svdd = torch.cdist(
            #     embedding, embedding.mean(dim=0, keepdim=True), p=2
            # ).mean()
            # loss_svdd = (
            #     ((embedding - embedding.mean(dim=0, keepdim=True)) ** 2)
            #     .mean(dim=1)
            #     .sqrt()
            # ).mean()

            loss_svdd = svdd(embedding, cls_id)

            # loss_svdd_noise = svdd(embedding_noisy, cls_id)
            # loss_svdd = (
            #     loss_svdd_norm
            #     # + loss_svdd_noise
            #     # + F.cosine_similarity(embedding, embedding_noisy).mean()
            # )
            # loss_cos = 1 - F.cosine_similarity(embedding, embedding_noisy).mean()
            loss = loss_rd + loss_svdd  # + loss_cos
            # loss = loss_rd
            loss_rd_sum += loss_rd.item()
            # loss_rd_noise_sum += rd_noise.item()
            loss_svdd_sum += loss_svdd.item()
            # loss_svdd_noise_sum += loss_svdd_noise.item()
            # loss_cos_sum += loss_cos.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        history["loss_rd"].append((loss_rd_sum) / len(train_dataloader))
        history["loss_svdd"].append((loss_svdd_sum) / len(train_dataloader))
        history["loss_total"].append(
            (loss_rd_sum + loss_svdd_sum) / len(train_dataloader)
        )

        print(
            "epoch [{}/{}], loss_total:{:.4f}, loss_rd:{:.4f}, loss_svdd:{:.4f}".format(
                epoch + 1,
                args.epochs,
                history["loss_total"][-1],
                history["loss_rd"][-1],
                # loss_rd_noise_sum / len(train_dataloader),
                history["loss_svdd"][-1],
                # loss_svdd_noise_sum / len(train_dataloader),
                # loss_cos_sum / len(train_dataloader),
            )
        )

        if (epoch + 1) % 5 == 0:
            auroc_px, auroc_sp, aupro_px = evaluation(
                encoder, bn, decoder, test_dataloader, device, train_data.classes
            )
            history["auroc_sp"].append(auroc_sp["mean"])
            history["auroc_px"].append(auroc_px["mean"])
            history["aupro_px"].append(aupro_px["mean"])

            print(
                f"Sample Auroc{auroc_sp}\nPixel Auroc:{auroc_px}\nPixel Aupro{aupro_px}"
            )

            if (auroc_sp["mean"] + auroc_px["mean"] + aupro_px["mean"]) / 3 > best[
                "best_score"
            ]:
                best["best_epoch"] = epoch + 1
                best["best_score"] = (
                    auroc_sp["mean"] + auroc_px["mean"] + aupro_px["mean"]
                ) / 3
                best["best_auroc_sp"] = auroc_sp
                best["best_auroc_px"] = auroc_px
                best["best_aupro_px"] = aupro_px
                torch.save(
                    {
                        "bn": bn.state_dict(),
                        "decoder": decoder.state_dict(),
                        "svdd": svdd.state_dict(),
                    },
                    os.path.join(
                        args.save_path,
                        "best_model.pth",
                    ),
                )
    metrics = {"class": [], "AUROC_sample": [], "AUROC_pixel": [], "AUPRO_pixel": []}

    for key in best["best_auroc_sp"]:
        metrics["class"].append(key)
        metrics["AUROC_sample"].append(best["best_auroc_sp"][key])
        metrics["AUROC_pixel"].append(best["best_auroc_px"][key])
        metrics["AUPRO_pixel"].append(best["best_aupro_px"][key])

    pd.DataFrame(metrics).to_csv(f"{args.save_path}/best_results.csv", index=False)

    print(
        "Epoch {} is the Best !\nSample Auroc: {:.4f}, Pixel Auroc: {:.4f}, Pixel Aupro: {:.4f}".format(
            best["best_epoch"],
            best["best_auroc_sp"]["mean"],
            best["best_auroc_px"]["mean"],
            best["best_aupro_px"]["mean"],
        )
    )

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 12)
    fig, ax = plt.subplots(3, 2, figsize=(8, 12))
    ax[0][0].plot(range(1, len(history["auroc_sp"]) + 1, 1), history["auroc_sp"])
    ax[0][0].set_title("auroc_sp")
    ax[0][1].plot(range(1, len(history["auroc_px"]) + 1, 1), history["auroc_px"])
    ax[0][1].set_title("auroc_px")
    ax[1][0].plot(range(1, len(history["aupro_px"]) + 1, 1), history["aupro_px"])
    ax[1][0].set_title("aupro_px")
    ax[1][1].plot(range(1, len(history["loss_rd"]) + 1, 1), history["loss_rd"])
    ax[1][1].set_title("loss_rd")
    ax[2][0].plot(range(1, len(history["loss_svdd"]) + 1, 1), history["loss_svdd"])
    ax[2][0].set_title("loss_svdd")
    ax[2][1].plot(range(1, len(history["loss_total"]) + 1, 1), history["loss_total"])
    ax[2][1].set_title("loss_total")
    plt.savefig(f"{args.save_path}/monitor_traning.png", dpi=100)


if __name__ == "__main__":
    # setup_seed(args.seed)
    train()
