import os
import hashlib
import sys
import warnings
from argparse import ArgumentParser

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

print("infer.py started", flush=True)
print(f"python = {sys.executable}", flush=True)
print(f"cwd = {os.getcwd()}", flush=True)


def exit_with_error(message, exc=None):
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    if exc is not None:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    raise SystemExit(1)


try:
    import torch
except Exception as exc:
    exit_with_error(
        "Failed to import PyTorch. Please run infer.py with the same environment "
        "used for training, or install torch in the selected Python environment.",
        exc,
    )

try:
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
    from utils.utils import evaluation
except Exception as exc:
    exit_with_error("Failed to import project modules.", exc)

BACKBONES = {
    "resnet18": (resnet18, de_resnet18),
    "resnet34": (resnet34, de_resnet34),
    "resnet50": (resnet50, de_resnet50),
    "wide_resnet50_2": (wide_resnet50_2, de_wide_resnet50_2),
    "wide_resnet101_2": (wide_resnet101_2, de_wide_resnet101_2),
}


def print_metrics(title, metrics):
    print(title, flush=True)
    for name, value in metrics.items():
        print(f"{name}: {float(value):.3f}", flush=True)


def print_encoder_status(encoder, pretrained):
    conv1_weight = encoder.conv1.weight.detach().cpu()
    conv1_mean = float(conv1_weight.mean())
    conv1_std = float(conv1_weight.std())
    print(
        f"encoder conv1.weight mean = {conv1_mean:.6f}, std = {conv1_std:.6f}",
        flush=True,
    )
    if not pretrained:
        print("WARNING: encoder is randomly initialized (--no_pretrained).", flush=True)
    elif conv1_std < 0.05:
        print(
            "WARNING: encoder conv1 statistics look close to random initialization. "
            "Please check pretrained weight loading.",
            flush=True,
        )


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="checkpoints/mvtec_best_model.pth", type=str
    )
    parser.add_argument(
        "--dataset",
        default="mvtec",
        type=str.lower,
        choices=["mvtec", "visa", "realiad"],
    )
    parser.add_argument("--data_root", default="D:/documents/datasets/mvtec", type=str)
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
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="Load ImageNet encoder weights with torch.hub.",
    )
    parser.add_argument(
        "--no_pretrained",
        dest="pretrained",
        action="store_false",
        help="Do not load ImageNet encoder weights.",
    )
    parser.set_defaults(pretrained=True)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    args.dataset = normalize_dataset_name(args.dataset)
    if args.backbone not in BACKBONES:
        raise ValueError(f"Unsupported backbone: {args.backbone}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"Dataset root not found: {args.data_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)
    print(f"dataset = {args.dataset}", flush=True)
    print(f"backbone = {args.backbone}", flush=True)
    print(f"checkpoint = {args.checkpoint}", flush=True)
    print(f"checkpoint sha256 = {file_sha256(args.checkpoint)}", flush=True)
    print(f"data_root = {args.data_root}", flush=True)
    print(f"pretrained = {args.pretrained}", flush=True)

    print("loading dataset...", flush=True)
    test_data = build_dataset(args.dataset, args.img_size, args.data_root, mode="test")
    if len(test_data) == 0:
        raise RuntimeError(
            "No test samples were found. Please check --dataset and --data_root."
        )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"test samples = {len(test_data)}", flush=True)

    print("loading checkpoint...", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    encoder_fn, decoder_fn = BACKBONES[args.backbone]

    print("building models...", flush=True)
    encoder, bn = encoder_fn(pretrained=args.pretrained)
    decoder = decoder_fn(pretrained=False)
    print_encoder_status(encoder, args.pretrained)

    print("loading model weights...", flush=True)
    if "bn" not in checkpoint or "decoder" not in checkpoint:
        raise KeyError("Checkpoint must contain 'bn' and 'decoder' state_dicts.")
    bn.load_state_dict(checkpoint["bn"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.to(device).eval()
    bn.to(device).eval()
    decoder.to(device).eval()

    print("running inference...", flush=True)
    auroc_px, auroc_sp, aupro_px = evaluation(
        encoder,
        bn,
        decoder,
        test_loader,
        device,
        test_data.classes,
        show_progress=True,
        progress_desc="Inference",
    )
    print_metrics("Sample Auroc", auroc_sp)
    print_metrics("Pixel Auroc", auroc_px)
    print_metrics("Pixel Aupro", aupro_px)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        exit_with_error("Inference failed.", exc)
