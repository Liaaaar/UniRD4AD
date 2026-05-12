import csv
import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MVTEC_CLASSES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

VISA_CLASSES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

REALIAD_CLASSES = [
    "audiojack",
    "bottle_cap",
    "button_battery",
    "end_cap",
    "eraser",
    "fire_hood",
    "mint",
    "mounts",
    "pcb",
    "phone_battery",
    "plastic_nut",
    "plastic_plug",
    "porcelain_doll",
    "regulator",
    "rolled_strip_base",
    "sim_card_set",
    "switch",
    "tape",
    "terminalblock",
    "toothbrush",
    "toy",
    "toy_brick",
    "transistor1",
    "usb",
    "usb_adaptor",
    "u_block",
    "vcpill",
    "wooden_beads",
    "woodstick",
    "zipper",
]


def get_transforms(size):
    img_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((size, size), antialias=False),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    mask_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((size, size), antialias=False),
        ]
    )
    return img_transform, mask_transform


def resolve_path(root, path):
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(root, path.lstrip("/\\"))


class Uni_MVTecDataset(Dataset):
    def __init__(self, img_size, root, mode):
        self.img_size = img_size
        self.root = root
        self.mode = mode
        self.img_transform, self.mask_transform = get_transforms(img_size)
        self.classes = sorted(
            [
                name
                for name in os.listdir(root)
                if os.path.isdir(os.path.join(root, name))
            ]
        )
        self.class_map = {name: idx for idx, name in enumerate(self.classes)}

        if mode == "train":
            self.imgs_path, self.imgs_class, self.imgs_class_id = self.get_train_data()
        else:
            (
                self.imgs_path,
                self.masks_path,
                self.imgs_class,
                self.imgs_class_id,
            ) = self.get_test_data()

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        cls = self.imgs_class[idx]
        cls_id = self.imgs_class_id[idx]
        img = self.img_transform(Image.open(img_path).convert("RGB"))

        if self.mode == "train":
            return img, cls, cls_id

        mask_path = self.masks_path[idx]
        if mask_path is None:
            mask = torch.zeros([1, img.size(-2), img.size(-1)])
        else:
            mask = self.mask_transform(Image.open(mask_path).convert("L"))
        return img, mask, cls, cls_id

    def get_train_data(self):
        imgs_path = []
        imgs_class = []
        for cls in self.classes:
            cls_imgs = glob.glob(os.path.join(self.root, cls, "train", "good", "*.png"))
            imgs_path.extend(cls_imgs)
            imgs_class.extend([cls] * len(cls_imgs))
        imgs_class_id = [self.class_map[cls] for cls in imgs_class]
        return imgs_path, imgs_class, imgs_class_id

    def get_test_data(self):
        imgs_path = []
        masks_path = []
        imgs_class = []
        for cls in self.classes:
            defect_types = sorted(os.listdir(os.path.join(self.root, cls, "test")))
            for defect_type in defect_types:
                cls_imgs = sorted(
                    glob.glob(
                        os.path.join(self.root, cls, "test", defect_type, "*.png")
                    )
                )
                imgs_path.extend(cls_imgs)
                imgs_class.extend([cls] * len(cls_imgs))

                if defect_type == "good":
                    masks_path.extend([None] * len(cls_imgs))
                    continue

                cls_masks = sorted(
                    glob.glob(
                        os.path.join(
                            self.root, cls, "ground_truth", defect_type, "*.png"
                        )
                    )
                )
                masks_path.extend(cls_masks)

        imgs_class_id = [self.class_map[cls] for cls in imgs_class]
        return imgs_path, masks_path, imgs_class, imgs_class_id


class VisaDataset(Dataset):
    def __init__(self, img_size, root, mode):
        self.img_size = img_size
        self.root = root
        self.mode = mode
        self.classes = VISA_CLASSES
        self.class_map = {name: idx for idx, name in enumerate(self.classes)}
        self.img_transform, self.mask_transform = get_transforms(img_size)
        self.data = self.load_split()

    def load_split(self):
        csv_path = os.path.join(self.root, "split_csv", "1cls.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"ViSA split file not found: {csv_path}")

        data = []
        with open(csv_path, "r", newline="") as handle:
            reader = csv.reader(handle, delimiter=",")
            next(reader, None)
            for row in reader:
                if len(row) < 5 or row[1] != self.mode:
                    continue
                data.append(
                    {
                        "object": row[0],
                        "split": row[1],
                        "label": row[2],
                        "image": row[3],
                        "mask": row[4],
                    }
                )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        cls = item["object"]
        cls_id = self.class_map[cls]

        img_path = resolve_path(self.root, item["image"])
        img = self.img_transform(Image.open(img_path).convert("RGB"))

        if self.mode == "train":
            return img, cls, cls_id

        mask_path = resolve_path(self.root, item["mask"])
        if mask_path is not None and os.path.exists(mask_path):
            mask = self.mask_transform(Image.open(mask_path).convert("L"))
        elif item["label"] == "normal":
            mask = torch.zeros([1, img.size(-2), img.size(-1)])
        elif item["label"] == "anomaly":
            mask = torch.from_numpy(
                np.ones((1, img.size(-2), img.size(-1)), dtype=np.float32)
            )
        else:
            raise ValueError(f"Unsupported label: {item['label']}")

        return img, mask, cls, cls_id


class RealIADDataset(Dataset):
    def __init__(self, img_size, root, mode):
        self.img_size = img_size
        self.root = root
        self.mode = mode
        self.classes = REALIAD_CLASSES
        self.class_map = {name: idx for idx, name in enumerate(self.classes)}
        self.img_transform, self.mask_transform = get_transforms(img_size)
        self.data = self.load_split()

    def load_split(self):
        data = []
        for cls in self.classes:
            json_path = os.path.join(
                self.root, "realiad_jsons", "realiad_jsons", f"{cls}.json"
            )
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"Real-IAD split file not found: {json_path}")

            with open(json_path, "r") as handle:
                class_json = json.load(handle)

            if self.mode not in class_json:
                raise KeyError(f"Split '{self.mode}' not found in {json_path}")

            for sample in class_json[self.mode]:
                anomaly_class = sample["anomaly_class"]
                label = anomaly_class != "OK"
                data.append(
                    {
                        "class": cls,
                        "image": os.path.join(
                            self.root,
                            "realiad_1024",
                            cls,
                            sample["image_path"],
                        ),
                        "mask": os.path.join(
                            self.root,
                            "realiad_1024",
                            cls,
                            sample["mask_path"],
                        )
                        if label
                        else None,
                        "label": label,
                        "type": anomaly_class,
                    }
                )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        cls = item["class"]
        cls_id = self.class_map[cls]
        img = self.img_transform(Image.open(item["image"]).convert("RGB"))

        if self.mode == "train":
            return img, cls, cls_id

        if item["label"]:
            mask = self.mask_transform(Image.open(item["mask"]).convert("L"))
        else:
            mask = torch.zeros([1, img.size(-2), img.size(-1)])

        return img, mask, cls, cls_id


def normalize_dataset_name(dataset_name):
    return dataset_name.lower()


def build_dataset(dataset_name, img_size, root, mode):
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == "mvtec":
        return Uni_MVTecDataset(img_size, root, mode)
    if dataset_name == "visa":
        return VisaDataset(img_size, root, mode)
    if dataset_name == "realiad":
        return RealIADDataset(img_size, root, mode)
    raise ValueError(f"Unsupported dataset: {dataset_name}")
