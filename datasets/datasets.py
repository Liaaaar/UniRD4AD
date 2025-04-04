import os
import glob
import torch
from PIL import Image
from torchvision import transforms


# 获取数据的transform
def get_transforms(size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((size, size), antialias=False),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    mask_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((size, size), antialias=False),
        ]
    )
    return img_transforms, mask_transforms


class Uni_MVTecDataset(torch.utils.data.Dataset):
    """
    Unified情形下的MVTec(one model for multi-class)
    root:mvtec的路径
    mode:train/test
    class_map:类别和id的映射
    when mode==train:返回imgs,class,class_id
    when mode==test:返回imgs,masks,class,class_id
    """

    def __init__(self, img_size, root, mode):
        self.img_size = img_size
        self.img_transform, self.mask_transform = get_transforms(self.img_size)
        self.root = root
        self.mode = mode
        self.class_map = {
            "bottle": 0,
            "cable": 1,
            "capsule": 2,
            "carpet": 3,
            "grid": 4,
            "hazelnut": 5,
            "leather": 6,
            "metal_nut": 7,
            "pill": 8,
            "screw": 9,
            "tile": 10,
            "toothbrush": 11,
            "transistor": 12,
            "wood": 13,
            "zipper": 14,
        }
        # 获取所有类别并排序
        self.classes = sorted(
            [
                i
                for i in os.listdir(self.root)
                if os.path.isdir(os.path.join(self.root, i))
            ]
        )
        # 获取训练数据
        if self.mode == "train":
            self.imgs_path, self.imgs_class, self.imgs_class_id = self.get_train_data(
                self.root, self.classes
            )
        # 获取测试数据
        else:
            (
                self.imgs_path,
                self.masks_path,
                self.imgs_class,
                self.imgs_class_id,
            ) = self.get_test_data(self.root, self.classes)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        cls = self.imgs_class[idx]
        cls_id = self.imgs_class_id[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)
        if self.mode == "train":
            return img, cls, cls_id
        else:
            cls = self.imgs_class[idx]
            mask_path = self.masks_path[idx]
            if mask_path == None:
                mask = torch.zeros([1, img.size()[-2], img.size()[-1]])
            else:
                mask = Image.open(mask_path)
                mask = self.mask_transform(mask)
            return img, mask, cls, cls_id

    def get_train_data(self, root, classes):
        imgs_path = []
        imgs_class = []
        for cls in classes:
            imgs = glob.glob(os.path.join(root, cls, "train/good", "*.png"))
            imgs_path.extend(imgs)
            imgs_class.extend([cls] * len(imgs))
        imgs_class_id = [self.class_map[key] for key in imgs_class]
        return imgs_path, imgs_class, imgs_class_id

    def get_test_data(self, root, classes):
        imgs_path = []
        masks_path = []
        imgs_class = []
        for cls in classes:
            types = sorted(os.listdir(os.path.join(root, cls, "test")))
            for type in types:
                if type != "good":
                    imgs = sorted(
                        glob.glob(os.path.join(root, cls, "test", type, "*.png"))
                    )
                    imgs_path.extend(imgs)
                    masks = sorted(
                        glob.glob(
                            os.path.join(root, cls, "ground_truth", type, "*.png")
                        )
                    )
                    masks_path.extend(masks)
                    imgs_class.extend([cls] * len(imgs))
                else:
                    imgs = sorted(
                        glob.glob(os.path.join(root, cls, "test", type, "*.png"))
                    )
                    imgs_path.extend(imgs)
                    masks = [None] * len(
                        os.listdir(os.path.join(root, cls, "test/good"))
                    )
                    masks_path.extend(masks)
                    imgs_class.extend([cls] * len(imgs))
        imgs_class_id = [self.class_map[key] for key in imgs_class]
        return imgs_path, masks_path, imgs_class, imgs_class_id


# dataset = Uni_MVTecDataset(img_size=256, root="/data01/fkw/datasets/mvtec", mode="test")
# print(dataset[0][0].shape)
