import random
import warnings
from statistics import mean

import numpy as np
import torch
from numpy import ndarray
from scipy.ndimage import gaussian_filter
from sklearn.metrics import auc, roc_auc_score
from skimage import measure
from torch.nn import functional as F

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cal_anomaly_map(fs_list, ft_list, out_size, amap_mode="add"):
    anomaly_map = np.ones([out_size, out_size]) if amap_mode == "mul" else np.zeros(
        [out_size, out_size]
    )
    a_map_list = []

    for fs, ft in zip(fs_list, ft_list):
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = a_map.unsqueeze(1)
        a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)
        a_map = a_map[0, 0].detach().cpu().numpy()
        a_map_list.append(a_map)

        if amap_mode == "mul":
            anomaly_map *= a_map
        else:
            anomaly_map += a_map

    return anomaly_map, a_map_list


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200):
    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"

    records = {"pro": [], "fpr": [], "threshold": []}
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        records["pro"].append(mean(pros))
        records["fpr"].append(fpr)
        records["threshold"].append(th)

    fpr = np.asarray(records["fpr"])
    pro = np.asarray(records["pro"])
    valid = fpr < 0.3
    fpr = fpr[valid]
    pro = pro[valid]
    if len(fpr) == 0 or fpr.max() == 0:
        return 0.0
    fpr = fpr / fpr.max()
    return auc(fpr, pro)


def evaluation(
    encoder,
    bn,
    decoder,
    dataloader,
    device,
    classes,
    amap_mode="add",
    show_progress=False,
    progress_desc="Inference",
):
    encoder.eval()
    bn.eval()
    decoder.eval()

    gt_list_px = {cls: [] for cls in classes}
    pr_list_px = {cls: [] for cls in classes}
    gt_list_sp = {cls: [] for cls in classes}
    pr_list_sp = {cls: [] for cls in classes}
    aupro_list = {cls: [] for cls in classes}

    progress_loader = dataloader
    if show_progress and tqdm is not None:
        progress_loader = tqdm(dataloader, total=len(dataloader), desc=progress_desc)

    with torch.no_grad():
        for img, gt, cls, cls_id in progress_loader:
            del cls_id
            cls = cls[0]
            img = img.to(device)

            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(
                inputs, outputs, img.shape[-1], amap_mode=amap_mode
            )
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            gt = gt.clone()
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            if gt.max() != 0:
                aupro_list[cls].append(
                    compute_pro(
                        gt.squeeze(0).cpu().numpy().astype(int),
                        anomaly_map[np.newaxis, :, :],
                    )
                )

            gt_np = gt.cpu().numpy().astype(int)
            gt_list_px[cls].extend(gt_np.ravel())
            pr_list_px[cls].extend(anomaly_map.ravel())
            gt_list_sp[cls].append(np.max(gt_np))
            pr_list_sp[cls].append(np.max(anomaly_map))

    auroc_px = {}
    auroc_sp = {}
    for cls in classes:
        auroc_px[cls] = float(round(roc_auc_score(gt_list_px[cls], pr_list_px[cls]), 3))
        auroc_sp[cls] = float(round(roc_auc_score(gt_list_sp[cls], pr_list_sp[cls]), 3))
        aupro_list[cls] = float(round(np.mean(aupro_list[cls]), 3))

    auroc_px["mean"] = float(round(np.mean(list(auroc_px.values())), 3))
    auroc_sp["mean"] = float(round(np.mean(list(auroc_sp.values())), 3))
    aupro_list["mean"] = float(round(np.mean(list(aupro_list.values())), 3))
    return auroc_px, auroc_sp, aupro_list
