import torch
from torch import nn
import torch.nn.functional as F

try:
    import geomloss
except ImportError:
    geomloss = None


def gaussian(window_size, sigma, device=None, dtype=None):
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords = coords - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    return gauss / gauss.sum()


def create_window(window_size, channel, device=None, dtype=None):
    window_1d = gaussian(window_size, 1.5, device=device, dtype=dtype).unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window_2d = window_2d.float().unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        channel = img1.size(1)
        if (
            channel != self.channel
            or self.window.device != img1.device
            or self.window.dtype != img1.dtype
        ):
            self.window = create_window(
                self.window_size,
                channel,
                device=img1.device,
                dtype=img1.dtype,
            )
            self.channel = channel
        return _ssim(
            img1,
            img2,
            self.window,
            self.window_size,
            channel,
            self.size_average,
        )


class RDLoss(nn.Module):
    def __init__(self, mode="cosine"):
        super().__init__()
        self.mode = mode.lower()
        if self.mode not in {"cosine", "arc", "ssim", "mse", "mae"}:
            raise ValueError(f"Unsupported rd_loss mode: {self.mode}")
        self.ssim = SSIM() if self.mode == "ssim" else None

    def _pair_loss(self, teacher, student):
        if self.mode == "cosine":
            teacher = teacher.flatten(1)
            student = student.flatten(1)
            return (1 - F.cosine_similarity(teacher, student, dim=1)).mean()

        if self.mode == "arc":
            teacher = teacher.flatten(1)
            student = student.flatten(1)
            cosine = F.cosine_similarity(teacher, student, dim=1)
            cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
            return torch.acos(cosine).mean()

        if self.mode == "ssim":
            return 1 - self.ssim(teacher, student)

        if self.mode == "mse":
            return F.mse_loss(teacher, student)

        if self.mode == "mae":
            return F.l1_loss(teacher, student)

        raise ValueError(f"Unsupported rd_loss mode: {self.mode}")

    def forward(self, teacher_features, student_features):
        loss = 0.0
        for teacher, student in zip(teacher_features, student_features):
            loss += self._pair_loss(teacher, student)
        return loss


class ClusterLoss(nn.Module):
    DEFAULT_SCALES = {
        "cosine": 1.0,
        "arc": 1.0,
        "ssim": 1.0,
        "mse": 1.0,
        "mae": 1.0,
        "ssot": 0.01,
    }

    def __init__(
        self,
        channels=2048,
        spatial_size=8,
        mode="cosine",
        scale=None,
    ):
        super().__init__()
        self.mode = mode.lower()
        if self.mode not in self.DEFAULT_SCALES:
            raise ValueError(f"Unsupported cluster_loss mode: {self.mode}")
        self.channels = channels
        self.spatial_size = spatial_size
        self.scale = self.DEFAULT_SCALES[self.mode] if scale is None else scale

        if self.mode == "ssot":
            if geomloss is None:
                raise ImportError(
                    "geomloss is required when cluster_loss is set to 'ssot'."
                )
            self.ot_loss = geomloss.SamplesLoss()
            center_shape = (1, channels * spatial_size * spatial_size)
        else:
            self.ot_loss = None
            center_shape = (1, channels, spatial_size, spatial_size)

        self.center = nn.Parameter(torch.randn(*center_shape), requires_grad=True)
        self.ssim = SSIM() if self.mode == "ssim" else None

    def forward(self, embedding):
        if self.mode == "ssot":
            return self.scale * self.ot_loss(embedding.flatten(1), self.center)

        center = self.center.expand_as(embedding)

        if self.mode == "cosine":
            return (
                self.scale * (1 - F.cosine_similarity(embedding, center, dim=1)).mean()
            )

        if self.mode == "arc":
            cosine = F.cosine_similarity(embedding, center, dim=1)
            cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
            return self.scale * torch.acos(cosine).mean()

        if self.mode == "ssim":
            return self.scale * (1 - self.ssim(embedding, center))

        if self.mode == "mse":
            return self.scale * F.mse_loss(embedding, center)

        if self.mode == "mae":
            return self.scale * F.l1_loss(embedding, center)

        raise ValueError(f"Unsupported cluster_loss mode: {self.mode}")
