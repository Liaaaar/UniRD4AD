import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


# 线性分类器
class FC(nn.Module):
    # 当使用resnet系列的特征提取器时, 输入的维度为(N,2048,7,7)
    def __init__(self):
        super(FC, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 15)

    def forward(self, x):
        x_pooled = self.pool(x).view(x.shape[0], -1)
        x = self.fc(x_pooled)
        return x, x_pooled


# SVDD损失
class SVDD_loss(nn.Module):
    def __init__(self):
        super(SVDD_loss, self).__init__()
        # self.discriminator = FC()
        # self.center = nn.Parameter(torch.randn(1, 2048), requires_grad=True)
        # self.cos_center = nn.Parameter(torch.randn(1, 128, 16, 16), requires_grad=True)
        self.center = nn.Parameter(torch.randn(1, 2048, 8, 8), requires_grad=True)
        # self.len_center = nn.Parameter(torch.randn(1, 2048, 8, 8), requires_grad=True)
        #! self.center = nn.Parameter(torch.randn(1, 128, 8, 8), requires_grad=False)
        # self.center = nn.Parameter(torch.zeros(1, 2048), requires_grad=True)
        # self.classify_loss = nn.CrossEntropyLoss()

    def forward(self, embedding, cls_id):
        # predict_id, embedding_pooled = self.discriminator(embedding)
        # classify_loss = self.classify_loss(predict_id, cls_id)
        # cluster_loss = torch.cdist(embedding_pooled, self.center, p=1).mean()
        # embedding = F.normalize(embedding, dim=1)
        # center = F.normalize(self.center, dim=1)
        # cluster_loss = ((embedding - self.center) ** 2).mean()
        # cluster_loss = torch.cdist(embedding, self.center, p=2).mean()
        cluster_loss = (1 - F.cosine_similarity(embedding, self.center, dim=1)).mean()
        # cluster_loss_len = torch.nn.MSELoss()(embedding, self.len_center)

        # cluster_loss = (
        #     F.kl_div(
        #         embedding.view(embedding.shape[0], -1).log_softmax(dim=-1),
        #         self.center.view(self.center.shape[0], -1).softmax(dim=-1),
        #         reduction="batchmean",
        #     )
        #     + F.kl_div(
        #         self.center.view(self.center.shape[0], -1).log_softmax(dim=-1),
        #         embedding.view(embedding.shape[0], -1).softmax(dim=-1),
        #         reduction="batchmean",
        #     )
        # ) / 2
        # embedding = embedding.view(embedding.shape[0], -1)
        # cluster_loss = torch.acos(
        #     torch.clip(
        #         F.cosine_similarity(embedding, self.center, dim=1),
        #         -1 + 1e-7,
        #         1 - 1e-7,
        #     )
        # ).mean()
        # return 1.0 * classify_loss + 1.0 * cluster_loss
        return cluster_loss


# 蒸馏损失
class RD_loss(nn.Module):
    def __init__(self):
        super(RD_loss, self).__init__()
        self.cos_loss = torch.nn.CosineSimilarity()

    def forward(self, a, b):
        loss = 0
        for item in range(len(a)):
            # print(
            #     1
            #     - self.cos_loss(
            #         a[item].view(a[item].shape[0], -1),
            #         b[item].view(b[item].shape[0], -1),
            #     )
            # )
            loss += torch.mean(
                1
                - self.cos_loss(
                    a[item].view(a[item].shape[0], -1),
                    b[item].view(b[item].shape[0], -1),
                )
            )
            # loss += (1 - self.cos_loss(a[item], b[item])).mean()
        return loss


# Arcloss
class Arc_loss(nn.Module):
    def __init__(self):
        super(Arc_loss, self).__init__()
        self.cos_loss = torch.nn.CosineSimilarity()

    def forward(self, a, b):

        loss = 0
        for item in range(len(a)):
            loss += torch.mean(
                torch.acos(
                    torch.clip(
                        self.cos_loss(
                            a[item].view(a[item].shape[0], -1),
                            b[item].view(b[item].shape[0], -1),
                        ),
                        -1 + 1e-7,
                        1 - 1e-7,
                    )
                )
            )
            # loss += (1 - self.cos_loss(a[item], b[item])).mean()
        return loss


# SSIM_loss
def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


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

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class SSIM_loss(nn.Module):
    def __init__(self):
        super(SSIM_loss, self).__init__()
        self.ssim = SSIM()

    def forward(self, a, b):
        loss = 0
        for item in range(len(a)):
            # print(
            #     1
            #     - self.ssim(
            #         a[item],
            #         b[item],
            #     )
            # )
            loss += 1 - self.ssim(
                a[item],
                b[item],
            )
        return loss
