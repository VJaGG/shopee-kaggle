import sys
from timeit import main

from numpy.lib.npyio import mafromtxt
from transformers.file_utils import PT_RETURN_INTRODUCTION
sys.path.append("/home/data_normal/abiz/wuzhiqiang/wzq/shopee/code_v3/")
from common import *


class ArcMarginProduct(nn.Module):
    """Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.5,
        easy_margin: bool = False,
        smoothing: float = 0.0,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.smoothing = smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, labels):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight)).float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # ----------------------W----- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device=labels.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        if self.smoothing > 0:
            one_hot = (
                1 - self.smoothing
            ) * one_hot + self.smoothing / self.out_features

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class ArcMarginProductMargin(nn.Module):
    """Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        easy_margin: bool = False,
        smoothing: float = 0.0,
    ):
        super(ArcMarginProductMargin, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.smoothing = smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin

    def forward(self, inputs, labels, margin=0.5):
        cos_m = math.cos(margin)
        sin_m = math.sin(margin)
        th = math.cos(math.pi - margin)
        mm = math.sin(math.pi - margin) * margin

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight)).float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m - sine * sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > th, phi, cosine - mm)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device=labels.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        if self.smoothing > 0:
            one_hot = (
                1 - self.smoothing
            ) * one_hot + self.smoothing / self.out_features

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


def f1_score(y_true, y_pred):  # 计算f1 score
    assert len(y_true) == len(y_pred)
    scores = []
    for i in range(len(y_true)):
        intersect_n = len(np.intersect1d(y_true[i], y_pred[i]))  # 获得预测和target的交集
        score = 2 * intersect_n / (len(y_true[i]) + len(y_pred[i]))  # 计算f1 score
        scores.append(score)
    return scores


def precision_recall(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    precisions = []
    recalls = []
    for i in range(len(y_true)):
        precisions.append(np.mean([t in y_true[i] for t in y_pred[i]]))  # 计算精准率
        recalls.append(np.mean([t in y_pred[i] for t in y_true[i]]))  # 计算召回率
    return precisions, recalls


if __name__ == "__main__":
    arcmargin1 = ArcMarginProduct(in_features=100, out_features=300)
    arcmargin2 = ArcMarginProductMargin(in_features=100, out_features=300)
    x = torch.randn((2, 100))
    label = torch.tensor([10])

    output1 = arcmargin1(x, label)
    output2 = arcmargin2(x, label, margin=0.5)
    print(output1)
    print(output2)