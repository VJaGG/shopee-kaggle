#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import sys
sys.path.append("/home/data_normal/abiz/wuzhiqiang/wzq/shopee/code_v3/pytorch-image-models")
import timm
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from sklearn.preprocessing import LabelEncoder
import torch.utils.data as data
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModel
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


bert_model_arch = "bert-base-multilingual-uncased"
root = "/home/data_normal/abiz/wuzhiqiang/wzq/data/shopee-product-matching"
test_path = "/home/data_normal/abiz/wuzhiqiang/wzq/data/shopee-product-matching/train_images"
image_initial_checkpoint = "/home/data_normal/abiz/wuzhiqiang/wzq/shopee/code_v3/result/image/00062000_model.pth"
text_initial_checkpoint = "/home/data_normal/abiz/wuzhiqiang/wzq/shopee/code_v3/result/text/00052000_model.pth"
# df = pd.read_csv(root)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# df.head()


image_threshold = 50
text_threshold = 70
batch_size = 4
width, height = 640, 640
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
null_augment = A.Compose(
    [A.Resize(width, height)]
)


def collate_fn(batch):
    batch_size = len(batch)
    image = []
    index = []
    input_ids = []
    token_type_ids = []
    attention_mask = []
    for r in batch:
        image.append((r['image'] / 255.0 - mean) / std)
        index.append(r['index'])
        input_ids.append(r['items']['input_ids'])
        token_type_ids.append(r['items']['token_type_ids'])
        attention_mask.append(r['items']['attention_mask'])
    
    image = np.stack(image)
    input_ids = torch.stack(input_ids)
    token_type_ids = torch.stack(token_type_ids)
    attention_mask = torch.stack(attention_mask)

    image = image.transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).contiguous().float()

    return {
        'index': index,
        'image': image,
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
    }


class ShopeeDataset(data.Dataset):
    def __init__(self, df, augment=null_augment, bert_model_arch=bert_model_arch):
        self.df = df
        self.augment = augment
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_arch)
        texts = df['title'].fillna("NaN").tolist()  # 所有标题生成的文本列表
        self.encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
        )

    def __getitem__(self, index):
        posting_info = self.df.iloc[index]
        title = posting_info.title
        image_name = posting_info.image
        image_path = os.path.join(test_path, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)

        items = {k: torch.tensor(v[index]) for k, v in self.encodings.items()}
        
        if self.augment:
            augmented = self.augment(image=image)
            image = augmented['image']

        r = {
            'index': index,
            'image': image,
            'items': items
        }
        return r

    def __len__(self):
        return self.df.shape[0]

    def __str__(self):
        string = ''
        string += '\tlen     = %d\n' % len(self)
        return string


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


class NFNetMargin(nn.Module):
    def __init__(self, arch="eca_nfnet_l1", dim=1792, num_classes=11014, pretrained=False):
        super(NFNetMargin, self).__init__()

        self.backbone = timm.create_model(arch, pretrained=pretrained)
        final_in_features = self.backbone.head.fc.in_features
        self.backbone.head.global_pool = nn.Identity()
        self.backbone.head.fc = nn.Identity()
        self.conv = nn.Conv2d(final_in_features, dim, kernel_size=3, stride=1)
        self.silu = nn.SiLU()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(dim)
        
        self.margin = ArcMarginProductMargin(in_features=dim,
                                             out_features=num_classes)
        self.__init_params()
    
    def __init_params(self):
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, labels=None, margin=0.5):
        feature = self.backbone(x)
        feature = self.conv(feature)
        feature = self.silu(feature)
        feature = self.pooling(feature).view(x.size(0), -1)
        feature = self.bn(feature)  # (4, 2048)
        if labels is not None:
            return self.margin(feature, labels, margin)
        return feature



class BERTMargin(nn.Module):
        
    def __init__(self, arch='bert-base-multilingual-uncased',
                    hidden_size=768, dim=1024, num_classes=11014):
        
        super(BERTMargin, self).__init__()
#         path = "../input/huggingface-bert/bert-base-multilingual-uncased"
        config = AutoConfig.from_pretrained(arch,
                                            output_hidden_states=True)
        self.bert_model = AutoModel.from_pretrained(
            arch,
            cache_dir=None,
            config=config,
        )
        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(
            hidden_size, 
            dim
        )
        self.bn = nn.BatchNorm1d(dim)
        self.margin = ArcMarginProductMargin(in_features=dim,
                                            out_features=num_classes)
        self._init_params()


    def _init_params(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, input_ids, attentions_mask, labels=None, margin=0.8):
        # input_ids: (None, 128)
        # attentions_mask: (None, 128)
        output = self.bert_model(input_ids=input_ids, attention_mask=attentions_mask)
        hs = output.hidden_states
        hs_idxs = [-1, -2, -3, -4]
        seq_output = torch.stack([hs[idx] for idx in hs_idxs]).mean(dim=0)
        avg_output = torch.sum(
            seq_output * attentions_mask.unsqueeze(-1),
            dim=1,
            keepdim=False
        )
        avg_output = avg_output / torch.sum(attentions_mask, dim=-1, keepdim=True)
        x = avg_output

        out = self.fc(x)
        out = self.bn(out)
        if labels is not None:
            return self.margin(out, labels, margin)
        return out, x


def extract_image_feat(net, valid_loader):
    features = []
    with torch.no_grad():
        net.eval()
        valid_num = 0
        for t, batch in enumerate(valid_loader):
            index = batch['index']
            image = batch['image'].cuda()
            feat = net(image)
            features += [feat.detach().cpu()]
            valid_num += len(index)
        assert (valid_num == len(valid_loader.dataset))
    features = torch.cat(features).cpu().numpy()
    return features



def extract_text_feat(net, valid_loader):
    features = []
    with torch.no_grad():
        net.eval()
        valid_num = 0
        for t, batch in enumerate(valid_loader):
            index = batch['index']
            input_ids = batch['input_ids'].cuda()
            token_type_ids = batch['token_type_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            feat, _ = net(input_ids, attention_mask)
            features += [feat.detach().cpu()]
            valid_num += len(index)
        assert (valid_num == len(valid_loader.dataset))
    features = torch.cat(features).cpu().numpy()
    return features


def find_matches(posting_ids, threshold, features, n_batches, min_indices=1):
    assert len(posting_ids) == len(features)
    sim_threshold = threshold / 100
    n_rows = features.shape[0]
    bs = n_rows // n_batches
    if bs <= 0:
        n_batches = 1
    batches = []
    distances = []
    indices = []
    for i in range(n_batches):
        left = bs * i
        right = bs * (i + 1)
        if i == n_batches - 1:
            right = n_rows
        batches.append(features[left: right, :])
    for batch in batches:
        dot_product = batch @ features.T
        selection = dot_product > sim_threshold
        for j in range(len(selection)):
            # top_inds = selection[j]  # 阈值之内的相似图片
            top_inds = torch.where(selection[j])[0]
            top_dist = dot_product[j][top_inds]
            # if torch.sum(top_inds) < min_indices:  # 如果匹配的图片小于最小的个数
            #     top_inds = torch.argsort(dot_product[j])[-min_indices:]
            distances.append(top_dist)
            indices.append(top_inds)
    return distances, indices


def get_nearest(features, n_batches, K=None, sorted=True):
    if K is None:
        K = min(51, len(features))
    n_rows = features.shape[0]
    bs = n_rows // n_batches
    if bs <= 0:
        n_batches = 1
    batches = []
    distances = []
    indices = []
    for i in range(n_batches):
        left = bs * i
        right = bs * (i + 1)
        if i == n_batches - 1:
            right = n_rows
        batches.append(features[left: right, :])
    for batch in batches:
        dot_product = batch @ features.T
        top_vals, top_inds = dot_product.topk(K, dim=0, sorted=True)  # 返回topk的检索结果
        distances.append(top_vals.T)
        indices.append(top_inds.T)
    return torch.cat(distances), torch.cat(indices)

def combined_distances(image_features, text_features, image_indices, text_indices, threshold=0.7):

    res_inds, res_dists = [], []
    for x, (img_indice, text_indice) in enumerate(zip(image_indices, text_indices)):
        inds = torch.cat([img_indice, text_indice]).unique()
        Ds = [image_features[None, x] @ image_features[inds].T, text_features[None, x] @ text_features[inds].T]
        D = Ds[0] + Ds[1] - Ds[0] * Ds[1]
        # top_dists, top_inds = D.topk(K)
        top_inds = torch.where(D[0] > threshold)
        res_inds.append(inds[top_inds].cpu().numpy())
    return res_inds

fold = 0
df = pd.read_csv('folds.csv')
df['image_path'] = df['image'].apply(lambda x: os.path.join(root, 'train_images', x))

encoder = LabelEncoder()
df_valid = df[df['fold']==fold].reset_index(drop=True)[0: 100]
df_valid['label_group'] = encoder.fit_transform(df_valid['label_group'])
test_dataset = ShopeeDataset(df_valid)
test_dataloader = data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=batch_size,
    collate_fn=collate_fn
)


imgnet = NFNetMargin(num_classes=8811)
txtnet = BERTMargin(num_classes=8811)
imgnet.to(device)
txtnet.to(device)

state_dict = torch.load(image_initial_checkpoint, map_location=lambda storage, loc: storage)['state_dict']
imgnet.load_state_dict(state_dict,strict=False)
imgnet.eval()
del state_dict

state_dict = torch.load(text_initial_checkpoint, map_location=lambda storage, loc: storage)['state_dict']
txtnet.load_state_dict(state_dict,strict=False)
txtnet.eval()
del state_dict


image_features = extract_image_feat(imgnet, test_dataloader)
text_features = extract_text_feat(txtnet, test_dataloader)


image_features = F.normalize(torch.from_numpy(image_features))
text_features = F.normalize(torch.from_numpy(text_features))

# get_nearest(image_features, n_batches=10)
# res_inds, res_dists = combined_distances(image_features, text_features)
# text_features = F.normalize(torch.from_numpy(text_features)).numpy()
# concat_feature = F.normalize(torch.from_numpy(concat_feature)).numpy()

posting_ids = df['posting_id']
image_distances, image_indices = find_matches(posting_ids, image_threshold, image_features, n_batches=10)
text_distances, text_indices = find_matches(posting_ids, text_threshold, image_features, n_batches=10)
print(image_indices)
print(text_indices)
y_pred = combined_distances(image_features, text_features, image_indices, text_indices)
print(y_pred)
# y_text_pred =  find_matches(posting_ids, text_threshold, text_features, n_batches=10)
# concat_pred =  find_matches(posting_ids, 0.5, concat_feature, n_batches=10)

# y_pred = [list(set(i + j + k)) for i, j, k in zip(y_image_pred, y_text_pred, concat_pred)]

# df['matches'] = y_pred
# df['matches'] = df['matches'].apply(lambda x: " ".join(x))

# df[['posting_id','matches']].to_csv('submission.csv',index=False)
# pd.read_csv('submission.csv').head()

