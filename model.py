from common import *


class NFNetMargin(nn.Module):
    def __init__(self, arch="eca_nfnet_l1", dim=1792, num_classes=11014, pretrained=True):
        super(NFNetMargin, self).__init__()
        # print(timm.__version__())
        # print(timm.list_models())
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

    def forward(self, x, labels=None, margin=0.8):
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


if __name__ == "__main__":
    # # net = NFNet()
    # # print(net)
    # # https://github.com/sksq96/pytorch-summary
    # from torchsummary import summary
    # net = NFNet()
    # # net = timm.create_model("efficientnet_b3")
    # net = timm.create_model("eca_nfnet_l1")
    # print(timm.list_models())
    # print(timm.__version__)
    # # x = torch.randn((2, 3, 640, 640))
    # # label = torch.tensor([10])
    # # # print(x.shape)
    # # # print(label)
    # # net(x, label)
    # # feature = net(x)
    # # print(feature.shape) 
    # # summary(net, (3, 640, 640), device='cpu')   

    net = BERTMargin()