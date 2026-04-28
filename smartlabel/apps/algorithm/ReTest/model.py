import torch.nn as nn
import torch.nn.functional as F
from models import cnn
from models.widerResNet import WideResNet_encoder
from models.preact_resnet import PreActResNet18
from models import resnet
from models.head import NoneLinearProjectionHead, MLPHead
from models.newsnet import NewsNet
import torchvision.models.resnet as resnetModels
import torchvision.models.vgg as vggModels
from typing import Any, List, Optional, Tuple, Union
import torch


class Encoder(nn.Module):
    def __init__(self, cfg: dict, num_classes: int = 200):
        super().__init__()
        arch = cfg.get("architecture", "resnet18")
        self.arch = arch
        pretrained = cfg.get("pretrained", False)
        input_channel = cfg.get("input_channel", 3)
        realworld = cfg.get("realworld", False)
        dataset = cfg.get("dataset", "")

        if arch.lower().startswith("resnet"):
            if pretrained:
                assert input_channel == 3
            if (not realworld and arch in resnet.__all__) or (realworld and dataset == "animal10N"):
                Resnet = resnet.__dict__[arch](pretrained=pretrained, in_channels=input_channel)
            elif realworld and dataset != "animal10N" and arch in resnetModels.__all__:
                Resnet = resnetModels.__dict__[arch](pretrained=pretrained)
            else:
                raise ValueError(f"Unknown resnet arch: {arch}")
            self.encoder = nn.Sequential(*list(Resnet.children())[:-1])
            self.feature_dim = Resnet.fc.in_features

        elif arch.lower().startswith("vgg") and arch in vggModels.__all__:
            vggnet = vggModels.__dict__[arch](pretrained=pretrained)
            self.feature_dim = list(vggnet.classifier.children())[-1].in_features
            vggnet.classifier = nn.Sequential(*list(vggnet.classifier.children())[:-1])
            self.encoder = vggnet

        elif arch.lower().startswith("cnn") and arch in cnn.__all__:
            cnn_model = cnn.__dict__[arch](input_channel=input_channel, n_outputs=num_classes)
            self.encoder = nn.Sequential(*list(cnn_model.children())[:-1])
            self.feature_dim = cnn_model.classifier.in_features

        elif arch == "preactresnet":
            assert input_channel == 3 and not pretrained
            preactResnet = PreActResNet18(num_classes=num_classes)
            self.encoder = nn.Sequential(*list(preactResnet.children())[:-1])
            self.feature_dim = preactResnet.linear.in_features

        elif arch == "wideresnet":
            assert input_channel == 3 and not pretrained
            self.encoder = WideResNet_encoder(num_classes=num_classes)
            self.feature_dim = self.encoder.out_feature

        elif arch == "newsnet":
            news = NewsNet(None)
            self.encoder = nn.Sequential(*list(news.children())[:-1])
            self.feature_dim = news.classifier.in_features

        else:
            raise AssertionError(f"{arch} is not supported!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return h.view(h.shape[0], -1)

    def feature_list(self, x: torch.Tensor, _list: List[str] = ["7"]) -> List[torch.Tensor]:
        feature_list: List[torch.Tensor] = []
        for name, module in self.encoder._modules.items():
            if self.arch == "resnet34" and name == "6":
                for subname, submodule in module._modules.items():
                    if subname in ["0", "1", "2"]:
                        x = submodule(x)
                        feature_list.append(x)
            else:
                x = module(x)
            if self.arch != "resnet34" and name in _list:
                feature_list.append(x)
        return feature_list


class Model(nn.Module):
    def __init__(
        self,
        cfg: dict,
        num_classes: int = 200,
        mlp_hidden: int = 2,
        not_ood: bool = False,
        has_projection: bool = False,
        projection_size: int = 128,
        MLP_classifier: bool = True,
        contrastive_head: bool = False,
        non_linear_projection_head: bool = False,
        recon: bool = False,
        classifier_feat_dim: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = Encoder(cfg, num_classes)
        self.has_projection = has_projection
        self.contrastive_head = contrastive_head

        low_dim = cfg.get("low_dim", 128)

        if not not_ood:
            if MLP_classifier:
                self.classifier = MLPHead(self.encoder.feature_dim, mlp_hidden, num_classes + 1, projection_dim=classifier_feat_dim)
            else:
                self.classifier = nn.Linear(self.encoder.feature_dim, num_classes + 1)
        else:
            if MLP_classifier:
                self.classifier = MLPHead(self.encoder.feature_dim, mlp_hidden, num_classes, projection_dim=classifier_feat_dim)
            else:
                self.classifier = nn.Linear(self.encoder.feature_dim, num_classes)

        if self.has_projection:
            self.projection = nn.Sequential(
                nn.Linear(self.encoder.feature_dim, 512, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(512, projection_size, bias=True),
            )

        if self.contrastive_head:
            if not non_linear_projection_head:
                self.contrast_head = nn.Linear(self.encoder.feature_dim, low_dim)
            else:
                self.contrast_head = NoneLinearProjectionHead(
                    dim_in=self.encoder.feature_dim, dim_out=low_dim, dim_hidden=self.encoder.feature_dim
                )

        if recon:
            self.recon = nn.Linear(low_dim, self.encoder.feature_dim)

    def forward(
        self,
        x: torch.Tensor,
        ret_feature: bool = False,
        ret_low_dim_feature: bool = False,
        ret_high_dim_feature: bool = False,
        recon_out: bool = False,
        ret_classifier_feat: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        feature = self.encoder(x)

        if isinstance(self.classifier, MLPHead):
            logits = self.classifier(feature, projection_ret=ret_classifier_feat)
        else:
            logits = self.classifier(feature)

        if not ret_low_dim_feature:
            if self.has_projection:
                proj = self.projection(feature)
            if not ret_feature:
                return (logits, proj) if self.has_projection else logits
            else:
                return (feature, logits, proj) if self.has_projection else (feature, logits)

        else:
            low_dim_feature = self.contrast_head(feature)
            low_dim_feature = F.normalize(low_dim_feature, dim=1, p=2)
            if not ret_high_dim_feature:
                if not recon_out:
                    return low_dim_feature, logits
                else:
                    recon = F.relu(self.recon(low_dim_feature))
                    error = torch.mean((recon - feature) ** 2, dim=1)
                    return low_dim_feature, error, logits
            else:
                return low_dim_feature, feature, logits
