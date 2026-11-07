import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Function

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def grad_reverse(x, lambda_=1.0):
    return GradientReversal.apply(x, lambda_)

class DisentangleNet(nn.Module):
    def __init__(self, num_classes=7, identity_dim=512, baseline_mode=False,
                 use_grl=True, expr_channels=288, use_identity=True):
        super().__init__()
        self.baseline_mode = baseline_mode
        self.use_grl = use_grl
        self.use_identity = use_identity

        # 骨干网络：必须使用 pretrained=True 才能复现 84% 的精度
        backbone = models.mobilenet_v3_small(pretrained=True)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.feat_channels = 576

        if not baseline_mode:
            self.expr_channels = expr_channels
            self.ident_channels = self.feat_channels - expr_channels

            self.expr_classifier = nn.Sequential(
                nn.Linear(self.expr_channels, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

            if use_identity:
                self.ident_projector = nn.Sequential(
                    nn.Linear(self.ident_channels, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, identity_dim)
                )
            else:
                self.ident_projector = None
        else:
            self.classifier = nn.Sequential(
                nn.Linear(576, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

    def forward(self, x, lambda_grl=0.1):
        feat_map = self.features(x)          # (B, 576, 7, 7) for 224x224 input
        x = self.avgpool(feat_map)           # (B, 576, 1, 1)
        x = torch.flatten(x, 1)              # (B, 576)

        if self.baseline_mode:
            return self.classifier(x)

        # 静态通道拆分 (Static Channel Split)
        expr_feat = x[:, :self.expr_channels]  # 前 288 维
        
        if not self.use_identity:
            expr_out = self.expr_classifier(expr_feat)
            return expr_out, None, expr_feat, None

        ident_feat = x[:, self.expr_channels:self.expr_channels + self.ident_channels] # 后 288 维

        if self.use_grl:
            ident_feat_rev = grad_reverse(ident_feat, lambda_=lambda_grl)
            ident_out = self.ident_projector(ident_feat_rev)
        else:
            ident_out = self.ident_projector(ident_feat)

        expr_out = self.expr_classifier(expr_feat)

        return expr_out, ident_out, expr_feat, ident_feat
