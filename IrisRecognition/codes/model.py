import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


# -----------------------------------------------------
# NGICAM (Channel Attention)
# -----------------------------------------------------
class NGICAM(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(NGICAM, self).__init__()
        self.channels = channels
        self.gamma = gamma
        self.b = b

        k = self.get_kernel_size(channels)
        self.kernel_size = k if k % 2 else k + 1

        self.conv1d = nn.Conv1d(
            1, 1,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1) // 2,
            bias=False
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def get_kernel_size(self, channels):
        k = int(abs((math.log2(channels) / self.gamma) + (self.b / self.gamma)))
        k = k if k % 2 else k + 1
        return max(3, k)

    def forward(self, x):
        b, c, h, w = x.size()

        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(1, 2)
        y = self.conv1d(y)
        y = y.transpose(1, 2)
        y = self.sigmoid(y).unsqueeze(-1)
        return x * y.expand_as(x)


# -----------------------------------------------------
# LSAM (Spatial Attention)
# -----------------------------------------------------
class LSAM(nn.Module):
    def __init__(self):
        super(LSAM, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)

        y = torch.cat([max_out, avg_out], dim=1)
        y = self.conv1(y)
        y = self.conv2(y)

        y = self.sigmoid(y)
        return x * y


# -----------------------------------------------------
# LAM (Combined Channel + Spatial Attention)
# -----------------------------------------------------
class LAM(nn.Module):
    def __init__(self, channels):
        super(LAM, self).__init__()

        self.channel_attention = NGICAM(channels)
        self.spatial_attention = LSAM()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# -----------------------------------------------------
# ResNet50 Backbone + LAM modules
# -----------------------------------------------------
class ResNet50WithLAM(nn.Module):
    def __init__(self, num_classes=50, feature_dim=2048):
        super(ResNet50WithLAM, self).__init__()

        resnet50 = models.resnet50(pretrained=True)

        self.conv1 = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool
        )

        self.layer1 = resnet50.layer1
        self.lam1 = LAM(256)

        self.layer2 = resnet50.layer2
        self.lam2 = LAM(512)

        self.layer3 = resnet50.layer3
        self.lam3 = LAM(1024)

        self.layer4 = resnet50.layer4
        self.lam4 = LAM(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(2048, feature_dim)

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_features=False):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.lam1(x)

        x = self.layer2(x)
        x = self.lam2(x)

        x = self.layer3(x)
        x = self.lam3(x)

        x = self.layer4(x)
        x = self.lam4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        features = self.fc(x)

        if return_features:
            return features

        logits = self.classifier(features)
        return logits, features


# -----------------------------------------------------
# Enhanced Triplet Loss
# -----------------------------------------------------
class EnhancedTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(EnhancedTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, use_soft_margin=True):
        dist_ap = torch.sum((anchor - positive) ** 2, dim=1)
        dist_an = torch.sum((anchor - negative) ** 2, dim=1)

        if use_soft_margin:
            losses = torch.log(1 + torch.exp(dist_ap - dist_an))
        else:
            losses = F.relu(dist_ap - dist_an + self.margin)

        return losses.mean()


# -----------------------------------------------------
# IrisRecognitionModel (Supports return_embeddings!)
# -----------------------------------------------------
class IrisRecognitionModel(nn.Module):
    def __init__(self, num_classes=50, feature_dim=2048, pretrained=True):
        super(IrisRecognitionModel, self).__init__()

        self.backbone = ResNet50WithLAM(num_classes=num_classes, feature_dim=feature_dim)
        self.num_classes = num_classes

    def forward(self, x, return_features=False, return_embeddings=False):
        """
        Unified interface supporting:
        - return_embeddings=True  -> (logits, features)
        - return_features=True    -> features only
        - default                 -> (logits, features)
        """

        # ✔ Case for unlearning code
        if return_embeddings:
            logits, emb = self.backbone(x, return_features=False)
            return logits, emb

        # ✔ Case for triplet-loss training
        if return_features:
            return self.backbone(x, return_features=True)

        # ✔ Default classification
        logits, emb = self.backbone(x, return_features=False)
        return logits, emb

    def extract_features(self, x):
        return self.backbone(x, return_features=True)


# -----------------------------------------------------
# Factory Function
# -----------------------------------------------------
def create_iris_recognition_model(num_classes=50, feature_dim=2048, pretrained=True):
    model = IrisRecognitionModel(
        num_classes=num_classes,
        feature_dim=feature_dim,
        pretrained=pretrained
    )
    return model



# -----------------------------------------------------
# TESTING (Optional)
# -----------------------------------------------------
if __name__ == "__main__":
    print("Creating Iris Recognition Model...")
    model = create_iris_recognition_model(num_classes=50)

    batch = torch.randn(4, 3, 224, 224)

    print("\nTest classification:")
    logits, feats = model(batch)
    print(logits.shape, feats.shape)

    print("\nTest return_embeddings=True:")
    logits, feats = model(batch, return_embeddings=True)
    print(logits.shape, feats.shape)

    print("\nTest return_features=True:")
    feats = model(batch, return_features=True)
    print(feats.shape)

    print("\nModel OK!")
