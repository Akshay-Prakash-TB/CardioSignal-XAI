import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multimodal_model import CardioSignalXAI, ConvBlock1D

# =====================================================================
# MODEL 1: CardioSignal-CrossAttn (Flagship Cross-Attention Multimodal)
# Defined in models/multimodal_model.py
# =====================================================================


# =====================================================================
# MODEL 2: ResNet1D-18 Multimodal (Deep Residual 1D Network + Late Fusion)
# =====================================================================
class ResNet1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet1DMultimodal(nn.Module):
    """
    1D ResNet with 4 residual stages for 12-lead ECG signals,
    fused with demographic risk features via late feature concatenation.
    """
    def __init__(self, in_leads=12, num_classes=5, base_filters=32):
        super().__init__()
        self.in_channels = base_filters
        self.prep = nn.Sequential(
            nn.Conv1d(in_leads, base_filters, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(base_filters, 2, stride=1)
        self.layer2 = self._make_layer(base_filters * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_filters * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_filters * 8, 2, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Metadata MLP
        self.meta_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.LayerNorm(64)
        )

        ecg_feat_dim = base_filters * 8
        self.classifier = nn.Sequential(
            nn.Linear(ecg_feat_dim + 64, 128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        layers = [ResNet1DBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResNet1DBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, ecg, metadata):
        x = self.prep(ecg)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        ecg_emb = self.gap(x).squeeze(-1)  # (Batch, 256)

        meta_emb = self.meta_net(metadata)  # (Batch, 64)
        combined = torch.cat([ecg_emb, meta_emb], dim=1)
        return self.classifier(combined)


# =====================================================================
# MODEL 3: InceptionTime-BiGRU (Multi-Scale Inception Convolutions + Recurrent)
# =====================================================================
class Inception1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 11, 23, 41), bottleneck_channels=32):
        super().__init__()
        self.use_bottleneck = in_channels > bottleneck_channels
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            conv_in = bottleneck_channels
        else:
            self.bottleneck = nn.Identity()
            conv_in = in_channels

        branch_out = out_channels // 4
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(conv_in, branch_out, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_channels, branch_out, kernel_size=1, bias=False)

        total_out = branch_out * (len(kernel_sizes) + 1)
        self.proj = nn.Conv1d(total_out, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

        # Residual shortcut
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        bx = self.bottleneck(x)
        outputs = [conv(bx) for conv in self.conv_branches]
        outputs.append(self.pool_conv(self.maxpool(x)))
        merged = torch.cat(outputs, dim=1)
        out = self.bn(self.proj(merged)) + res
        return self.act(out)


class InceptionTimeBiGRU(nn.Module):
    """
    Multi-Scale Inception 1D blocks for capturing both high-frequency QRS complexes
    and low-frequency ST-T wave morphology, followed by BiGRU temporal modeling.
    """
    def __init__(self, in_leads=12, num_classes=5, hidden_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_leads, 48, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(48),
            nn.GELU()
        )
        self.inc1 = Inception1DBlock(48, 64)
        self.pool1 = nn.MaxPool1d(2)
        self.inc2 = Inception1DBlock(64, 128)
        self.pool2 = nn.MaxPool1d(2)
        self.inc3 = Inception1DBlock(128, hidden_dim)

        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.meta_encoder = nn.Sequential(
            nn.Linear(2, 48),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(48, hidden_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, ecg, metadata):
        # ecg: (B, 12, 1000)
        x = self.stem(ecg)
        x = self.pool1(self.inc1(x))
        x = self.pool2(self.inc2(x))
        x = self.inc3(x)  # (B, hidden_dim, T=125)

        x_seq = x.permute(0, 2, 1)  # (B, T, hidden_dim)
        gru_out, _ = self.bigru(x_seq)  # (B, T, hidden_dim)
        ecg_vec = torch.mean(gru_out, dim=1)  # Global average over time: (B, hidden_dim)

        meta_vec = self.meta_encoder(metadata)  # (B, hidden_dim)
        combined = torch.cat([ecg_vec, meta_vec], dim=1)  # (B, hidden_dim*2)
        return self.classifier(combined)


# =====================================================================
# MODEL 4: ECG-ViT (1D Temporal Patch Transformer + Gated Multimodal Unit)
# =====================================================================
class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class GatedMultimodalUnit(nn.Module):
    """Dynamically gates waveform embedding with demographic embeddings."""
    def __init__(self, d_model):
        super().__init__()
        self.gate_dense = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ecg_token, meta_token):
        combined = torch.cat([ecg_token, meta_token], dim=1)
        gate = self.gate_dense(combined)
        h = torch.tanh(self.proj(combined))
        fused = gate * ecg_token + (1.0 - gate) * h
        return self.norm(fused)


class ECGViTGated(nn.Module):
    """
    1D Vision Transformer treating ECG lead segments as tokens,
    fused with demographic data via a Gated Multimodal Unit (GMU).
    """
    def __init__(self, in_leads=12, num_classes=5, patch_size=50, d_model=128, n_heads=4, n_layers=3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = 1000 // patch_size  # 20 tokens
        patch_dim = in_leads * patch_size     # 12 * 50 = 600

        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoder = PositionalEncoding1D(d_model, max_len=self.num_patches + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Demographic projection
        self.meta_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model)
        )

        self.gmu = GatedMultimodalUnit(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, ecg, metadata):
        # ecg: (B, 12, 1000) -> Reshape into (B, num_patches, patch_dim)
        b, c, t = ecg.shape
        x_patches = ecg.unfold(2, self.patch_size, self.patch_size).permute(0, 2, 1, 3).contiguous()
        x_patches = x_patches.view(b, self.num_patches, -1)  # (B, 20, 600)

        tokens = self.patch_embed(x_patches)  # (B, 20, d_model)
        cls_tokens = self.cls_token.expand(b, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 21, d_model)
        tokens = self.pos_encoder(tokens)

        encoded = self.transformer(tokens)  # (B, 21, d_model)
        cls_out = encoded[:, 0, :]  # CLS token representation: (B, d_model)

        meta_out = self.meta_encoder(metadata)  # (B, d_model)
        fused = self.gmu(cls_out, meta_out)     # (B, d_model)
        return self.classifier(fused)


# =====================================================================
# MODEL 5: DenseNet1D-SE (1D DenseNet with Squeeze-and-Excitation Lead Attention)
# =====================================================================
class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, max(channels // reduction, 8), kernel_size=1),
            nn.GELU(),
            nn.Conv1d(max(channels // reduction, 8), channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.se(x)
        return x * weights


class Dense1DLayer(nn.Module):
    def __init__(self, in_features, growth_rate):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.conv = nn.Conv1d(in_features, growth_rate, kernel_size=7, padding=3, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        new_feats = self.conv(self.act(self.bn(x)))
        return torch.cat([x, new_feats], dim=1)


class Dense1DBlock(nn.Module):
    def __init__(self, num_layers, in_features, growth_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(Dense1DLayer(in_features + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)
        self.se = SqueezeExcitation1D(in_features + num_layers * growth_rate)

    def forward(self, x):
        feats = self.block(x)
        return self.se(feats)


class Transition1D(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.trans = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.GELU(),
            nn.Conv1d(in_features, out_features, kernel_size=1, bias=False),
            nn.AvgPool1d(2)
        )

    def forward(self, x):
        return self.trans(x)


class DenseNet1DSE(nn.Module):
    """
    1D DenseNet with Squeeze-and-Excitation attention across feature channels,
    modeling cross-lead anatomical interactions and demographic risk factors.
    """
    def __init__(self, in_leads=12, num_classes=5, growth_rate=16, init_features=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_leads, init_features, kernel_size=11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(init_features),
            nn.GELU(),
            nn.MaxPool1d(2)
        )

        num_feats = init_features
        self.block1 = Dense1DBlock(num_layers=3, in_features=num_feats, growth_rate=growth_rate)
        num_feats += 3 * growth_rate
        self.trans1 = Transition1D(num_feats, num_feats // 2)
        num_feats = num_feats // 2

        self.block2 = Dense1DBlock(num_layers=3, in_features=num_feats, growth_rate=growth_rate)
        num_feats += 3 * growth_rate
        self.trans2 = Transition1D(num_feats, num_feats // 2)
        num_feats = num_feats // 2

        self.block3 = Dense1DBlock(num_layers=3, in_features=num_feats, growth_rate=growth_rate)
        num_feats += 3 * growth_rate

        self.bn_final = nn.BatchNorm1d(num_feats)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.meta_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64)
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_feats + 64, 128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, ecg, metadata):
        x = self.stem(ecg)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.block3(x)
        x = F.gelu(self.bn_final(x))
        ecg_emb = self.gap(x).squeeze(-1)

        meta_emb = self.meta_net(metadata)
        fused = torch.cat([ecg_emb, meta_emb], dim=1)
        return self.classifier(fused)


# =====================================================================
# MODEL 6: Baseline-CNN1D (Standard 3-Layer Convolutional Baseline)
# =====================================================================
class BaselineCNN1D(nn.Module):
    """
    Standard 3-stage 1D CNN baseline with direct metadata concatenation.
    Acts as an ablation benchmark to quantify gains from advanced attention/residual/recurrent modules.
    """
    def __init__(self, in_leads=12, num_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock1D(in_leads, 32, kernel_size=15, pool=2),
            ConvBlock1D(32, 64, kernel_size=7, pool=2),
            ConvBlock1D(64, 128, kernel_size=5, pool=2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.meta_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, ecg, metadata):
        ecg_vec = self.conv(ecg).squeeze(-1)
        meta_vec = self.meta_fc(metadata)
        return self.classifier(torch.cat([ecg_vec, meta_vec], dim=1))


# =====================================================================
# MODEL REGISTRY / FACTORY
# =====================================================================
MODEL_REGISTRY = {
    "CardioSignal-CrossAttn": {
        "class": CardioSignalXAI,
        "name": "CardioSignal-CrossAttn (Flagship)",
        "description": "1D-CNN + BiLSTM + Demographic MLP with Multi-Head Cross-Attention Fusion",
        "family": "Attention & Recurrent",
        "checkpoint_pattern": "cardiosignal_crossattn*.ckpt"
    },
    "ResNet1D-18": {
        "class": ResNet1DMultimodal,
        "name": "ResNet1D-18 Multimodal",
        "description": "18-Layer Deep 1D Residual Network with Skip Connections & Late Demographics Fusion",
        "family": "Residual CNN",
        "checkpoint_pattern": "resnet1d_18*.ckpt"
    },
    "InceptionTime-BiGRU": {
        "class": InceptionTimeBiGRU,
        "name": "InceptionTime-BiGRU",
        "description": "Multi-Scale Inception 1D Convolutions (k=5,11,23,41) with Bidirectional GRU",
        "family": "Multi-Scale & Recurrent",
        "checkpoint_pattern": "inceptiontime_bigru*.ckpt"
    },
    "ECG-ViT-Gated": {
        "class": ECGViTGated,
        "name": "ECG-ViT Gated Transformer",
        "description": "Patch-based 1D Temporal Vision Transformer with Gated Multimodal Unit (GMU)",
        "family": "Transformer / ViT",
        "checkpoint_pattern": "ecg_vit_gated*.ckpt"
    },
    "DenseNet1D-SE": {
        "class": DenseNet1DSE,
        "name": "DenseNet1D-SE Attention",
        "description": "Densely Connected 1D Network with Squeeze-and-Excitation Lead Attention",
        "family": "Dense & Channel Attention",
        "checkpoint_pattern": "densenet1d_se*.ckpt"
    },
    "Baseline-CNN1D": {
        "class": BaselineCNN1D,
        "name": "Baseline 1D-CNN",
        "description": "Standard 3-Stage 1D Convolutional Network with Direct Concatenation (Ablation)",
        "family": "Baseline CNN",
        "checkpoint_pattern": "baseline_cnn1d*.ckpt"
    }
}


def build_model(model_key: str, num_classes: int = 5, **kwargs):
    """Instantiates a model architecture from the registry."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}")
    model_cls = MODEL_REGISTRY[model_key]["class"]
    return model_cls(num_classes=num_classes, **kwargs)


def list_available_models():
    """Returns a dictionary summarizing all available architectures."""
    return {k: {"name": v["name"], "description": v["description"], "family": v["family"]} 
            for k, v in MODEL_REGISTRY.items()}
