import torch
import torch.nn as nn

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pool=2):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))

class ECGEncoder(nn.Module):
    def __init__(self, in_leads=12, d_model=128):
        super().__init__()
        # Downsamples time from 1000 -> 500 -> 250 -> 125
        self.net = nn.Sequential(
            ConvBlock1D(in_leads, 32, kernel_size=15, pool=2),
            ConvBlock1D(32, 64, kernel_size=7, pool=2),
            ConvBlock1D(64, d_model, kernel_size=5, pool=2)
        )
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        feats = self.net(x)            # (Batch, d_model, 125)
        feats = feats.permute(0, 2, 1)  # (Batch, 125, d_model)
        out, _ = self.bilstm(feats)     # (Batch, 125, d_model)
        return out

class MetadataEncoder(nn.Module):
    def __init__(self, in_features=2, d_model=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        return self.mlp(x).unsqueeze(1)  # (Batch, 1, d_model)

class CrossAttentionFusion(nn.Module):
    """Novelty 1: Demographics Querying ECG Morphological Tokens."""
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads, 
            batch_first=True,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ecg_tokens, meta_token):
        attn_out, attn_weights = self.cross_attn(
            query=meta_token,
            key=ecg_tokens,
            value=ecg_tokens
        )
        fused = self.norm(meta_token + attn_out).squeeze(1)
        return fused, attn_weights

class CardioSignalXAI(nn.Module):
    def __init__(self, num_classes=5, d_model=128, n_heads=4):
        super().__init__()
        self.ecg_encoder = ECGEncoder(in_leads=12, d_model=d_model)
        self.meta_encoder = MetadataEncoder(in_features=2, d_model=d_model)
        self.fusion = CrossAttentionFusion(d_model=d_model, n_heads=n_heads)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, ecg, metadata):
        ecg_tokens = self.ecg_encoder(ecg)
        meta_token = self.meta_encoder(metadata)
        fused, _ = self.fusion(ecg_tokens, meta_token)
        return self.classifier(fused)