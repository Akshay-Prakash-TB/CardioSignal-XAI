import torch
from models.multimodal_model import CardioSignalXAI

def test():
    # 4 samples, 12 leads, 1000 length
    ecg = torch.randn(4, 12, 1000)
    meta = torch.randn(4, 2)
    model = CardioSignalXAI(num_classes=5)
    out = model(ecg, meta)
    assert out.shape == (4, 5), f"Unexpected shape {out.shape}"
    print("[✓] Model architecture check passed! Output shape:", out.shape)

if __name__ == "__main__":
    test()