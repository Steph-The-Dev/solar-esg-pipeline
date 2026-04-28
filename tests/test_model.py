import torch
import pytest
from src.train import build_model

def test_build_model():
    device = torch.device("cpu")
    model = build_model(device)
    
    # Check if model is on CPU
    assert next(model.parameters()).device.type == "cpu"
    
    # Check input channels (we expect 4: Blue, Green, Red, NIR)
    # The first layer of the encoder should have 4 input channels
    # For resnet34 in SMP, it's model.encoder.conv1
    assert model.encoder.conv1.in_channels == 4
    
    # Check output classes (we expect 1 for binary segmentation)
    # In SMP Unet, it's model.segmentation_head[0]
    # Actually SMP Unet structure might vary, let's just check output shape
    
    dummy_input = torch.randn(1, 4, 64, 64)
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape == (1, 1, 64, 64)
