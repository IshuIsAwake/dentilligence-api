import torch
import torch.nn as nn
import torchvision.models as models


class Classifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        self.backbone = nn.Module()
        self.backbone.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


model = Classifier(num_classes=5)
state = torch.load("classifier.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()

torch.onnx.export(
    model,
    torch.randn(1, 3, 224, 224),
    "classifier.onnx",
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
print("Exported classifier.onnx")
