import torch
import torch.nn as nn
import torchvision.models as models


DROPOUT = 0.1


class Classifier(nn.Module):
    # Mirror of models.EfficientNetB0Classifier so the trained state_dict
    # produces identical logits when run through ONNX.
    def __init__(self, num_classes=5):
        super().__init__()
        net = models.efficientnet_b0(weights=None)
        net.classifier = nn.Identity()
        self.backbone = net
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(1280, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


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
