import torch
from model import CNN_classifier  # Importing from model.py

# Load the trained model
n_classes = 37  # Change according to your setup
model = CNN_classifier(n_classes)
model.load_state_dict(torch.load("/teamspace/studios/this_studio/model_cnn.pth", map_location=torch.device('cpu')))
model.eval()


# Hook function to capture feature maps
feature_maps = {}

def hook_fn(module, input, output):
    feature_maps["conv1"] = output.detach()

# Register the hook
model.conv1.register_forward_hook(hook_fn)

# Example input tensor (replace with actual image preprocessing)
input_tensor = torch.randn(1, 3, 22, 22)  # Assuming input size of 22x22

# Forward pass
_ = model(input_tensor)

# Feature map extraction
print(feature_maps["conv1"].shape)  # Expected: (1, 16, 22, 22)

