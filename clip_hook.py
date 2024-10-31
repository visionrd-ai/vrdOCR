import torch
import clip

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Dictionary to store the shapes at each layer
layer_shapes = {}

# Hook function to capture the shape of the tensors
def hook_fn(module, input, output):
    if isinstance(output, tuple):
        layer_shapes[module] = output[0].shape
    else:
        layer_shapes[module] = output.shape

# Register hooks on all layers
for name, layer in model.named_modules():
    layer.register_forward_hook(hook_fn)

# Example input image (you can replace this with your own image tensor)
# Shape here is assumed for demonstration, adjust as per your input format
input_image = torch.randn(1, 3, 224, 224).to(device)
input_text = clip.tokenize(["a cat", "a dog"]).to(device)

# Forward pass to capture tensor shapes
with torch.no_grad():
    model.encode_image(input_image)
    model.encode_text(input_text)

# Print out the layer shapes
for layer, shape in layer_shapes.items():
    print(f"Output Shape: {shape}")
