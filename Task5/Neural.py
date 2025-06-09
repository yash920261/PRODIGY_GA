import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing transforms for VGG
vgg_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
unnormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                   std=[1/0.229, 1/0.224, 1/0.225])

# Load and preprocess image
def load_image(image_path, max_size=512):
    image = Image.open(image_path).convert('RGB')
    size = max(image.size)
    if size > max_size:
        scale = max_size / size
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, Image.LANCZOS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        vgg_normalize
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image

# Display image
def imshow(tensor, title=None):
    image = tensor.clone().detach().squeeze(0).cpu()
    image = unnormalize(image)
    image = torch.clamp(image, 0, 1)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Gram matrix
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2)) / (c * h * w)
    return G

# Style Transfer Model
class StyleTransferModel(nn.Module):
    def __init__(self, style_layers, content_layers):
        super(StyleTransferModel, self).__init__()
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        self.model = nn.Sequential()
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.layer_mapping = {}

        i = 0
        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
            self.model.add_module(name, layer)
            self.layer_mapping[name] = len(self.model) - 1
            if name == 'conv_4_2':
                break
            if isinstance(layer, nn.Conv2d):
                i += 1

    def forward(self, x):
        style_feats, content_feats = [], []
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.style_layers:
                style_feats.append(x)
            if name in self.content_layers:
                content_feats.append(x)
        return style_feats, content_feats

# Compute loss
def compute_loss(style_outputs, content_outputs, style_targets, content_targets,
                 style_weight=1e6, content_weight=1):
    style_loss = sum(nn.functional.mse_loss(gram_matrix(o), gram_matrix(t))
                     for o, t in zip(style_outputs, style_targets))
    content_loss = sum(nn.functional.mse_loss(o, t)
                       for o, t in zip(content_outputs, content_targets))
    return style_weight * style_loss + content_weight * content_loss

# Load images
content_image = load_image("content.jpg")
style_image = load_image("style.jpg", max_size=content_image.shape[-1])

# Define layer names to extract features from
style_layer_names = ['conv_0', 'conv_5', 'conv_10', 'conv_19', 'conv_28']
content_layer_names = ['conv_21']

# Create model
model = StyleTransferModel(style_layers=style_layer_names,
                           content_layers=content_layer_names).to(device)

# Extract targets
with torch.no_grad():
    style_targets, content_targets = model(style_image)

# Input image
input_image = content_image.clone().requires_grad_(True)

# Optimizer
optimizer = optim.Adam([input_image], lr=0.01)

# Optimization loop
num_steps = 300
for step in range(num_steps):
    optimizer.zero_grad()
    style_outputs, content_outputs = model(input_image)
    loss = compute_loss(style_outputs, content_outputs, style_targets, content_targets)
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# Final output clamp and save
final_image = input_image.clone().detach().squeeze(0).cpu()
final_image = unnormalize(final_image)
final_image = torch.clamp(final_image, 0, 1)
save_image(final_image, "stylized_image.png")
print("Stylized image saved as stylized_image.png")

# Show the final image
imshow(input_image, title="Stylized Image")
