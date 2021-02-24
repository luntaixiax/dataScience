import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import save_image
from tqdm import tqdm

from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
#model = models.vgg19(pretrained = True).features
'''
Sequential(
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace=True)
  (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU(inplace=True)
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (6): ReLU(inplace=True)
  (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (8): ReLU(inplace=True)
  (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (11): ReLU(inplace=True)
  (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (13): ReLU(inplace=True)
  (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (15): ReLU(inplace=True)
  (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (17): ReLU(inplace=True)
  (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (20): ReLU(inplace=True)
  (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (22): ReLU(inplace=True)
  (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (24): ReLU(inplace=True)
  (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (26): ReLU(inplace=True)
  (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (29): ReLU(inplace=True)
  (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (31): ReLU(inplace=True)
  (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (33): ReLU(inplace=True)
  (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (35): ReLU(inplace=True)
  (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
'''

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.chosen_features = [0, 5, 10, 19, 28]  # conv layers selected from vgg19
        self.model = models.vgg19(pretrained = True).features[:29]  # take only 29 layers

    def forward(self, x):
        features = []

        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.chosen_features:
                features.append(x)

        return features

model = VGG().to(device)
model.eval() # no change to model

image_size = 356
transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
)

def load_image(image_name):
    image = Image.open(image_name)
    img = transform(image).unsqueeze(0)
    return img.to(device)

# images (1 original + 1 style + 1 generated)
original_img = load_image("images/shiyuanlimei.jpg")
style_img = load_image("images/style.jpg")
generated = original_img.clone().requires_grad_(True)

# hyper parameters
total_steps = 6000
learning_rate = 0.001
alpha = 1  # generator loss weight
beta = 0.01  # style loss weight

optimizer = optim.Adam([generated], lr = learning_rate)  # no weights to update, only optimize generated

for step in range(total_steps):
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_features = model(style_img)

    style_loss = 0
    original_loss = 0
    for gen_feature, orig_feature, style_feature in zip(generated_features, original_img_features, style_features):
        batch_size, channel, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - orig_feature)**2)

        # compute gram matrix
        g = gen_feature.view(channel, height*width)
        G = g.mm(g.t())

        s = style_feature.view(channel, height*width)
        S = s.mm(s.t())

        style_loss += torch.mean((G - S)**2)

    total_loss = alpha * original_loss + beta * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(total_loss.item())
        save_image(generated, "images/generated_%d.png" % step)



