import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision

device = "cuda" if torch.cuda.is_available() else "cpu"

# hyperparameters
in_channels = 3
num_classes = 10
learning_rate = 0.001
batch_size = 1024
num_epochs = 5


# load model from pretrained ones
model = torchvision.models.vgg16(pretrained=True)

# modify model
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

model.avgpool = Identity()
model.classifier = nn.Linear(in_features=512, out_features=10)
model.to(device)

# load data
transform = transforms.Compose(
    [transforms.ToTensor()]
)

train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

test_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


# loss fucntion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # data, targets: (batch_size, channel=1, img_width, img_height)
        data = data.to(device)
        targets = targets.to(device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()

# check accuracy

def check_accurary(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # set to evaluation mode

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            max_values, max_pos = scores.max(dim=1)
            num_correct += (max_pos == y).sum().item()
            num_samples += max_pos.size(0)

        print("Got {num_correct} / {num_samples} with accuracy {rate:.2f}".format(
            num_correct = num_correct, num_samples = num_samples, rate = num_correct/num_samples))

    model.train()  # set back to train mode

check_accurary(train_loader, model)
check_accurary(test_loader, model)


