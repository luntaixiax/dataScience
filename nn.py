import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        # (28x28)
        super(NN, self).__init__()

        self.m = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=num_classes),
        )

    def forward(self, x):
        return self.m(x)


# hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# load data
transform = transforms.Compose(
    [transforms.ToTensor()]
)

train_dataset = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)

test_dataset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

# Initialize model
model = NN(input_size, num_classes).to(device)

# loss fucntion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# check accuracy
def check_accurary(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # set to evaluation mode

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).reshape(x.shape[0], -1)
            y = y.to(device)

            scores = model(x)
            max_values, max_pos = scores.max(dim=1)
            num_correct += (max_pos == y).sum().item()
            num_samples += max_pos.size(0)

        print("Got {num_correct} / {num_samples} with accuracy {rate:.2f}".format(
            num_correct = num_correct, num_samples = num_samples, rate = num_correct/num_samples))

    model.train()  # set back to train mode

def main():
    # Train network
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), leave = False, total = len(train_loader))
        for batch_idx, (data, targets) in loop:
            # data, targets: (batch_size, channel=1, img_width, img_height)
            data = data.to(device).reshape(data.shape[0], -1)  # reshape the last three dimension
            targets = targets.to(device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent
            optimizer.step()

            # update progress bar
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss = loss.item())

    check_accurary(train_loader, model)
    check_accurary(test_loader, model)


if __name__ == '__main__':
    main()