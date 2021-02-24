import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"



class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.fc = nn.Linear(in_features=16*7*7, out_features=num_classes)

    def forward(self, x):
        x = self.cnn(x).reshape(x.shape[0], -1) # flatten
        return self.fc(x)



# hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# load data
transform = transforms.Compose(
    [transforms.ToTensor()]
)

train_dataset = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)

test_dataset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)

# Initialize model
model = CNN(in_channels, num_classes).to(device)

# loss fucntion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

writer = SummaryWriter(f"runs/MNIST/tryingout_tensorboard")

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

def main():
    # Train network
    step = 0
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), leave = False, total = len(train_loader))
        for batch_idx, (data, targets) in loop:
            step += 1
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

            # update progress bar
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss = loss.item())

            # calculate running accurary
            max_values, max_pos = scores.max(1)
            num_correct = (max_pos == targets).sum()
            running_train_acc = float(num_correct) / float(data.shape[0])

            writer.add_scalar("Training Loss", loss, global_step = step)
            writer.add_scalar("Training Accuracy", running_train_acc, global_step = step)

    check_accurary(train_loader, model)
    check_accurary(test_loader, model)

if __name__ == '__main__':
    main()


