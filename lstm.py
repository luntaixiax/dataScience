import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# hyperparameters
input_size = 28  # number of features, feature length of one word (K)
sequence_length = 28  # how many words in one sentence (time stamp T)
num_layers = 2
hidden_size = 256 # hidden state size of c and h (H)

num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        # x.size(0) = batch_size
        h0 = torch.zeros((self.num_layers * 2, x.size(0), self.hidden_size)).to(device)
        c0 = torch.zeros((self.num_layers * 2, x.size(0), self.hidden_size)).to(device)
        # forward pass
        # out: (batch_size, seq_length, hidden_size*2)
        # ht, ct: (num_layers*2, batch_size, hidden_size)
        out, (ht, ct) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1:, :]) # out-> (batch_size, 1, hidden_size*2) -> (batch_size, 1, num_classes)
        return out.squeeze(1) # out-> (batch_size, num_classes)


# load data
transform = transforms.Compose(
    [transforms.ToTensor()]
)

train_dataset = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

test_dataset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize model
model = BiLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

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
            x = x.to(device).squeeze(1)
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
            data = data.to(device).squeeze(1)  # remove dimension 1 (Nx1x28x28) -> (Nx28x28)
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
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss = loss.item())

    check_accurary(train_loader, model)
    check_accurary(test_loader, model)


if __name__ == '__main__':
    main()


