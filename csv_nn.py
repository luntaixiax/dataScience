import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from customDataset import CSVSingleDataset


device = "cuda" if torch.cuda.is_available() else "cpu"

class NN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(NN, self).__init__()

        self.nn = nn.Sequential(
            nn.BatchNorm1d(num_features = in_channels),
            nn.Linear(in_features = in_channels, out_features=6),
            nn.ReLU(),
            nn.Linear(in_features = 6, out_features = 8),
            nn.ReLU(),
            nn.Linear(in_features = 8, out_features = num_classes),
        )

    def forward(self, x):
        return self.nn(x)



# hyperparameters
in_channels = 13
num_classes = 3
learning_rate = 0.001
batch_size = 10
num_epochs = 5

# load data
transform = None

x_cols = ['Alcohol','Malic.acid','Ash','Acl','Mg','Phenols','Flavanoids','Nonflavanoid.phenols','Proanth','Color.int','Hue','OD','Proline']
csvFile = "dataset/simpleTest/wine.csv"

train_dataset = CSVSingleDataset(csvFile, x_cols = x_cols, y_col = 'Wine', transform = transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)

test_dataset = CSVSingleDataset(csvFile, x_cols = x_cols, y_col = 'Wine', transform = transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

# Initialize model
model = NN(in_channels, num_classes).to(device)

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
    for epoch in range(num_epochs):

        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            # forward
            outputs = model(data)
            loss = criterion(outputs, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent
            optimizer.step()

            print(loss.item())


    check_accurary(train_loader, model)
    check_accurary(test_loader, model)

if __name__ == '__main__':
    main()