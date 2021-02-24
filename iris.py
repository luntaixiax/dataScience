from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 3),
        )

    def forward(self, x):
        return self.nn(x)



model = Model(input_dim = 4).to(device)


# loss fucntion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

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


class IrisDataset(Dataset):
    def __init__(self, train = True, transform = None):
        features, labels = load_iris(return_X_y = True)
        self.features_train, self.features_test, self.labels_train, self.labels_test = train_test_split(features, labels, random_state = 42,
                                                                                    shuffle = True)
        self.transform = transform

        if train is True:
            self.Xs = torch.from_numpy(self.features_train)
            self.Ys = torch.from_numpy(self.labels_train)
        else:
            self.Xs = torch.from_numpy(self.features_test)
            self.Ys = torch.from_numpy(self.labels_test)

    def __len__(self):
        return len(self.Ys)

    def __getitem__(self, item):
        """
        @param item: index
        @return: img dataframe, label
        """
        X = self.Xs[item].float()
        y = self.Ys[item].long()

        if self.transform is not None:
            X = self.transform(X)

        return X, y

train_dataset = IrisDataset(train = True, transform = None)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=6, pin_memory=True)

test_dataset = IrisDataset(train = False, transform = None)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=6, pin_memory=True)

def main():
    # Train network
    num_epochs = 30
    for epoch in range(num_epochs):

        loop = tqdm(enumerate(train_loader), leave = True, total = len(train_loader))
        for batch_idx, (data, targets) in loop:
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

            # update progress bar
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss = loss.item())


    check_accurary(train_loader, model)
    check_accurary(test_loader, model)

if __name__ == '__main__':
    main()