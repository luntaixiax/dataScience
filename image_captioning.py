import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from tqdm import tqdm

from customDataset import get_FlickrLoader, get_FlickrDataset


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN = False):
        super(EncoderCNN, self).__init__()

        self.train_CNN = train_CNN

        self.incecption = models.inception_v3(pretrained = True, aux_logits = False)
        self.incecption.fc = nn.Linear(in_features = self.incecption.fc.in_features, out_features = embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.5)

    def forward(self, images):
        features = self.incecption(images)

        for name, param in self.incecption.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN

        return self.dropout(self.relu(features))

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()

        self.embed = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(in_features = hidden_size, out_features = vocab_size)
        self.dropout = nn.Dropout(p = 0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim = 0)

        hiddens, (h_n, c_n) = self.lstm(embeddings)  # h0, c0 default to 0
        outputs = self.linear(hiddens)

        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()

        self.encoderCNN = EncoderCNN(embed_size, train_CNN=False)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length = 50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for i in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))

                predicted_pos = output.argmax(1)
                predicted_str = vocabulary.itos[predicted_pos.item()]

                result_caption.append(predicted_str)

                x = self.decoderRNN.embed(predicted_pos).unsqueeze(0)

                if predicted_str == "<EOS>":
                    break

        return result_caption

def train():
    transform = transforms.Compose(
        [
            # https://pytorch.org/hub/pytorch_vision_inception_v3/
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )

    dataset = get_FlickrDataset(transform)
    train_loader = get_FlickrLoader(
        dataset, batch_size = 32, num_workers = 6, shuffle = True, pin_memory = True
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = False

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    # initialize model, loss etc
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index = dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    # for tensorboard
    writer = SummaryWriter("runs/flickr")

    for epoch in range(num_epochs):
        if save_model:
            checkpoint = {
                "state_dict" : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "step" : epoch,
            }
            save_checkpoint(checkpoint)

        loop = tqdm(enumerate(train_loader), leave = False, total = len(train_loader))
        for idx, (imgs, captions) in loop:
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            # predict size: (seq_len, N, vocab_size)  target: (seq_len, N)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)) # convert 3d to 2d

            writer.add_scalar("Training loss", loss.item(), global_step= epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update progress bar
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss = loss.item())


if __name__ == "__main__":
    train()