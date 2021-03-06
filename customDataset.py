import os
import pandas as pd
import numpy as np
import spacy
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.nn.utils.rnn import pad_sequence # pad batch
from PIL import Image # load img
from skimage import io

class CustomIMGSet(Dataset):
    def __init__(self, img_dir, name_label_dict: dict, transform=None):
        """
        @param img_dir: the directory of images
        @param name_label_dict: a dictionary contains mapping of image filename and label (numerize)
        @param transform: torchvision.transforms object
        """
        self.img_dir = img_dir
        self.names = list(name_label_dict.keys())
        self.labels = list(name_label_dict.values())
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        """
        @param item: index
        @return: img dataframe, label
        """
        imgpath = os.path.join(self.img_dir, self.names[item])
        label = self.labels[item]

        img = io.imread(imgpath)
        if self.transform:
            img = self.transform(img)

        return img, label

def get_CatDogDataSet(transform):
    img_dir = "D:\\LargeDatasets\\cats_dogs\\train"
    # label_csv = "D:\\LargeDatasets\\cats_dogs\\sampleSubmission.csv"

    name_label_dict = {}
    name_label_dict.update({"cat.%d.jpg" % i: 0  for i in range(12500)})
    name_label_dict.update({"dog.%d.jpg" % i: 1  for i in range(12500)})

    data_set = CustomIMGSet(img_dir, name_label_dict, transform = transform)
    return data_set

def get_CatDogLoader(data_set, batch_size = 32, num_workers = 0, shuffle = True, pin_memory = True):

    loader = DataLoader(
        dataset = data_set,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = shuffle,
        pin_memory = pin_memory
    )

    return loader





# if have OSError, do: python -m spacy download en
spacy_eng = spacy.load("en")

class Vocabulary():
    def __init__(self, freq_threshold):
        """
        @param freq_threshold: words that appear more than this threshold will keep in vocabulary
        """
        self.itos = {0 : "<PAD>", 1 : "<SOS>", 2 : "<EOS>", 3 : "<UNK>"}
        self.stoi = {"<PAD>" : 0, "<SOS>" : 1, "<EOS>" : 2, "<UNK>" : 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @ classmethod
    def tokenizer_eng(cls, sentence):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(sentence)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4  # already include 4 in init

        for sentence in sentence_list:
            tokened_sentence = self.tokenizer_eng(sentence)
            for word in tokened_sentence:
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, sentence):
        tokenized_s = self.tokenizer_eng(sentence)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_s
        ]


class FilckrDataset(Dataset):
    def __init__(self, img_dir, captions_txt, transform = None, freq_threshold = 5):
        self.img_dir = img_dir
        self.df = pd.read_csv(captions_txt)
        self.transform = transform

        # get img, caption columns
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        # initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        """
        @param item: index
        @return: img dataframe, label
        """
        caption = self.captions[item]
        img_id = self.imgs[item]
        img = Image.open(os.path.join(self.img_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        start_tag = self.vocab.stoi["<SOS>"]  # start of sentence
        end_tag = self.vocab.stoi["<EOS>"]  # end of sentence
        numericalized_caption = [start_tag]
        numericalized_caption.extend(self.vocab.numericalize(caption))
        numericalized_caption.append(end_tag)

        return img, torch.tensor(numericalized_caption)

class MyCollate():
    def __init__(self, pad_idx):
        """
        Do padding to make sure all captions are same in length
        will automatically add <PAD> to blank placeholders
        @param pad_idx: index of <PAD> in vocab, in this case 0
        """
        self.pad_idx = pad_idx

    def __call__(self, batch):
        """
        1. concatenate all imgs in one batch into one tensor
        2. add padding to labels so that each label has same length
        @param batch: results from  FilckrDataset.__getitem__: [img, label]
        @return: concatenated img, padded label
        """
        imgs = [img.unsqueeze(0) for img, label in batch]  # first add a dimension in order to use torch.cat
        imgs = torch.cat(imgs, dim=0) # concat all imgs in a batch in one torch tensor
        targets = [label for img, label in batch]
        targets = pad_sequence(targets, batch_first = False, padding_value = self.pad_idx)
        return imgs, targets

def get_FlickrDataset(transform):
    img_dir = "D:\\LargeDatasets\\flickr8k\\images"
    captions_txt = "D:\\LargeDatasets\\flickr8k\\captions.txt"

    return FilckrDataset(img_dir, captions_txt, transform = transform)

def get_FlickrLoader(data_set, batch_size = 32, num_workers = 0, shuffle = True, pin_memory = True):

    loader = DataLoader(
        dataset = data_set,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = shuffle,
        pin_memory = pin_memory,
        collate_fn = MyCollate(pad_idx = data_set.vocab.stoi["<PAD>"])
    )

    return loader

def main():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    data_loader = get_FlickrLoader(transform = transform, num_workers = 8)

    return data_loader

# csv dataloader
class CSVSingleDataset(Dataset):
    def __init__(self, csvFile, x_cols, y_col, transform = None):
        df = pd.read_csv(csvFile, dtype = np.float32)
        self.Xs = torch.from_numpy(df[x_cols].values)
        self.Ys = torch.from_numpy(df[y_col].values)
        self.transform = transform

    def __len__(self):
        return len(self.Ys)

    def __getitem__(self, item):
        """
        @param item: index
        @return: img dataframe, label
        """
        X = self.Xs[item]
        y = self.Ys[item].long()

        if self.transform is not None:
            X = self.transform(X)

        return X, y


if __name__ == "__main__":
    csvFile = "dataset/simpleTest/wine.csv"

    c = CSVSingleDataset(csvFile, x_cols = ['Alcohol','Malic.acid','Ash','Acl','Mg','Phenols','Flavanoids','Nonflavanoid.phenols','Proanth','Color.int','Hue','OD','Proline'], y_col = 'Wine')

    dataloader = DataLoader(dataset = c, batch_size = 4, shuffle = False)

    for (data, target) in dataloader:
        print(data, target)

    # num_epochs = 2
    # total_samples = len(c)
    #
    # for epoch in range(num_epochs):
    #     for idx, (inputs, labels) in enumerate(dataloader):

