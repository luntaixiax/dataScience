import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

def convBlock(in_channels, out_channels, kernal_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernal_size, stride = stride, padding = padding, bias = False),
        nn.BatchNorm2d(num_features = out_channels),
        nn.LeakyReLU(0.2),
    )

def convTransBlock(in_channels, out_channels, kernal_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernal_size, stride = stride, padding = padding, bias = False),
        nn.BatchNorm2d(num_features = out_channels),
        nn.ReLU(),
    )

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, mean = 0.0, std = 0.02)


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()

        # Input: N x channels_img x 64 x 64
        self.disc = nn.Sequential(
            # N x channels_img x 64 x 64 -> N x features_d x 32 x 32
            nn.Conv2d(in_channels = channels_img, out_channels = features_d, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2),
            # N x features_d x 32 x 32 -> N x features_d*2 x 16 x 16
            convBlock(features_d, features_d * 2, 4, 2, 1),
            # N x features_d*2 x 16 x 16 -> N x features_d*4 x 8 x 8
            convBlock(features_d * 2, features_d * 4, 4, 2, 1),
            # N x features_d*4 x 8 x 8 -> N x features_d*8 x 4 x 4
            convBlock(features_d * 4, features_d * 8, 4, 2, 1),
            # N x features_d*8 x 4 x 4 -> N x 1 x 1 x 1
            nn.Conv2d(in_channels = features_d * 8, out_channels = 1, kernel_size = 4, stride = 2, padding = 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # N x channels_img x 64 x 64 -> N x 1 x 1 x 1
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            # N x Z_dim x 1 x 1 -> N x features_g*16 x 4 x 4
            convTransBlock(z_dim, features_g * 16, 4, 1, 0),
            # N x features_g*16 x 4 x 4 -> N x features_g*8 x 8 x 8
            convTransBlock(features_g * 16, features_g * 8, 4, 2, 1),
            # N x features_g*8 x 8 x 8 -> N x features_g*4 x 16 x 16
            convTransBlock(features_g * 8, features_g * 4, 4, 2, 1),
            # N x features_g*4 x 16 x 16 -> N x features_g*2 x 32 x 32
            convTransBlock(features_g * 4, features_g * 2, 4, 2, 1),
            # N x features_g*2 x 32 x 32 -> N x channels_img x 64 x 64
            nn.ConvTranspose2d(in_channels = features_g * 2, out_channels = channels_img, kernel_size = 4,
                               stride = 2, padding = 1),
            nn.Tanh(),
        )

    def forward(self, x):
        # N x Z_dim x 1 x 1 -> N x channels_img x 64 x 64
        return self.gen(x)


device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters:
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 64
FEATURES_GEN = 64
FEATURES_DISC = 64

transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for c in range(CHANNELS_IMG)], [0.5 for c in range(CHANNELS_IMG)]),
    ]
)

# dataset = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
dataset = datasets.ImageFolder(root='D:\LargeDatasets\celeb_dataset', transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr = LEARNING_RATE, betas = (0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr = LEARNING_RATE, betas = (0.5, 0.999))

criterion = nn.BCELoss()

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")

gen.train()
disc.train()

def main():
    step = 0

    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(loader), leave = False, total = len(loader))
        for batch_idx, (real, targets) in loop:
            # real : N x channels_img x 64 x 64
            # targets:

            real = real.to(device)
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)

            '''train discriminator'''
            # N x channels_img x 64 x 64 -> N x 1 x 1 x 1 -> N
            disc_real = disc(real).reshape(-1)
            real_label = torch.ones_like(disc_real) # N
            loss_disc_real = criterion(disc_real, real_label)

            disc_fake = disc(fake.detach()).reshape(-1)
            fake_label = torch.zeros_like(disc_fake)  # N
            loss_disc_fake = criterion(disc_fake, fake_label)

            disc.zero_grad()
            loss_disc = loss_disc_real + loss_disc_fake
            loss_disc.backward()
            opt_disc.step()

            '''train generator'''
            output = disc(fake).reshape(-1)
            real_label = torch.ones_like(output)  # N
            loss_gen = criterion(output, real_label)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()


            '''log loss and tensorboard'''
            # update progress bar
            loop.set_description(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")

            if batch_idx % 100 == 0:
                print(f"Loss D: {loss_disc: .4f}, Loss G: {loss_gen: .4f}")

                with torch.no_grad():
                    fake = gen(fixed_noise)

                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize = True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize = True)
                    writer_real.add_image("Celebrity Real Images", img_grid_real, global_step = step)
                    writer_fake.add_image("Celebrity Fake Images", img_grid_fake, global_step = step)

                step += 1




if __name__ == '__main__':
    main()