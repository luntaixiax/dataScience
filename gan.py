import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()

        # conv2d size: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        self.net = nn.Sequential(
            # (batch x channel_imgs x 64 x 64) -> (batch x features_d x 32 x 32)
            nn.Conv2d(in_channels = channels_img, out_channels = features_d, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2),
            # (batch x features_d x 32 x 32) -> (batch x features_d*2 x 16 x 16)
            nn.Conv2d(in_channels = features_d, out_channels = features_d * 2, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = features_d * 2),  # to keep stablelize (normalize)
            nn.LeakyReLU(0.2),
            # (batch x features_d*2 x 16 x 16) -> (batch x features_d*4 x 8 x 8)
            nn.Conv2d(in_channels = features_d * 2, out_channels = features_d * 4, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = features_d * 4),
            nn.LeakyReLU(0.2),
            # (batch x features_d*4 x 8 x 8) -> (batch x features_d*8 x 4 x 4)
            nn.Conv2d(in_channels = features_d * 4, out_channels = features_d * 8, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = features_d * 8),
            nn.LeakyReLU(0.2),

            # (batch x features_d*8 x 4 x 4) -> (batch x 1 x 1 x 1)
            nn.Conv2d(in_channels = features_d * 8, out_channels = 1, kernel_size = 4, stride = 2, padding = 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # (batch x channel_imgs x 64 x 64) -> (batch x 1 x 1 x 1)
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()

        # transposeconv2d size: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

        self.net = nn.Sequential(
            # (batch x channel_noise x 1 x 1) -> (batch x features_g x 4 x 4)
            nn.ConvTranspose2d(in_channels = channels_noise, out_channels = features_g * 16, kernel_size = 4, stride = 1, padding = 0),
            nn.BatchNorm2d(num_features = features_g * 16),
            nn.ReLU(),
            # (batch x features_g x 4 x 4) -> (batch x features_g*16 x 8 x 8)
            nn.ConvTranspose2d(in_channels = features_g * 16, out_channels = features_g * 8, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = features_g * 8),
            nn.ReLU(),
            # (batch x features_g*16 x 8 x 8) -> (batch x features_g*4 x 16 x 16)
            nn.ConvTranspose2d(in_channels = features_g * 8, out_channels = features_g * 4, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = features_g * 4),
            nn.ReLU(),
            # (batch x features_g*4 x 16 x 16) -> (batch x features_g*2 x 32 x 32)
            nn.ConvTranspose2d(in_channels = features_g * 4, out_channels = features_g * 2, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = features_g * 2),
            nn.ReLU(),

            # (batch x features_g*2 x 32 x 32) -> (batch x channels_img x 64 x 64)
            nn.ConvTranspose2d(in_channels = features_g * 2, out_channels = channels_img, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh(),
        )

    def forward(self, x):
        # (batch x channel_noise x 1 x 1) -> (batch x channels_img x 64 x 64)
        return self.net(x)


# Hyperparameters:
lr = 0.0002
batch_size = 64
image_size = 64 # 28x28 --> 64x64
channels_img = 1
channels_noise = 256
num_epochs = 10

features_d = 16
features_g = 16

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,)),
    ]
)

dataset = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# create model
netD = Discriminator(channels_img, features_d).to(device)
netG = Generator(channels_noise, channels_img, features_g).to(device)

# optimizer for G and D
optimizerD = optim.Adam(netD.parameters(), lr = lr, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = lr, betas = (0.5, 0.999))

netD.train()
netG.train()

criterion = nn.BCELoss()

real_label = 1
fake_label = 0

fixed_noise = torch.randn(batch_size, channels_noise, 1, 1).to(device)
writer_real = SummaryWriter(f"runs/GAN_MNIST/test_real")
writer_fake = SummaryWriter(f"runs/GAN_MNIST/test_fake")


def main():
    step = 0

    for epoch in range(num_epochs):
        loop = tqdm(enumerate(dataloader), leave = False, total = len(dataloader))
        for batch_idx, (data, targets) in loop:
            # data, targets: (batch_size, channel=1, img_width, img_height)
            data = data.to(device)
            #targets = targets.to(device)
            batch_size = data.shape[0]

            # train discriminator: max: log(D(x)) + log(1 - D(G(z))) given G(z) as constant
            netD.zero_grad()
            real_label = (torch.ones(batch_size) * 0.9).to(device)
            real_output = netD(data).reshape(-1)  # flatten

            lossD_real = criterion(real_output, real_label)


            D_x = real_output.mean().item()

            noise = torch.randn(batch_size, channels_noise, 1, 1).to(device)
            # generate fake data from noise through netG,
            fake_data = netG(noise)  # (batch x channels_img x 64 x 64)
            #fake_label = torch.zeros(batch_size).to(device)
            fake_label = (torch.ones(batch_size) * 0.1).to(device)
            fake_output = netD(fake_data.detach()).reshape(-1)  # flatten, detach = fix netG as constant, dont calculate grad
            lossD_fake = criterion(fake_output, fake_label)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            # train generator: max log(D(G(z)))
            netG.zero_grad()
            real_label = torch.ones(batch_size).to(device)
            output = netD(fake_data).reshape(-1)  # flatten, fake_data is output from netG

            lossG = criterion(output, real_label)
            lossG.backward()
            optimizerG.step()

            # update progress bar
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")

            # summary writer
            if batch_idx % 100 == 0:
                step += 1
                print(f"Loss D: {lossD: .4f}, Loss G: {lossG: .4f}, D(x): {D_x: .4f}")

                with torch.no_grad():
                    fake = netG(fixed_noise)

                    img_grid_real = torchvision.utils.make_grid(data[:32], normalize = True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize = True)
                    writer_real.add_image("MNIST Real Images", img_grid_real, global_step = step)
                    writer_fake.add_image("MNIST Fake Images", img_grid_fake, global_step = step)


if __name__ == '__main__':
    main()