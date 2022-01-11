import os
import subprocess
from tqdm import tqdm

import torch
from torch.cuda import is_available
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image


device = 'cuda:0' if is_available() else 'cpu'

def prepare_data() -> DataLoader:
    if not os.path.exists(os.path.join('./data', 'lfw-deepfunneled')):
        subprocess.run(args=[
            'wget',
            '-P',
            './data',
            'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz'
        ])

        subprocess.run(args=[
            'tar',
            'xf',
            os.path.join('./data', 'lfw-deepfunneled.tgz'),
            '-C',
            './data'
        ])
    else:
        print('Files already downloaded and verified')

    img_data = ImageFolder(
        './data/lfw-deepfunneled',
        transform=transforms.Compose([
            transforms.Resize(80),
            transforms.CenterCrop(64),        
            transforms.ToTensor()                  
    ]))
    img_loader = DataLoader(img_data, batch_size=64, shuffle=True)
    return img_loader


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        nz = 100
        ngf = 32
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
  
    def forward(self, x):
        out = self.main(x)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        ndf = 32
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )
  
    def forward(self, x):
        out = self.main(x)
        return out.squeeze()


disctiminator = Discriminator().to(device)
generator = Generator().to(device)

disctiminator.load_state_dict(torch.load('./results/generated/lfw/d_004.prm'))
generator.load_state_dict(torch.load('./results/generated/lfw/g_004.prm'))

g_optimizer = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = Adam(disctiminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

ones = torch.ones(64).to(device)
zeros = torch.zeros(64).to(device)
criterion = nn.BCEWithLogitsLoss()

fixed = torch.randn(64, 100, 1, 1).to(device)


def _train(generator, discriminator, g_optimizer, d_optimizer, data_loader) -> list[float]:
    i = 0
    g_running_loss = 0.0
    d_running_loss = 0.0
    for real_img, _ in tqdm(data_loader):
        batch_len = len(real_img)

        real_img = real_img.to(device)
        z = torch.randn(batch_len, 100, 1, 1).to(device)
        fake_img = generator(z)

        fake_img_tensor = fake_img.detach()
        out = discriminator(fake_img)
        g_loss = criterion(out, ones[:batch_len])
        g_running_loss += g_loss.item()

        # generatorの学習.
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        real_out = discriminator(real_img)
        d_loss_real = criterion(real_out, ones[:batch_len])

        fake_img = fake_img_tensor
        fake_out = discriminator(fake_img_tensor)
        d_loss_fake = criterion(fake_out, zeros[:batch_len])

        d_loss = d_loss_real + d_loss_fake
        d_running_loss += d_loss.item()

        # discriminatorの学習.
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        i += 1
    return [g_running_loss / i, d_running_loss / i]


def _eval(generator: nn.Module) -> list[float]:
    generated_img: torch.Tensor = generator(fixed)
    save_image(generated_img, './results/generated/lfw/{:03d}.jpg'.format(epoch))
    return []


img_loader = prepare_data()
for epoch in range(100):
    _train(generator, disctiminator, g_optimizer, d_optimizer, img_loader)
    _eval(generator)

    torch.save(generator.state_dict(), './results/generated/lfw/g_{:03d}.prm'.format(epoch), pickle_protocol=4)
    torch.save(disctiminator.state_dict(), './results/generated/lfw/d_{:03d}.prm'.format(epoch), pickle_protocol=4)
