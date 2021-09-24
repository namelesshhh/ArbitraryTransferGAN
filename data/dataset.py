# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data
import torchvision.datasets as dset

class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            imgs.append(line)
            print(line)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        fn = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    image_size = 64
    dataset = dset.ImageFolder(root=r'G:\crops',
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                             shuffle=True, num_workers=0)
    ngpu = 0
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Plot some training images
    real_batch = next(iter(dataloader))
    print('real_batch: ', len(real_batch[0]))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig('G:\mydata.jpg')