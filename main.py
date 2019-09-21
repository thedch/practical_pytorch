from ImagenetteDataset import ImagenetteDataset
from arch import ConvNet
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
from torch import nn


def main():
    train_dataset = ImagenetteDataset(training=True)
    val_dataset = ImagenetteDataset(training=False)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=16)

    net = ConvNet(in_ch=3)
    optim = torch.optim.SGD(net.parameters(), lr=0.05)
    lossfxn = nn.CrossEntropyLoss()

    for batch in tqdm(train_dataloader):
        imgs = batch['image']
        labels = batch['label']
        preds = net(imgs)

        loss = lossfxn(preds, labels)
        print(loss.mean())
        optim.zero_grad()
        loss.backward()


if __name__ == '__main__':
    main()
