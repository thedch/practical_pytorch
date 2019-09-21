from ImagenetteDataset import ImagenetteDataset
from arch import ConvNet
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
from torch import nn
from helpers import assert_shapes
from hooks import Hooks


def main():
    train_dataset = ImagenetteDataset(training=True)
    val_dataset = ImagenetteDataset(training=False)
    bs = 64
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=bs)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=bs)

    net = ConvNet(in_ch=3).cuda()
    hooks = Hooks(net)
    optim = torch.optim.SGD(net.parameters(), lr=0.5)
    lossfxn = nn.CrossEntropyLoss()

    for epoch in range(5):
        pbar = tqdm(train_dataloader)
        for batch in pbar:
            optim.zero_grad()
            imgs = batch['image'].cuda()
            labels = batch['label'].cuda()
            preds = net(imgs)

            loss = lossfxn(preds, labels)
            pbar.set_postfix({'Loss': float(loss)})

            loss.backward()
            optim.step()

        hooks.show_me()

        pbar = tqdm(val_dataloader)
        total = 0
        correct = 0
        for batch in pbar:
            imgs = batch['image'].cuda()
            labels = batch['label'].cuda()
            with torch.no_grad():
                preds = net(imgs)

            assert_shapes(labels, ['bs'], preds, ['bs', 10])
            total += labels.numel()
            correct += (labels == torch.argmax(preds, dim=1)).sum().item()

        print(f'Top 1 accuracy is {correct/total}')



if __name__ == '__main__':
    main()
