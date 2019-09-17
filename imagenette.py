from ImagenetteDataset import ImagenetteDataset
from arch import ConvNet
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
    train_dataset = ImagenetteDataset(training=True)
    val_dataset = ImagenetteDataset(training=False)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=16)

    net = ConvNet(in_ch=3)

    for batch in tqdm(train_dataloader):
        pass

if __name__ == '__main__':
    main()
