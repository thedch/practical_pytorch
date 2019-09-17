from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor, Compose, CenterCrop, Lambda

def get_fnames(path):
    fnames = []
    classes = set()
    for _dir in list(path.iterdir()):
        for fname in list(_dir.iterdir()):
            assert fname.suffix == '.JPEG', fname
            classes.add(fname.parent.name)
            fnames.append(fname)

    classes = sorted(list(classes))
    mapping = {k: v for v, k in enumerate(classes)}

    return fnames, mapping


class ImagenetteDataset:
    def __init__(self, training:bool):
        basepath = Path('/Users/daniel/.fastai/data/imagenette-160')
        if training:
            path = basepath/'train'
        else:
            path = basepath/'val'

        self.fnames, self.mapping = get_fnames(path)
        self.tfms = Compose([
            CenterCrop(128),
            Lambda(lambda img: img.convert('RGB')),
            ToTensor(),
        ])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img = Image.open(self.fnames[idx])
        label = self.mapping[self.fnames[idx].parent.name] # use folder names as labels
        tensor = self.tfms(img)

        return {
            'image': tensor,
            'label': label,
        }
