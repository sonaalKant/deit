# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os,glob
import json
import pandas as pd
import pickle
import random
from PIL import Image
from torch.utils.data import Dataset

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

class ImagentSubset(Dataset):
    def __init__(self, root, transform) -> None:
        super().__init__()
        classnames = os.listdir(root)
        
        self.nb_classes = 100
        self.transform = transform

        if not os.path.isfile("imnet_class_info.pkl"):
            classnames = random.sample(classnames, self.nb_classes)
            class2idx = {cls : i for i,cls in enumerate(classnames) }
            pickle.dump(class2idx, open("imnet_class_info.pkl", "wb"))
        else:
            class2idx = pickle.load(open("imnet_class_info.pkl", "rb"))
            classnames = list(class2idx.keys())

        self.imnames = list()
        self.labels = list()

        for i,cls in enumerate(classnames):
            ims = glob.glob(os.path.join(root, cls + "/*"))
            lab = [class2idx[cls]]*len(ims)

            self.imnames += ims
            self.labels += lab

    def __len__(self):
        return len(self.imnames)
    
    def __getitem__(self, idx):
        img = self.imnames[idx]
        label = self.labels[idx]
        img = Image.open(img).convert('RGB')
        return self.transform(img), label


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        # dataset = datasets.ImageFolder(root, transform=transform)
        # nb_classes = 1000
        dataset = ImagentSubset(root,transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

if __name__ == '__main__':
    is_train = True
    data_path = "/fs/vulcan-datasets/imagenet/"
    data_path = os.path.join(data_path, 'train' if is_train else 'val')
    transform = []
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    transform = transforms.Compose(transform)

    
