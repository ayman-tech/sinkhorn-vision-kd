"""CIFAR-10/100 loaders with the usual pad-crop + flip augmentation.

When val_fraction>0 we carve a held-out chunk off the training set for the
adaptive-Sinkhorn outer loop. The split uses a seeded generator so the
val set is identical across runs (and across different methods).
"""
import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as T
from typing import Tuple, Optional

# means/stds from the train sets (standard published values).
CIFAR10_MEAN  = (0.4914, 0.4822, 0.4465)
CIFAR10_STD   = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

CIFAR100_CLASSES=[
    "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle",
    "bicycle","bottle","bowl","boy","bridge","bus","butterfly","camel",
    "can","castle","caterpillar","cattle","chair","chimpanzee","clock",
    "cloud","cockroach","couch","crab","crocodile","cup","dinosaur",
    "dolphin","elephant","flatfish","forest","fox","girl","hamster",
    "house","kangaroo","keyboard","lamp","lawn_mower","leopard","lion",
    "lizard","lobster","man","maple_tree","motorcycle","mountain","mouse",
    "mushroom","oak_tree","orange","orchid","otter","palm_tree","pear",
    "pickup_truck","pine_tree","plain","plate","poppy","porcupine",
    "possum","rabbit","raccoon","ray","road","rocket","rose","sea",
    "seal","shark","shrew","skunk","skyscraper","snail","snake","spider",
    "squirrel","streetcar","sunflower","sweet_pepper","table","tank",
    "telephone","television","tiger","tractor","train","trout","tulip",
    "turtle","wardrobe","whale","willow_tree","wolf","woman","worm"]

CIFAR10_CLASSES=["airplane","automobile","bird","cat","deer",
                 "dog","frog","horse","ship","truck"]


def get_cifar_loaders(dataset="cifar100", data_dir="./data", batch_size=128,
                      num_workers=4, pin_memory=True, val_fraction=0.0, seed=42
                      ) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    if dataset=="cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        DS=torchvision.datasets.CIFAR10
    elif dataset=="cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        DS=torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    tx_train=T.Compose([T.RandomCrop(32, padding=4),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize(mean, std)])
    tx_test=T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train=DS(root=data_dir, train=True,  download=True, transform=tx_train)
    test =DS(root=data_dir, train=False, download=True, transform=tx_test)

    val_loader=None
    if val_fraction>0:
        n=len(train)
        nv=int(n*val_fraction); nt=n-nv
        gen=torch.Generator().manual_seed(seed)
        train, val = random_split(train, [nt, nv], generator=gen)
        val_loader=DataLoader(val, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)

    train_loader=DataLoader(train, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader =DataLoader(test,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def get_class_names(dataset):
    if dataset=="cifar10":  return CIFAR10_CLASSES
    if dataset=="cifar100": return CIFAR100_CLASSES
    raise ValueError(f"unknown dataset {dataset!r}")
