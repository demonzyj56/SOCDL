"""Import all loaders."""
from .cifar import CIFAR10Loader, CIFAR100Loader
from .image_dataset import FruitLoader, CityLoader, ImageLoader
from .vgg_flowers import VGG17FlowersLoader, VGG102FlowersLoader
from .voc07 import VOC07Loader

__all__ = [
    'CIFAR10Loader', 'CIFAR100Loader',
    'FruitLoader', 'CityLoader', 'ImageLoader',
    'VGG17FlowersLoader', 'VGG102FlowersLoader',
    'VOC07Loader'
]
