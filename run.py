import argparse
import tqdm
from lightGE.core import Tensor, TcGraph
from lightGE.utils import SGD, Adam, Trainer, nll_loss
import numpy as np
from lightGE.data import DataLoader
from lightGE.data.dataLoaders.cifar100_dataloader import CIFAR100Dataset
from lightGE.data.dataLoaders.cifar10_dataloader import CIFAR10Dataset
from lightGE.data.dataLoaders.mnist_dataloader import MnistDataset

from lightGE.models.ViT import ViT
from lightGE.models.ResNet import ResNet_18, ResNet_50
from train import evaluate

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'cifar10', 'cifar100'], type=str, help='load data from dataset')
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'resnet50', 'vit'], type=str, help='chose a model')
    parser.add_argument('--checkpoint', default='./tmp', type=str, help='checkpoint dir')
    parser.add_argument('--device', default='gpu', type=str, help='device mode')

    args = parser.parse_args()

    if args.dataset == 'mnist':
        mnist_dataset = MnistDataset()
        mnist_dataset.load_data('./res/mnist/')
        train_dataset, eval_dataset = mnist_dataset.split(0.7)
    if args.dataset == 'cifar10':
        train_dataset = CIFAR10Dataset(True)
        train_dataset.load_data('./res/cifar-10-batches-py/')
        eval_dataset = CIFAR10Dataset(False)
        eval_dataset.load_data('./res/cifar-10-batches-py/')
    if args.dataset == 'cifar100':
        train_dataset = CIFAR100Dataset(True)
        train_dataset.load_data('./res/cifar-100-python/cifar-100-python/')
        eval_dataset =  CIFAR100Dataset(False)
        eval_dataset.load_data('./res/cifar-100-python/cifar-100-python/')

    cp = ''
    if args.model == 'resnet18':
        if args.dataset == 'cifar100':
            batch_size = 32
            m = ResNet_18(num_classes=100, in_channels=3)
            cp = '/cifar100_resnet18.pkl'
    if args.model == 'resnet50':
        batch_size = 8
        if args.dataset == 'mnist':
            m = ResNet_50(num_classes=10, in_channels=1)
            cp = '/mnist_resnet50.pkl'
    if args.model == 'vit':
        batch_size = 64
        if args.dataset == 'mnist':
            m = ViT(
                image_size=28,
                patch_size=4,
                num_classes=10,
                dim=64,
                channels=1,
                depth=1,
                heads=4,
                mlp_dim=128,
                dropout=0.1,
                emb_dropout=0.1,
            )
            cp = '/mnist_vit.pkl'
        if args.dataset == 'cifar10':
            m = ViT(
                image_size=32,
                patch_size=4,
                num_classes=10,
                dim=128,
                channels=3,
                depth=3,
                heads=4,
                mlp_dim=256,
                dropout=0.1,
                emb_dropout=0.1,
                pool='cls'
            )
            cp = '/cifar10_vit.pkl'

    cp_path = args.checkpoint
    cp_path += cp
    device = args.device
    m.to(device)
    opt = Adam(parameters=m.parameters(), lr=0.01)
    trainer = Trainer(model=m, optimizer=opt, loss_fun=nll_loss,
                      config={'batch_size': batch_size,
                              'epochs': 10,
                              'shuffle': True,
                              'save_path': cp_path})
    trainer.load_model(cp_path)
    evaluate(trainer.m, eval_dataset, batch_size=batch_size)

