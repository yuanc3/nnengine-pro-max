import tqdm
from lightGE.core import Tensor,TcGraph
from lightGE.utils import SGD,Adam, Trainer, nll_loss
import numpy as np
from lightGE.data import DataLoader
from lightGE.data.dataLoaders.cifar100_dataloader import CIFAR100Dataset
from lightGE.data.dataLoaders.cifar10_dataloader import CIFAR10Dataset
from lightGE.data.dataLoaders.mnist_dataloader import MnistDataset

from lightGE.models.ViT import ViT
from lightGE.models.ResNet import ResNet_18
def evaluate(model, dataset, batch_size=128):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    bar = tqdm.tqdm(dataloader)
    i = 0
    for x, y in bar:
        x = Tensor(x).to(model.device)
        y_pred = model(x)
        y_pred.to('cpu')
        y_pred = np.argmax(y_pred.data, axis=1)
        
        y = np.argmax(y.data, axis=1)
        correct += np.sum(y_pred == y)
        i += batch_size
        
        bar.set_description("acc: {}".format(correct / i))
        TcGraph.Clear()
        
    return correct / len(dataset)

if __name__ == '__main__':
    data = 'mnist' #change here
    
    if data == 'mnist':
        mnist_dataset = MnistDataset()
        mnist_dataset.load_data('./res/mnist/')
        train_dataset, eval_dataset = mnist_dataset.split(0.7)
        m = ViT(
            image_size = 28,
            patch_size = 4,
            num_classes = 10,
            dim = 64,
            channels = 1,
            depth = 1,
            heads = 4,
            mlp_dim = 128,
            dropout = 0.1,
            emb_dropout = 0.1,
        )
        cache_path = 'tmp/mnist_vit.pkl'
    if data == 'cifar10':
        train_dataset = CIFAR10Dataset(True)
        train_dataset.load_data('./res/cifar-10-batches-py/')
        eval_dataset =  CIFAR10Dataset(False)
        eval_dataset.load_data('./res/cifar-10-batches-py/')
        m = ViT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 128,
            channels = 3,
            depth = 3,
            heads = 4,
            mlp_dim = 256,
            dropout = 0.1,
            emb_dropout = 0.1,
            pool = 'cls'
        )
        cache_path = 'tmp/cifar10_vit.pkl'
    if data == 'cifar100':
        train_dataset = CIFAR100Dataset(True)
        train_dataset.load_data('./res/cifar-100-python/cifar-100-python/')
        eval_dataset = CIFAR100Dataset(False)
        eval_dataset.load_data('./res/cifar-100-python/cifar-100-python/')
        m = ResNet_18(num_classes=100, in_channels=3)
        cache_path = 'tmp/cifar100_resnet18.pkl'
    
    device = 'gpu'
    m.to(device)
    opt = Adam(parameters=m.parameters(), lr=0.01)

    # cache_path = 'tmp/cifar100_resnet.pkl'
    trainer = Trainer(model=m, optimizer=opt, loss_fun=nll_loss,
                      config={'batch_size': 64,
                              'epochs': 10,
                              'shuffle': True,
                              'save_path': cache_path})

    # trainer.load_model(cache_path)
    trainer.train(train_dataset, eval_dataset)
    evaluate(trainer.m, eval_dataset, batch_size=64)
