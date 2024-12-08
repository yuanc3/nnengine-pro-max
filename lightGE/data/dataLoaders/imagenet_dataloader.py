import numpy as np
import tqdm
from lightGE.core.nn import Model, Sequential, ReLU, Conv2d, MaxPool2d, Linear
from lightGE.core import Tensor
from lightGE.data.dataloader import Dataset, DataLoader
from lightGE.utils import Adam, Trainer, crossEntropyLoss
from lightGE.models.ViT import ViT
import os
import numpy as np
from PIL import Image

# 未测试，待完善~
# 数据加载方式未重写
class ImageNetDataset(Dataset):
    def __init__(self):
        super(ImageNetDataset, self).__init__()

    def load_data(self, data_dir):
        # 代码来加载和处理ImageNet数据集
        # 需要根据实际的lightGE库的API和ImageNet数据集的存储格式编写
        # 关于图像处理的部分，之后可以移到data_preprocess里
        self.x = []
        self.y = []

        # 假设每个类别的数据存放在单独的子目录中
        for label, class_dir in enumerate(sorted(os.listdir(data_dir))):
            class_path = os.path.join(data_dir, class_dir)
            if not os.path.isdir(class_path):
                continue

            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((224, 224))  # 将图片缩放到224x224
                img_array = np.asarray(img) / 255.0  # 归一化

                self.x.append(img_array)
                self.y.append(label)

        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int32)
        self.y = np.eye(1000)[self.y]  # one-hot编码

        # 打乱数据集
        indices = np.arange(len(self.x))
        np.random.shuffle(indices)
        self.x = self.x[indices]
        self.y = self.y[indices]

class ImageNetViT(Model):
    def __init__(self):
        super(ImageNetViT, self).__init__()
        self.model = ViT(
            image_size=224,  # ViT对于ImageNet通常使用更大的图片尺寸
            patch_size=16,
            num_classes=1000,  # ImageNet具有1000个类别
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            pool='cls'
        )

    def forward(self, x: Tensor):
        return self.model(x)

def evaluate(model, dataset):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    correct = 0
    bar = tqdm.tqdm(dataloader)
    for x, y in bar:
        x = Tensor(x).to(model.device)
        y_pred = model(x)
        y_pred.to('cpu')
        y_pred = np.argmax(y_pred.data, axis=1)
        y = np.argmax(y.data, axis=1)
        correct += np.sum(y_pred == y)
        bar.set_description("acc: {}".format(correct / len(dataset)))
    return correct / len(dataset)

if __name__ == '__main__':
    imagenet_dataset = ImageNetDataset()
    imagenet_dataset.load_data('./res/imagenet/')

    train_dataset, eval_dataset = imagenet_dataset.split(0.7)
    device = 'gpu'
    model = ImageNetViT()
    model.to(device)

    opt = Adam(parameters=model.parameters(), lr=0.003)
    cache_path = 'tmp/imagenet_vit.pkl'
    trainer = Trainer(model=model, optimizer=opt, loss_fun=crossEntropyLoss,
                      config={'batch_size': 64,  # 根据GPU内存调整
                              'epochs': 10,
                              'shuffle': True,
                              'save_path': cache_path})
    trainer.train(train_dataset, eval_dataset)
    evaluate(model, eval_dataset)