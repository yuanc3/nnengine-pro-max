from lightGE.data.dataloader import Dataset
import os
import gzip
import numpy as np
import urllib.request

class MnistDataset(Dataset):
    def __init__(self):
        super(MnistDataset, self).__init__()

    #下载并解压MNIST数据集
    def download_mnist(self, data_dir, urls):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        for url in urls:
            filename = url.split('/')[-1]
            filepath = os.path.join(data_dir, filename)

            if not os.path.isfile(filepath):
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)

            # with gzip.open(filepath, 'rb') as f_in:
            #     with open(filepath.replace('.gz', ''), 'wb') as f_out:
            #         f_out.write(f_in.read())
    def load_data(self, data_dir):
        def extract_data(filename, num_data, head_size, data_size):
            with gzip.open(filename) as bytestream:
                bytestream.read(head_size)
                buf = bytestream.read(data_size * num_data)
                data = np.frombuffer(buf, dtype=np.uint8).astype(np.float64)
            return data

        mnist_urls = [
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
        ]

        # 检查MNIST数据是否已下载，如果没有，则下载
        self.download_mnist(data_dir, mnist_urls)
        # 使用 os.path.join 来拼接路径
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')

        data = extract_data(train_images_path, 60000, 16, 28 * 28)
        trX = data.reshape((60000, 28, 28, 1))

        data = extract_data(train_labels_path, 60000, 8, 1)
        trY = data.reshape((60000))

        data = extract_data(test_images_path, 10000, 16, 28 * 28)
        teX = data.reshape((10000, 28, 28, 1))

        data = extract_data(test_labels_path, 10000, 8, 1)
        teY = data.reshape((10000))

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        x = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int32)

        data_index = np.arange(x.shape[0])
        np.random.shuffle(data_index)
        # data_index = data_index[:128]
        x = x[data_index, :, :, :]
        y = y[data_index]
        y_vec = np.zeros((len(y), 10), dtype=np.float64)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0

        x /= 255.
        x = x.transpose(0, 3, 1, 2)
        self.x = x
        self.y = y_vec

