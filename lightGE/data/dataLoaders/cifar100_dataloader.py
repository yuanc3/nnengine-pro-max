import numpy as np
import pickle
import os
from lightGE.data.dataloader import Dataset
import urllib.request
import tarfile

class CIFAR100Dataset(Dataset):
    def __init__(self, train = True):
        super(CIFAR100Dataset, self).__init__()
        self.train = train

    # download cifar-10 dataset
    def download_and_extract(self, url, dest_path):
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        filename = url.split('/')[-1]
        filepath = os.path.join(dest_path, filename)
        if not os.path.exists(filepath):
            print(f'Downloading {filename}...')
            urllib.request.urlretrieve(url, filepath)
            print(f'Extracting {filename}...')
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(path=dest_path)
                members = tar.getmembers()
                top_level_folder = os.path.commonpath([m.name for m in members if m.type != tarfile.DIRTYPE])
                return top_level_folder

    def load_data(self, data_dir):
        # URL to download CIFAR-100 dataset
        cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

        file_list = ['train', 'test', 'meta']
        # Check if files exists, and download CIFAR-100 dataset if it doesn't
        download = False
        for file in file_list:
            if not os.path.exists(os.path.join(data_dir, file)):
                download = True
                break
        if download:
            extracted_folder = self.download_and_extract(cifar10_url, data_dir)
            data_dir = os.path.join(data_dir, extracted_folder)
            print(data_dir)

        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='latin1')
            return dict

        def convert_images(raw):
            raw_float = np.array(raw, dtype=float) / 255.0
            images = raw_float.reshape([-1, 3, 32, 32])
            return images


        if self.train:
            # Load training data
            train_data = []
            data_batch = unpickle(os.path.join(data_dir, f'train'))
            train_data = convert_images(data_batch['data'])
            train_labels = np.array(data_batch['fine_labels'])
            self.x  = train_data
            self.y = np.eye(100)[train_labels]
        else:
            # Load test data
            test_batch = unpickle(os.path.join(data_dir, 'test'))
            test_data = convert_images(test_batch['data'])
            test_labels = np.array(test_batch['fine_labels'])
            self.x = test_data
            self.y = np.eye(100)[test_labels]




