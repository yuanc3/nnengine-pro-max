import numpy as np
import pickle
import os
from lightGE.data.dataloader import Dataset
import urllib.request
import tarfile

class CIFAR10Dataset(Dataset):
    def __init__(self, train = True):
        super(CIFAR10Dataset, self).__init__()
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
        # URL to download CIFAR-10 dataset
        cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

        file_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
        # Check if files exists, and download CIFAR-10 dataset if it doesn't
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
            train_labels = []
            for i in range(1, 6):
                data_batch = unpickle(os.path.join(data_dir, f'data_batch_{i}'))
                train_data.append(convert_images(data_batch['data']))
                train_labels += data_batch['labels']
            train_data = np.concatenate(train_data, axis=0)
            train_labels = np.array(train_labels)
            self.x = train_data
            self.y = train_labels
            self.y = np.eye(10)[self.y]  # Convert labels to one-hot encoding   
        else:
             # Load test data
            test_batch = unpickle(os.path.join(data_dir, 'test_batch'))
            test_data = convert_images(test_batch['data'])
            test_labels = np.array(test_batch['labels'])
            self.x = test_data
            self.y = test_labels
            self.y = np.eye(10)[self.y]
