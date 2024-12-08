import pickle
from tqdm import tqdm
import numpy as np
try:
    import cupy as cp
except:
    cp = np
    
from lightGE.data import DataLoader
from lightGE.core.tensor import Tensor, TcGraph
import logging
import gc
import sys

logging.basicConfig(level=logging.INFO)


def del_tensor(t: Tensor):
    if t.creation_op is not None:
        for tt in t.creation_op.input:
            del_tensor(tt)
    del t
    try:
        print("!", sys.getrefcount(t))
    except:
        pass
class Trainer(object):

    def __init__(self, model, optimizer, loss_fun, config, schedule=None):
        self.m = model
        self.opt = optimizer
        self.sche = schedule
        self.lf = loss_fun

        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.shuffle = config['shuffle']
        self.save_path = config['save_path']

    def train(self, train_dataset, eval_dataset):
        train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle=self.shuffle)
        eval_dataloader = DataLoader(eval_dataset, self.batch_size, shuffle=self.shuffle)

        min_eval_loss, best_epoch = float('inf'), 0
        for epoch_idx in range(self.epochs):
            self._train_epoch(train_dataloader)
            eval_loss = self._eval_epoch(eval_dataloader)

            if self.sche is not None:
                self.sche.step(eval_loss)

            logging.info("Lr: {}".format(self.opt.lr))
            # self.save_model(self.save_path)
            if eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                self.save_model(self.save_path)
                best_epoch = epoch_idx

        logging.info("Best epoch: {}, Best validation loss: {}".format(best_epoch, min_eval_loss))

        return min_eval_loss

    def _train_epoch(self, train_dataloader) -> [float]:
        self.m.train()
        losses = []

        bar = tqdm(train_dataloader)
        batch_idx = 0

        for batch_x, batch_y in bar:
            batch_idx += 1
            y_truth = Tensor(batch_y, autograd=False).to(self.m.device)
            input = Tensor(batch_x, autograd=False).to(self.m.device)

            y_pred = self.m(input)
            loss: Tensor = self.lf(y_pred, y_truth)
            loss.backward()
            self.opt.step()
 
            losses.append(loss.data)
            if self.m.device == "cpu":
                bar.set_description("Batch: {}/{} ".format(batch_idx, len(train_dataloader)) +
                                    'Training loss: {},'.format(np.mean(losses)))
            else:
                bar.set_description("Batch: {}/{} ".format(batch_idx, len(train_dataloader)) +
                                    'Training loss: {},'.format(cp.mean(cp.array(losses))))
            TcGraph.Clear()
            gc.collect()
        return

    def _eval_epoch(self, eval_dataloader):
        self.m.eval()
        losses = []
        bar = tqdm(eval_dataloader)
        batch_idx = 0
        for batch_x, batch_y in bar:
            batch_idx += 1
            y_truth = Tensor(batch_y, autograd=False).to(self.m.device)
            input = Tensor(batch_x, autograd=False).to(self.m.device)
            y_pred = self.m(input)
            loss: Tensor = self.lf(y_pred, y_truth)
            losses.append(loss.data)
            TcGraph.Clear()
            if self.m.device == "cpu":
                bar.set_description("Batch: {}/{} ".format(batch_idx, len(eval_dataloader)) +
                                    'Validation loss: {},'.format(np.mean(losses)))
            else:
                bar.set_description("Batch: {}/{} ".format(batch_idx, len(eval_dataloader)) +
                                    'Validation loss: {},'.format(cp.mean(cp.array(losses))))
        if self.m.device == "cpu":
            logging.info("Validation loss: {}".format(np.mean(losses)))
            return np.mean(losses)
        else:
            logging.info("Validation loss: {}".format(cp.mean(cp.array(losses))))
            return cp.mean(cp.array(losses))

        

    def load_model(self, cache_name):
        [self.m, self.opt, self.sche] = pickle.load(open(cache_name, 'rb'))

    def save_model(self, cache_name):
        pickle.dump([self.m, self.opt, self.sche], open(cache_name, 'wb'))
