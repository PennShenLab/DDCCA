import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler

import time
import logging
try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle
import os

torch.set_default_tensor_type(torch.DoubleTensor)


class Solver_opt():
    def __init__(self, trial, model, linear_cca, outdim_size, epoch_num, batch_size, learning_rate, reg_par,
                 device=torch.device('cpu'), optimizer_name='Adam'):
        self.model = nn.DataParallel(model)
        self.model.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.loss = model.loss
        self.optimizer = getattr(torch.optim, optimizer_name)(
            self.model.parameters(), lr=learning_rate, weight_decay=reg_par)
        self.device = device

        self.linear_cca = linear_cca

        self.outdim_size = outdim_size

        formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
        logging.basicConfig(
            level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Pytorch")
        fh = logging.FileHandler(f"Solver_opt_{os.getpid()}.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(self.model)
        self.logger.info(self.optimizer)


    def fit(self, x1, x2, vx1=None, vx2=None, checkpoint='_checkpoint.model', emb=None):
        """
        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]
        """
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        data_size = x1.size(0)

        if vx1 is not None and vx2 is not None:
            best_val_loss = 1.0
            vx1 = vx1.to(self.device)
            vx2 = vx2.to(self.device)

        train_losses = []
        early_stopping = EarlyStopping(patience=150)
        for epoch in range(self.epoch_num):
            epoch_start_time = time.time()
            self.model.train()
            batch_idxs = list(BatchSampler(RandomSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            for batch_idx in batch_idxs:
                self.optimizer.zero_grad()
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                if emb is not None:  # for DDCCA
                    o1, o2 = self.model(batch_x1, batch_x2, emb.to(self.device))
                else:  # for DCCA
                    o1, o2 = self.model(batch_x1, batch_x2)
                loss = self.loss(o1, o2, self.model.module.get_w())
                train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            train_loss = np.mean(train_losses)
            info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f}"
            if vx1 is not None and vx2 is not None:
                with torch.no_grad():
                    self.model.eval()
                    val_loss = self.test(vx1, vx2, emb=emb)
                    info_string += " - val_loss: {:.4f}".format(val_loss)
                    if val_loss < best_val_loss:
                        self.logger.info(
                            "Epoch {:d}: val_loss improved from {:.4f} to {:.4f}, saving model to {}".format(epoch + 1, best_val_loss, val_loss, checkpoint))
                        best_val_loss = val_loss
                        torch.save({'state_dict':self.model.state_dict(), 'w':self.model.module.get_w()}, checkpoint)
                    else:
                        self.logger.info("Epoch {:d}: val_loss did not improve from {:.4f}".format(
                            epoch + 1, best_val_loss))

                    if early_stopping(val_loss):
                        self.logger.info(f'Early Stopping, since val_loss has not improved for {early_stopping.patience} epochs')
                        break
            else:
                torch.save({'state_dict':self.model.state_dict(), 'w':self.model.module.get_w()}, checkpoint)
            epoch_time = time.time() - epoch_start_time
            self.logger.info(info_string.format(
                epoch + 1, self.epoch_num, epoch_time, train_loss))

        checkpoint_ = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_['state_dict'])
        self.model.module.set_w(checkpoint_['w'])
        torch.save({'emb':emb, 'state_dict':self.model.state_dict(), 'w':self.model.module.get_w()}, checkpoint)

        if vx1 is not None and vx2 is not None:
            self.logger.info("loss on validation data: {:.4f}".format(best_val_loss))
            return best_val_loss


    def test(self, x1, x2, use_linear_cca=False, emb=None):
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        with torch.no_grad():
            losses, outputs = self._get_outputs(x1, x2, emb=emb)

            if use_linear_cca:
                print("Linear CCA started!")
                outputs = self.linear_cca.test(outputs[0], outputs[1])
                return np.mean(losses), outputs
            else:
                return np.mean(losses)


    def _get_outputs(self, x1, x2, emb=None):
        """Get the outputs from the trained nn model"""
        with torch.no_grad():
            self.model.eval()
            data_size = x1.size(0)
            batch_idxs = list(BatchSampler(SequentialSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            losses = []
            outputs1 = []
            outputs2 = []
            for batch_idx in batch_idxs:
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                if emb is not None:  # for DDCCA
                    o1, o2 = self.model(batch_x1, batch_x2, emb.to(self.device))
                else:  # for DCCA
                    o1, o2 = self.model(batch_x1, batch_x2)
                outputs1.append(o1)
                outputs2.append(o2)
                loss = self.loss(o1, o2, self.model.module.get_w(), training=False)
                losses.append(loss.item())
        outputs = [torch.cat(outputs1, dim=0).to('cpu').numpy(),
                   torch.cat(outputs2, dim=0).to('cpu').numpy()]
        return losses, outputs


    def load_from_checkpt_dict(self, checkpt_dict):
        self.model.load_state_dict(checkpt_dict['state_dict'])
        self.model.module.set_w(checkpt_dict['w'])



class EarlyStopping:
    '''
    A callable class to determine if to early stop or not
    '''
    def __init__(self, patience=150, min_delta=0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = 1.0

    def __call__(self, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False