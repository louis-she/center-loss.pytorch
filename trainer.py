import os

import torch

from device import device
from loss import compute_center_loss, get_center_delta

class Trainer(object):

    def __init__(self, optimizer, model, training_dataloader, validation_dataloader, log_dir=False,
            max_epoch=100, resume=False, persist_stride=10, persist_best=True, lamda=0.0003, alpha=0.5):
        self.log_dir = log_dir
        self.optimizer = optimizer
        self.model = model
        self.max_epoch = max_epoch
        self.resume = resume
        self.persist_stride = persist_stride
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.training_losses = {'center': [], 'cross_entropy': [], 'total': []}
        self.validation_losses = {'center': [], 'cross_entropy': [], 'total': []}
        self.current_epoch = 1
        self.lamda = lamda
        self.alpha = alpha

        if not self.log_dir:
            self.log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        if resume:
            #TODO: add resume functionality
            pass

    def train(self):
        for self.current_epoch in range(1, self.max_epoch):
            self.train_epoch()
            self.validate_epoch()
            if not (self.current_epoch % self.persist_stride):
                self.persist()

    def train_epoch(self):
        print("start train epoch {}".format(self.current_epoch))
        total_cross_entropy_loss = 0
        total_center_loss = 0
        total_loss = 0
        for images, targets, names in self.training_dataloader:
            targets = torch.tensor(targets).to(device)
            images = images.to(device)
            centers = self.model.centers

            logits, features = self.model(images)

            cross_entropy_loss = torch.nn.functional.cross_entropy(logits, targets)
            center_loss = compute_center_loss(features, centers, targets, self.lamda)
            loss = cross_entropy_loss + center_loss
            print("cross entropy loss: {} - center loss: {} - total loss: {}".format(cross_entropy_loss, center_loss, loss))

            total_cross_entropy_loss += cross_entropy_loss
            total_center_loss += center_loss
            total_loss += loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # make features untrack by autograd, or there will be a memory
            # leak when updating the centers
            center_deltas = get_center_delta(features.data, centers, targets, self.alpha)
            self.model.centers = centers - center_deltas

        self.training_losses['center'].append(total_center_loss)
        self.training_losses['cross_entropy'].append(total_cross_entropy_loss)
        self.training_losses['total'].append(total_loss)
        print("epoch {} training finished. cross entropy loss: {} - center loss: {} - total loss: {}"
                .format(self.current_epoch, total_cross_entropy_loss, total_center_loss, total_loss))

    def validate_epoch(self):
        print("start validate epoch {}".format(self.current_epoch))
        #TODO: add validation process
        print("epoch {} validation finished.".format(self.current_epoch))

        # True indicates that this epoch is the best out of all epoches
        return True

    def persist(self, is_best=False):
        model_dir = os.path.join(self.log_dir, 'models')
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        file_name = ("epoch_{}_best.pth.tar" if is_best else "epoch_{}.pth.tar").format(self.current_epoch)

        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses
        }
        state_path = os.path.join(model_dir, file_name)
        torch.save(state, state_path)