import torch
import torch.nn.functional as F
import torchvision.transforms as tvtf
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_lightning as pl

from classifier import *


class NNLightning(pl.LightningModule):
    def __init__(self, model: nn.Module, threshold: int = 0.5):
        """
        Wraps the original model object with Torch Lightning Module
        """
        super().__init__()
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        self.threshold = threshold
        self.train_step_outputs = {'loss': [], 'accuracy': [], 'mod_accuracy': []}
        self.valid_step_outputs = {'loss': [], 'accuracy': [], 'mod_accuracy': []}
        self.test_outputs = {'loss': [], 'accuracy': [], 'mod_accuracy': []}

        
    def forward(self, x):
        x = self.model(x)
        
        return x
    
    def _common_step(self, batch, batch_idx):
        X, y = batch
        y_scores = self.forward(X) 
        loss = self.loss_fn(y_scores, y)
        
        return loss, y_scores, y
    
    def training_step(self, batch, batch_idx):
        loss, y_scores, y = self._common_step(batch, batch_idx)
        mod_acc = self._modified_accuracy(y_scores, y)
        acc = self._actual_accuracy(y_scores, y)
        
        self.log('train_loss', loss, sync_dist=True)
        
        self.train_step_outputs['loss'].append(loss)
        self.train_step_outputs['accuracy'].append(acc)
        self.train_step_outputs['mod_accuracy'].append(mod_acc)
        return loss
    
    def _actual_accuracy(self, scores, targets):
        """
        Computes actual accuracy for multi-label classification, using both true positives and negatives
        Input: 2 binary tensors 
        preds = [[1, 1, 0], [0, 1, 0]] -- targets = [[1, 0, 0], [1, 1, 0]] => accuracy = 4/6 
        """
        proba = F.sigmoid(scores)
        num_correct = ((proba > self.threshold) == targets).sum()
        num_preds = proba.size(0) * proba.size(1)
        
        acc = num_correct / num_preds
        
        return acc
        
    
    def _modified_accuracy(self, scores, targets):
        """
        Computes accuracy for multi-label classification, using only true positives
        Input: 2 binary tensors 
        preds = [[1, 1, 0], [0, 1, 0]] -- targets = [[1, 0, 0], [1, 1, 0]] => accuracy = 2/6 (only counts 1)
        """
        # Count number of TRUE POSITIVE predictions
        proba = F.sigmoid(scores)
        num_correct = ((proba > self.threshold) * targets).sum()

        # Total possible true positives 
        num_positives = (targets == 1).sum()
        
        if num_positives == 0:
            return 0

        # Compute accuracy
        accuracy = num_correct / num_positives

        return accuracy
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_step_outputs['loss']).mean()  
        
        # all_preds = torch.cat([x["preds"] for x in outputs])
        # all_targets = torch.cat([x["targets"] for x in outputs])
        
        acc = torch.stack(self.train_step_outputs['accuracy']).mean()
        mod_acc = torch.stack(self.train_step_outputs['mod_accuracy']).mean()
        
        self.log_dict({"train_loss": avg_loss, "train_acc": acc, 'train_mod_acc': mod_acc})
        print(f"TRAINING -- {avg_loss=} -- {acc=} -- {mod_acc=}")
    
    def validation_step(self, batch, batch_idx):
        loss, y_scores, y = self._common_step(batch, batch_idx)
        mod_acc = self._modified_accuracy(y_scores, y)
        acc = self._actual_accuracy(y_scores, y)
        
        self.log('valid_loss', loss, sync_dist=True)
        
        self.valid_step_outputs['loss'].append(loss)
        self.valid_step_outputs['accuracy'].append(acc)
        self.valid_step_outputs['mod_accuracy'].append(mod_acc)
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.valid_step_outputs['loss']).mean()  
        acc = torch.stack(self.valid_step_outputs['accuracy']).mean()
        mod_acc = torch.stack(self.valid_step_outputs['mod_accuracy']).mean()
        
        self.log_dict({"valid_loss": avg_loss, "valid_acc": acc, 'valid_mod_acc': mod_acc})
        print(f"VALIDATION -- {avg_loss=} -- {acc=} -- {mod_acc=}")
        
    def test_step(self, batch, batch_idx):
        loss, y_scores, y = self._common_step(batch, batch_idx)
        mod_acc = self._modified_accuracy(y_scores, y)
        acc = self._actual_accuracy(y_scores, y)
        
        self.log('test_loss', loss, sync_dist=True)
        
        self.test_step_outputs['loss'].append(loss)
        self.test_step_outputs['accuracy'].append(acc)
        self.test_step_outputs['mod_accuracy'].append(mod_acc)
        return loss
    
    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_step_outputs['loss']).mean()  
        acc = torch.stack(self.test_step_outputs['accuracy']).mean()
        mod_acc = torch.stack(self.test_step_outputs['mod_accuracy']).mean()
        
        self.log_dict({"test_loss": avg_loss, "test_acc": acc, 'test_mod_acc': mod_acc})
        print(f"TESTING -- {avg_loss=} -- {acc=} -- {mod_acc=}")
        
    
    def predict_step(self, batch, batch_idx):
        X, y = batch
        # preds = self.classify(X) # from MultiLabelClassifier model object
        y_scores = self.forward(X)
        y_proba = F.sigmoid(y_scores)
        preds = (y_proba >= self.threshold)
        
        return preds
    
    def configure_optimizers(self):
        return optim.SGD([
                {'params': list(self.model.parameters())[:-1], 'lr': 1e-4, 'momentum':0.9},
                {'params': list(self.model.parameters())[-1], 'lr': 5e-2, 'momentum': 0.9}
            ])
    















