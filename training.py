import os
import abc
import sys
import torch
import torch.nn as nn
import tqdm.auto
from torch import Tensor
from typing import Any, Tuple, Callable, Optional, cast
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from train_results import FitResult, BatchResult, EpochResult

class Trainer(abc.ABC):
    '''
    A class abstracting various tasks of training models
    
    Provides methods at multiple levels of granularity:
        - Multiple epochs (fit)
        - Single epoch (train_epoch / test_epoch)
        - Single batch (train_batch / test_batch)
    '''
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.device = device
        
        if self.device:
            model = model.to(self.device)
            
    def fit(
        self,
        dl_train: DataLoader,
        dl_test: DataLoader,
        num_epochs: int,
        checkpoints: str = None,
        early_stopping: int = None,
        print_every: int = 1,
        **kw
    ) -> FitResult:
        
        train_loss, train_acc, test_loss, test_acc = [], [], [], [] # initialize metrics
        best_acc = None
        
        for epoch in range(num_epochs):
            verbose = False
            if print_every > 0 and (
                epoch % print_every == 0 or epoch == num_epochs
            ):
                verbose = True
            self._print(f'--- EPOCH {epoch+1}/{num_epochs} ---', verbose)
            
            # TRAIN
            train_result = self.train_epoch(dl_train, **kw) # EpochResult obj: losses: List[float], accuracy: float        http://localhost:8889/lab?token=25246ee22d2fc35d09171fcd46973e92523336543092466b

            train_loss += [l.item() for l in train_result.losses]
            train_acc.append(train_result.accuracy)
            
            #TEST
            test_result = self.test_epoch(dl_test, **kw)
            test_loss += [l.item() for l in test_result.losses]
            test_acc.append(test_result.accuracy)
            
            # EARLY STOPPING
            pass
        
        
        return FitResult(num_epochs, train_loss, train_acc, test_loss, test_acc)
    
    
    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        '''
        Train once over a training set (single epoch)
        '''
        self.model.train(True)
        return self._foreach_batch(dl_train, self.train_batch, **kw)
    
    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        '''
        Evaluate once over test / valid set (single epoch)
        '''
        self.model.train(False)
        return self._foreach_batch(dl_test, self.test_batch, **kw)
    
    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        '''
        Run a single batch forward through model - calculate loss - back prop - update weight
        '''
        raise NotImplementedError() # to be overriden
        
    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        raise NotImplementedError()
        
    @staticmethod
    def _print(msg, verbose=True):
        if verbose:
            print(msg)
            
    @staticmethod
    def _foreach_batch(
        dl: DataLoader,
        forward_fn: Callable[[Any], BatchResult],
        verbose=True,
        max_batches=None,
    ) -> EpochResult:
        '''
        Evaluates the given forward function on batches from the given DL, and prints result
        '''
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)
        
        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size
        
        if verbose:
            pbar_fn = tqdm.auto.tqdm
            pbar_file = sys.stdout
        else:
            pbar_fn = tqdm.tqdm
            pbar_file = open(os.devnull, 'w')
        
        pbar_name = forward_fn.__name__
        with pbar_fn(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)
                
                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()
                
                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct
            
            avg_loss = sum(losses) / num_batches
            accuracy = 100.0 * num_correct / num_samples
            pbar.set_description(
                f'{pbar_name} '
                f'(Avg Loss {avg_loss:.3f}, '
                f'Accuracy {accuracy:.3f})'
            )
        
        if not verbose:
            pbar_file.close()
        
        return EpochResult(losses=losses, accuracy=accuracy)

    
class ViTTrainer(Trainer):
    '''
    Trainer for the ViT Model, for multi-label classification
    '''
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        device: Optional[torch.device] = None
    ):
        super().__init__(model, device)
        self.optimizer = optimizer,
        self.loss_fn = loss_fn
        
    
    def train_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X, y = X.to(self.device), y.to(self.device)
            
        self.model: nn.Module
        batch_loss: float
        num_correct: int
        
        # setup
        self.optimizer.zero_grad()
        
        # forward
        y_scores = self.model(X)
        batch_loss = self.loss_fn(y_scores, y)
        
        # backward
        batch_loss.backward()
        self.optimizer.step()
        
        num_correct = 0 # to be modified
        
        return BatchResult(batch_loss, num_correct)
    
    def test_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X, y = X.to(self.device), y.to(self.device)
            
        self.model: nn.Module
        batch_loss: float
        accuracy: float
        
        with torch.no_grad():
            y_scores = self.model(X)
            
            # loss
            batch_loss = self.loss_fn(y_scores, y)
            
            num_correct = 0 # to be modified
        
        return BatchResult(batch_loss, accuracy)
        
        




print("TESTING")