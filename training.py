import os
import abc
import sys
import torch
import torch.nn as nn
import tqdm.auto
from torch import Tensor
import numpy as np
from typing import Any, Tuple, Callable, Optional, cast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from train_results import FitResult, BatchResult, EpochResult
from classifier import Classifier

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
        self.writer = SummaryWriter()
        
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
        
        actual_num_epochs = 0
        epochs_without_improvement = 0
        
        train_loss, train_acc, test_loss, test_acc = [], [], [], [] # initialize metrics
        best_acc = None
        
        for epoch in range(num_epochs):
            verbose = False
            if print_every > 0 and (
                epoch % print_every == 0 or epoch == num_epochs - 1
            ):
                verbose = True
            self._print(f'--- EPOCH {epoch+1}/{num_epochs} ---', verbose)
            
            # TRAIN
            train_result = self.train_epoch(dl_train, **kw) # EpochResult obj: losses: List[float], accuracy: float        

            train_loss += [l.item() for l in train_result.losses]
            train_acc.append(train_result.accuracy)
            
            # adds Tensorboard logging
            self.writer.add_scalar('Train Loss', np.mean(train_result.losses), epoch) 
            self.writer.add_scalar('Train Accuracy', train_result.accuracy, epoch)
            
            #TEST
            test_result = self.test_epoch(dl_test, **kw)
            test_loss += [l.item() for l in test_result.losses]
            test_acc.append(test_result.accuracy)
            
            # adds Ten
            self.writer.add_scalar('Test Loss', np.mean(test_result.losses), epoch)
            self.writer.add_scalar('Test Accuracy', test_result.accuracy, epoch)
            
            # Closing tensorboard writer
            self.writer.close()
            
            actual_num_epochs += 1
            
            # EARLY STOPPING
            if best_acc is None or test_result.accuracy > best_acc:
                best_acc = test_result.accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if early_stopping is not None and epochs_without_improvement >= early_stopping:
                    print(f"---- Stopping training after {epochs_without_improvement=} ----")
                    break
                    
            
            # SAVING CHECKPOINTS
            if checkpoints is not None and best_acc is not None and test_result.accuracy > best_acc:
                self.save_checkpoint(checkpoints)
            

        return FitResult(num_epochs, train_loss, train_acc, test_loss, test_acc)
    
    
    def save_checkpoint(self, cp_filename):
        torch.save(self.model, cp_filename)
        print(f"\n*** Saved checkpoint {cp_filename}")
    
    
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
                
                pbar.set_description(f'{pbar_name} ({batch_res.loss:.7f})')
                pbar.update()
                
                losses.append(batch_res.loss)
                #num_correct += batch_res.num_correct
                num_correct += np.sum(batch_res.num_correct, axis=0)
            
            avg_loss = sum(losses) / num_batches
            accuracy = 100.0 * np.sum(num_correct) / (num_samples * 20) # 20: n_classes
            pbar.set_description(
                f'{pbar_name} '
                f'(Avg Loss {avg_loss:.7f}, '
                f'Accuracy {accuracy:.7f})'
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
        model: Classifier,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        device: Optional[torch.device] = None
    ):
        super().__init__(model, device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
    
    def train_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X, y = X.to(self.device), y.to(self.device)
            
        self.model: Classifier
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
        
        # Calculate subset accuracy
        y_pred = self.model.classify_scores(y_scores)
        
        y_pred = y_pred.cpu().numpy()
        y = y.cpu().numpy()

        #num_samples = y.shape[0]

        # Compare all elements of y_pred and y
        # all_match = (y_pred == y).all(axis=1)
        # num_correct = all_match.sum()
        
        # Calculate accuracy for each class
        class_accuracy = np.mean(y_pred == y, axis=0)

        # Calculate overall accuracy
        overall_accuracy = np.mean(class_accuracy)
        num_correct = np.sum(y_pred == y)
        
        return BatchResult(batch_loss, num_correct)
    
    def test_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X, y = X.to(self.device), y.to(self.device)
            
        self.model: Classifier
        batch_loss: float
        num_correct: float
        
        with torch.no_grad():
            y_scores = self.model(X)
            
            # loss
            batch_loss = self.loss_fn(y_scores, y)
                        
            y_pred = self.model.classify_scores(y_scores)
            y_pred = y_pred.cpu().numpy()
            y = y.cpu().numpy()

            #num_samples = y.shape[0]

            # Compare all elements of y_pred and y
#             all_match = (y_pred == y).all(axis=1)

#             num_correct = all_match.sum()

            # Calculate accuracy for each class
            class_accuracy = np.mean(y_pred == y, axis=0)

            # Calculate overall accuracy
            overall_accuracy = np.mean(class_accuracy)
            num_correct = np.sum(y_pred == y)
            # print(f'{y_pred=} --- {y=}')
            # print(f'{num_correct=}')
        
        return BatchResult(batch_loss, num_correct)
        
        