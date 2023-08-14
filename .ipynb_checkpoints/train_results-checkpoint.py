from typing import List, NamedTuple

class BatchResult(NamedTuple):
    '''
    Represents the result of training for a single batch:
        - Loss
        - Number of correct classifications
    '''
    loss: float
    num_correct: int
    

class EpochResult(NamedTuple):
    '''
    Represents the result of training for a single epoch:
        - loss per batch
        - accuracy on dataset (train / test)
    '''
    losses: List[float]
    accuracy: float
    

class FitResult(NamedTuple):
    '''
    Represents the result of fitting a model for multiple epochs 
        given a training and valid (test) set
    Losses are for each batch and accuracies are per epoch
    '''
    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]



