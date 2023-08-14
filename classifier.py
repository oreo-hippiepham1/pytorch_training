import torch
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from torch import Tensor, nn
from typing import Optional


class Classifier(nn.Module, ABC):
    '''
    Wraps a model which produces raw class scores, and provides methods to compute
        class label and probabilities
    '''
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        
        
    def forward(self, X: Tensor) -> Tensor:
        z: Tensor = None
        z = self.model(X)     
        assert z.shape[0] == X.shape[0]
        return z
      
    def predict_proba(self, X: Tensor) -> Tensor:
        z = self.model(X)
        return self.predict_proba_scores(z)
    
    def predict_proba_scores(self, z: Tensor) -> Tensor:
        return nn.functional.sigmoid(z)
    
    def classify(self, X: Tensor) -> Tensor:
        y_proba = self.predict_proba(X)
        
        return self._classify(y_proba)
    
    def classify_scores(self, z: Tensor) -> Tensor:
        y_proba = self.predict_proba_scores(z)
        
        return self._classify(y_proba)
    
    @abstractmethod
    def _classify(self, y_proba: Tensor) -> Tensor:
        pass
    

class MultiLabelClassifier(Classifier):
    def __init__(self, model, threshold: float = 0.5):
        super().__init__(model)
        self.threshold = threshold
        
    def _classify(self, y_proba: Tensor):
        return (y_proba > self.threshold).type(torch.float32)
    
    
    









