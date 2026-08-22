from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np

from vnpy.alpha.dataset import AlphaDataset, Segment

from .reweighter import Reweighter
from .serializable import Serializable


class BaseModel(Serializable, metaclass=ABCMeta):
    """Base interface shared by all predictive models."""

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> object:
        """Make predictions with a fitted model."""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> object:
        return self.predict(*args, **kwargs)



class AlphaModel(BaseModel):
    """Template class for machine learning algorithms"""

    @abstractmethod
    def fit(self, dataset: AlphaDataset, reweighter: Reweighter | None = None,) -> None:
        """
        Fit the model with dataset
        """
        pass

    @abstractmethod
    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        """
        Make predictions using the model
        """
        pass

    def detail(self) -> Any:
        """
        Output detailed information about the model
        """
        return


class FineTunableAlphaModel(AlphaModel):
    """Alpha model that can continue learning from a new dataset."""

    @abstractmethod
    def finetune(self, dataset: AlphaDataset, reweighter: Reweighter | None = None,) -> None:
        raise NotImplementedError
