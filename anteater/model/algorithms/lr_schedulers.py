from typing import List

from math import cos, pi
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupLR(_LRScheduler):
    """
    Learning rate warmup.

    warmup_method (string): Type of warmup used. It can be None(use no warmup),
        'constant', 'linear' or 'exp'
    warmup_iters (int): The number of iterations or epochs that warmup lasts
    warmup_ratio (float): LR used at the beginning of warmup equals to
        warmup_ratio * initial_lr
    """

    def __init__(
            self,
            optimizer: Optimizer,
            warmup_method: str = "linear",
            warmup_iters: int = 1000,
            warmup_ratio: float = 0.001,
            last_epoch: int = -1,
    ):
        # validate the "warmup" argument
        if warmup_method is not None:
            if warmup_method not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup_method}" is not a supported type for warming up, valid'
                    ' types are "constant", "linear" and "exp"')
        if warmup_method is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.warmup_ratio = warmup_ratio
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.by_iter = True
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:

        if self.warmup_method is None or self.last_epoch >= self.warmup_iters:
            return self.get_regular_lr(self.last_epoch)
        else:
            return self.get_warmup_lr(self.last_epoch)

    def get_warmup_lr(self, iter: int):
        warmup_ratio = self._get_warmup_ratio_at_iter(
            self.warmup_method, iter, self.warmup_iters, self.warmup_ratio
        )
        return [_lr * warmup_ratio for _lr in self.base_lrs]

    def get_regular_lr(self, iter: int):
        raise NotImplemented

    @staticmethod
    def _get_warmup_ratio_at_iter(
            method: str, iter: int, warmup_iters: int, warmup_ratio: float
    ) -> float:
        """
        Return the learning rate warmup factor at a specific iteration.
        See :paper:`ImageNet in 1h` for more details.
        Args:
            method (str): warmup method; either "constant" or "linear".
            iter (int): iteration at which to calculate the warmup factor.
            warmup_iters (int): the number of warmup iterations.
            warmup_ratio (float): the base warmup factor (the meaning changes according
                to the method used).
        Returns:
            float: the effective warmup factor at the given iteration.
        """
        if iter >= warmup_iters:
            return 1.0

        if method == "constant":
            return warmup_ratio
        elif method == "linear":
            alpha = iter / warmup_iters
            return warmup_ratio * (1 - alpha) + alpha
        elif method == 'exp':
            return warmup_ratio ** (1 - iter / warmup_iters)
        else:
            raise ValueError("Unknown warmup method: {}".format(method))


class WarmupPolyLR(WarmupLR):
    """
    Poly learning rate schedule with warmup.
    Paper: DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
        Atrous Convolution, and Fully Connected CRFs.
    Reference: https://github.com/tensorflow/models/blob/21b73d22f3ed05b650e85ac50849408dd36de32e/research/deeplab/utils/train_utils.py#L337  # noqa

    power (float): the power of poly lr.
    max_iters (int): when run by epoch, max_iters means number of epochs.
    warmup_method (string): Type of warmup used. It can be None(use no warmup),
        'constant', 'linear' or 'exp'
    warmup_iters (int): The number of iterations or epochs that warmup lasts
    warmup_ratio (float): LR used at the beginning of warmup equals to
        warmup_ratio * initial_lr
    """

    def __init__(
            self,
            optimizer: Optimizer,
            max_iters: int,
            power: float = 0.9,
            warmup_method: str = "linear",
            warmup_iters: int = 1000,
            warmup_ratio: float = 0.001,
            constant_ending: float = 0.0,
            min_lr: float = 0.,
            last_epoch: int = -1,
    ):
        super().__init__(optimizer, warmup_method, warmup_iters, warmup_ratio, last_epoch)
        self.max_iters = max_iters
        self.min_lr = min_lr
        self.power = power
        self.constant_ending = constant_ending

    def get_regular_lr(self, iter: int):
        coeff = (1.0 - iter / self.max_iters) ** self.power
        # Constant ending lr.
        if coeff < self.constant_ending:
            lr = [_lr * self.constant_ending for _lr in self.base_lrs]
        else:
            lr = [(_lr - self.min_lr) * coeff + self.min_lr for _lr in self.base_lrs]

        return lr


class WarmupCosineAnnealingLR(WarmupLR):
    """
    Cosine annealing learning rate schedule with warmup.

    max_iters (int): when run by epoch, max_iters means number of epochs.
    warmup_method (string): Type of warmup used. It can be None(use no warmup),
        'constant', 'linear' or 'exp'
    warmup_iters (int): The number of iterations or epochs that warmup lasts
    warmup_ratio (float): LR used at the beginning of warmup equals to
        warmup_ratio * initial_lr
    """

    def __init__(
            self,
            optimizer: Optimizer,
            max_iters: int,
            warmup_method: str = "linear",
            warmup_iters: int = 1000,
            warmup_ratio: float = 0.001,
            min_lr: float = 0.,
            last_epoch: int = -1,
    ):
        super().__init__(optimizer, warmup_method, warmup_iters, warmup_ratio, last_epoch)
        self.max_iters = max_iters
        self.min_lr = min_lr

    def get_regular_lr(self, iter: int):
        factor = iter / self.max_iters
        cos_out = cos(pi * factor) + 1
        lr = [self.min_lr + 0.5 * (_lr - self.min_lr) * cos_out for _lr in self.base_lrs]
        return lr
