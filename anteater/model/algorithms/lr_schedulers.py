from typing import List

from math import cos
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupLR(_LRScheduler):
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
        if iter >= warmup_iters:
            return 1.0
        if method == "constant":
            return warmup_ratio
        elif method == "linear" and warmup_iters != 0:
            alpha = iter / warmup_iters
            return warmup_ratio * (1 - alpha) + alpha
        elif method == 'exp' and warmup_iters != 0:
            return warmup_ratio ** (1 - iter / warmup_iters)
        else:
            raise ValueError("Unknown warmup method: {}".format(method))


class WarmupPolyLR(WarmupLR):
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
        if self.max_iters <= 0:
            return None
        coeff = (1.0 - iter / self.max_iters) ** self.power
        # Constant ending lr.
        if coeff < self.constant_ending:
            lr = [_lr * self.constant_ending for _lr in self.base_lrs]
        else:
            lr = [(_lr - self.min_lr) * coeff + self.min_lr for _lr in self.base_lrs]

        return lr
