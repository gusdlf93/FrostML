from typing import Optional
from collections import defaultdict, deque

import torch.distributed
import frostml.core.dist as dist

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


__all__ = ['EpochTracker']


class MAQue:

    def __init__(self, kernel_size: int = 20,):
        self.deque = deque(maxlen=kernel_size)
        self.total = 0.
        self.count = 0

    def update(self, value, n: int = 1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not dist.is_dist_available_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def average(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_average(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]


class EpochTracker:

    def __init__(self, delimiter: Optional[str] = None, writer: Optional[SummaryWriter] = None):
        self.trackers = defaultdict(MAQue)
        self.delimiter = delimiter if delimiter is not None else ' ' * 50
        self.writer = writer

    def __getattr__(self, item):
        if item in self.trackers:
            return self.trackers[item]
        if item in self.__dict__:
            return self.__dict__[item]
        raise AttributeError(f'`{type(self).__name__}` has no attribute `{item}`')

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, Tensor):
                v = v.item()
            self.trackers[k].update(v)

    def synchronize_between_processes(self):
        for tracker in self.trackers.values():
            tracker.synchronize_between_processes()

    def display(self, header: Optional[str] = ''):
        message_buffer = [header]
        for k, v in self.trackers.items():
            if k.startswith('acc'):
                k = f'acc@{list(k)[-1]}'
                message_buffer.append(f'{k}: {v.average:.3f}%')
            else:
                message_buffer.append(f'{k}: {v.average:.3f}')
        message = ' - '.join(message_buffer)
        print(f'\r{message}', end=self.delimiter)

    def summarize(self, header: Optional[str] = ''):
        message_buffer = [header]
        for k, v in self.trackers.items():
            if k.startswith('acc'):
                k = f'acc@{list(k)[-1]}'
                message_buffer.append(f'{k}: {v.global_average:.3f}%')
            else:
                message_buffer.append(f'{k}: {v.global_average:.3f}')
        message = ' - '.join(message_buffer)
        print(f'\r{message}', end=f'{self.delimiter}\n')

    def publish_to_tensorboard(self, prefix, postfix, global_step):
        if self.writer:
            if postfix == 'batch':
                for k, v in self.trackers.items():
                    self.writer.add_scalar(f'{prefix} / {k} ({postfix})', v.value, global_step)
            elif postfix == 'epoch':
                for k, v in self.trackers.items():
                    self.writer.add_scalar(f'{prefix} / {k} ({postfix})', v.global_average, global_step)
            else:
                raise ValueError
