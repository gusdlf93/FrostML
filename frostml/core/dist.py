import os

import torch.cuda
import torch.distributed


def setup_for_distributed(is_main: bool) -> None:
    import builtins as __builtins__
    builtin_print_fn = __builtins__.print

    def dist_print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print_fn(*args, **kwargs)

    __builtins__.print = dist_print


def is_dist_available_and_initialized() -> bool:
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size() -> int:
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank() -> int:
    if not is_dist_available_and_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def save_on_main_process(*args, **kwargs) -> None:
    if is_main_process():
        torch.save(*args, **kwargs)


def initialize_distributed_mode(args) -> None:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('distributed mode | disabled')
        args.distributed = False
        return

    args.distributed = True
    args.dist_backend = 'nccl'
    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    host = os.environ['MASTER_ADDR']
    print(f'distributed mode | initialized (rank {args.rank}) :: {host}', flush=True)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
