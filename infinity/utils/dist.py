import datetime
import functools
import os
import sys
from typing import List
from typing import Union

import pytz
import torch
import torch.distributed as tdist
import torch.multiprocessing as mp


__rank, __local_rank, __world_size, __device = 0, 0, 1, 'cpu'
__rank_str_zfill = '0'
__initialized = False


def initialized():
    return __initialized


def __initialize(fork=False, backend='nccl', gpu_id_if_not_distibuted=0, timeout_minutes=30):
    global __device
    if not torch.cuda.is_available():
        print(f'[dist initialize] cuda is not available, use cpu instead', file=sys.stderr)
        return
    elif 'RANK' not in os.environ:
        torch.cuda.set_device(gpu_id_if_not_distibuted)
        __device = torch.empty(1).cuda().device
        print(f'[dist initialize] env variable "RANK" is not set, use {__device} as the device', file=sys.stderr)
        return
    # then 'RANK' must exist
    global_rank, num_gpus = int(os.environ['RANK']), torch.cuda.device_count()
    local_rank = global_rank % num_gpus
    torch.cuda.set_device(local_rank)
    print(f"global_rank:{global_rank} local_rank:{local_rank} num_gpus:{num_gpus}")
    
    # ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py#L29
    """
    if mp.get_start_method(allow_none=True) is None:
        method = 'fork' if fork else 'spawn'
        print(f'[dist initialize] mp method={method}')
        mp.set_start_method(method)
    """
    tdist.init_process_group(backend=backend,
                            device_id=torch.device(f'cuda:{local_rank}'), 
                            timeout=datetime.timedelta(seconds=timeout_minutes * 60))
    
    global __rank, __local_rank, __world_size, __initialized, __rank_str_zfill
    __local_rank = local_rank
    __rank, __world_size = tdist.get_rank(), tdist.get_world_size()
    __rank_str_zfill = str(__rank).zfill(len(str(__world_size)))
    __device = torch.device(local_rank)
    __initialized = True
    
    assert tdist.is_initialized(), 'torch.distributed is not initialized!'
    print(f'[lrk={get_local_rank()}, rk={get_rank()}]')


def get_rank():
    return __rank


def get_rank_given_group(group: tdist.ProcessGroup):
    return tdist.get_rank(group=group)


def get_rank_str_zfill():
    return __rank_str_zfill


def get_local_rank():
    return __local_rank


def get_world_size():
    return __world_size


def get_device():
    return __device


def set_gpu_id(gpu_id: int):
    if gpu_id is None: return
    global __device
    if isinstance(gpu_id, (str, int)):
        torch.cuda.set_device(int(gpu_id))
        __device = torch.empty(1).cuda().device
    else:
        raise NotImplementedError


def is_master():
    return __rank == 0


def is_local_master():
    return __local_rank == 0


def is_visualizer():
    return __rank == 0
    # return __rank == max(__world_size - 8, 0)


def parallelize(net, syncbn=False):
    if syncbn:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    return net


def new_group(ranks: List[int]):
    if __initialized:
        return tdist.new_group(ranks=ranks)
    return None


def new_local_machine_group():
    if __initialized:
        cur_subgroup, subgroups = tdist.new_subgroups()
        return cur_subgroup
    return None


def barrier():
    if __initialized:
        tdist.barrier()


def allreduce(t: torch.Tensor, async_op=False):
    if __initialized:
        if not t.is_cuda:
            cu = t.detach().cuda()
            ret = tdist.all_reduce(cu, async_op=async_op)
            t.copy_(cu.cpu())
        else:
            ret = tdist.all_reduce(t, async_op=async_op)
        return ret
    return None


def allgather(t: torch.Tensor, cat=True) -> Union[List[torch.Tensor], torch.Tensor]:
    if __initialized:
        if not t.is_cuda:
            t = t.cuda()
        ls = [torch.empty_like(t) for _ in range(__world_size)]
        tdist.all_gather(ls, t)
    else:
        ls = [t]
    if cat:
        ls = torch.cat(ls, dim=0)
    return ls


def allgather_diff_shape(t: torch.Tensor, cat=True) -> Union[List[torch.Tensor], torch.Tensor]:
    if __initialized:
        if not t.is_cuda:
            t = t.cuda()
        
        t_size = torch.tensor(t.size(), device=t.device)
        ls_size = [torch.empty_like(t_size) for _ in range(__world_size)]
        tdist.all_gather(ls_size, t_size)
        
        max_B = max(size[0].item() for size in ls_size)
        pad = max_B - t_size[0].item()
        if pad:
            pad_size = (pad, *t.size()[1:])
            t = torch.cat((t, t.new_empty(pad_size)), dim=0)
        
        ls_padded = [torch.empty_like(t) for _ in range(__world_size)]
        tdist.all_gather(ls_padded, t)
        ls = []
        for t, size in zip(ls_padded, ls_size):
            ls.append(t[:size[0].item()])
    else:
        ls = [t]
    if cat:
        ls = torch.cat(ls, dim=0)
    return ls


def broadcast(t: torch.Tensor, src_rank) -> None:
    if __initialized:
        if not t.is_cuda:
            cu = t.detach().cuda()
            tdist.broadcast(cu, src=src_rank)
            t.copy_(cu.cpu())
        else:
            tdist.broadcast(t, src=src_rank)


def dist_fmt_vals(val: float, fmt: Union[str, None] = '%.2f') -> Union[torch.Tensor, List]:
    if not initialized():
        return torch.tensor([val]) if fmt is None else [fmt % val]
    
    ts = torch.zeros(__world_size)
    ts[__rank] = val
    allreduce(ts)
    if fmt is None:
        return ts
    return [fmt % v for v in ts.cpu().numpy().tolist()]


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force or is_master():
            ret = func(*args, **kwargs)
        else:
            ret = None
        barrier()
        return ret
    return wrapper


def local_master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force or is_local_master():
            ret = func(*args, **kwargs)
        else:
            ret = None
        barrier()
        return ret
    return wrapper


def for_visualize(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_visualizer():
            # with torch.no_grad():
            ret = func(*args, **kwargs)
        else:
            ret = None
        return ret
    return wrapper


def finalize():
    if __initialized:
        tdist.destroy_process_group()


def init_distributed_mode(local_out_path, fork=False, only_sync_master=False, timeout_minutes=30):
    try:
        __initialize(fork=fork, timeout_minutes=timeout_minutes)
        barrier()
    except RuntimeError as e:
        print(f'{"!"*80}   dist init error (NCCL Error?), stopping training!   {"!"*80}', flush=True)
        raise e
    
    if local_out_path is not None: os.makedirs(local_out_path, exist_ok=True)
    _change_builtin_print(is_local_master())
    if (is_master() if only_sync_master else is_local_master()) and local_out_path is not None and len(local_out_path):
        sys.stdout, sys.stderr = BackupStreamToFile(local_out_path, for_stdout=True), BackupStreamToFile(local_out_path, for_stdout=False)


def _change_builtin_print(is_master):
    import builtins as __builtin__
    
    builtin_print = __builtin__.print
    if type(builtin_print) != type(open):
        return
    
    def prt(*args, **kwargs):
        force = kwargs.pop('force', False)
        clean = kwargs.pop('clean', False)
        deeper = kwargs.pop('deeper', False)
        if is_master or force:
            if not clean:
                f_back = sys._getframe().f_back
                if deeper and f_back.f_back is not None:
                    f_back = f_back.f_back
                file_desc = f'{f_back.f_code.co_filename:24s}'[-24:]
                time_str = datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime('[%m-%d %H:%M:%S]')
                builtin_print(f'{time_str} ({file_desc}, line{f_back.f_lineno:-4d})=>', *args, **kwargs)
            else:
                builtin_print(*args, **kwargs)
    
    __builtin__.print = prt


class BackupStreamToFile(object):
    def __init__(self, local_output_dir, for_stdout=True):
        self.for_stdout = for_stdout
        self.terminal_stream = sys.stdout if for_stdout else sys.stderr
        fname = os.path.join(local_output_dir, 'b1_stdout.txt' if for_stdout else 'b2_stderr.txt')
        existing = os.path.exists(fname)
        self.file_stream = open(fname, 'a')
        if existing:
            time_str = datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime('[%m-%d %H:%M:%S]')
            self.file_stream.write('\n'*7 + '='*55 + f'   RESTART {time_str}   ' + '='*55 + '\n')
        self.file_stream.flush()
        os.system(f'ln -s {fname} /opt/tiger/run_trial/ >/dev/null 2>&1')
        self.enabled = True
    
    def write(self, message):
        self.terminal_stream.write(message)
        self.file_stream.write(message)
    
    def flush(self):
        self.terminal_stream.flush()
        self.file_stream.flush()
    
    def isatty(self):
        return True
    
    def close(self):
        if not self.enabled:
            return
        self.enabled = False
        self.file_stream.flush()
        self.file_stream.close()
        if self.for_stdout:
            sys.stdout = self.terminal_stream
            sys.stdout.flush()
        else:
            sys.stderr = self.terminal_stream
            sys.stderr.flush()
    
    def __del__(self):
        self.close()
