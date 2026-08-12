from collections import OrderedDict
from problems.problem import Problem
from itertools import count
import torch


class TrainingPriorityQueue:
    def __init__(self, maxsize, use_current_obs, data_size=None):
        self.maxsize = maxsize
        self.data_size = data_size
        self.key_proxy = count() # proxy for dict key
        self.queue = OrderedDict()
        self.use_current_obs = use_current_obs
        if self.use_current_obs:
            self.latest_value = None

    def push(self, value, priority):
        assert isinstance(value, torch.Tensor) 
        if self.data_size is None:
            self.data_size = value.shape
        if self.use_current_obs:
            self.latest_value = value

        if priority < 0.5:
            # reject if priority below threshold
            return

        key = next(self.key_proxy)
        if len(self.queue) < self.maxsize:
            self.queue[key] = (value, priority)
        else:
            # find lowest-priority element and replace
            min_key = min(self.queue.keys(), key=lambda k: self.queue[k][1])
            if priority >= self.queue[min_key][1]:
                del self.queue[min_key]
                self.queue[key] = (value, priority)

    def get(self):
        bf = [v for v, _ in self.queue.values()]
        if len(bf) < self.maxsize:
            if len(bf) == 0:
                assert self.data_size is not None
            bf += [torch.zeros(self.data_size),] * (self.maxsize - len(bf))
        if self.use_current_obs:
            bf.append(self.latest_value if self.latest_value is not None else torch.zeros(self.data_size))
        return torch.cat(bf)
    
    def empty(self):
        return len(self.queue) == 0
    
    def reset(self):
        self.queue.clear()
        if self.use_current_obs:
            self.latest_value = None

    def __len__(self):
        return len(self.queue)


class InferencePriorityQueue:
    def __init__(self, maxsize, use_current_obs, 
                 rejection_threshold=0.5, 
                 no_repeat_threshold=0.05,
                 data_size=None):
        self.maxsize = maxsize
        self.data_size = data_size
        self.key_proxy = count() # proxy for dict key
        self.queue = OrderedDict()
        self.use_current_obs = use_current_obs
        self.rejection_threshold = rejection_threshold
        self.no_repeat_threshold = no_repeat_threshold
        if self.use_current_obs:
            self.latest_value = None
        self.last_priority = 0

    def push(self, value, priority):
        assert isinstance(value, torch.Tensor) 
        if self.data_size is None:
            self.data_size = value.shape
        if self.use_current_obs:
            self.latest_value = value

        if priority < self.rejection_threshold or abs(priority - self.last_priority) < self.no_repeat_threshold:
            # reject if priority below threshold or if priority too close to previous priority
            return

        key = next(self.key_proxy)
        if len(self.queue) < self.maxsize:
            self.queue[key] = (value, priority)
        else:
            # find lowest-priority element (largest key if tied) and replace
            min_key = min(self.queue.keys(), key=lambda k: (self.queue[k][1], -k))
            if priority >= self.queue[min_key][1]:
                del self.queue[min_key]
                self.queue[key] = (value, priority)
        self.last_priority = priority

    def get(self):
        bf = [v for v, _ in self.queue.values()]
        if len(bf) < self.maxsize:
            if len(bf) == 0:
                assert self.data_size is not None
            bf += [torch.zeros(self.data_size),] * (self.maxsize - len(bf))
        if self.use_current_obs:
            bf.append(self.latest_value if self.latest_value is not None else torch.zeros(self.data_size))
        return torch.cat(bf)
    
    def empty(self):
        return len(self.queue) == 0
    
    def reset(self):
        self.queue.clear()
        if self.use_current_obs:
            self.latest_value = None
        self.last_priority = 0

    def __len__(self):
        return len(self.queue)


queue_dict = {
    "evict_past": TrainingPriorityQueue,
    "evict_latest_norepeat": InferencePriorityQueue,
}


class Process:
    def __init__(self, problem: Problem, queue_strategy="evict_past", queue_kwargs=None):
        self.problem = problem
        buffer_class = queue_dict.get(queue_strategy)
        self.buffer = buffer_class(maxsize=problem.buffer_size, 
                                    use_current_obs=problem.use_current_obs,
                                    **(queue_kwargs or {}))

    def get_obs(self, idx):
        obs = self.problem.get_sample(idx)
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs)
        return obs, idx/self.problem.seq_len
    
    def set_action(self, sample, priority):
        self.buffer.push(sample, priority)

    def get_buffer(self):
        return self.buffer.get()
    
    def get_target(self, t):
        target = self.problem.get_target(t)
        if target is not None and not isinstance(target, torch.Tensor):
            target = torch.as_tensor(target)
        return target
    
    def reset(self, sidx=None):
        if sidx is not None:
            self.problem.reset_sidx(sidx)
        else:
            self.problem.reset()
        self.buffer.reset()
