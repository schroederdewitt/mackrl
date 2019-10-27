import numpy as np
from collections import Counter
from .episode_buffer import BatchEpisodeBuffer
from torch import multiprocessing as mp

class ContiguousReplayBuffer(object):
    def __init__(self, data_scheme, n_bs, n_t, n_agents, batch_size, is_cuda=True, is_shared_mem=True, logging_struct=None):
        self.buffer = BatchEpisodeBuffer(data_scheme=data_scheme,
                                             n_bs=n_bs,
                                             n_t=n_t,
                                             n_agents=n_agents,
                                             is_cuda=is_cuda,
                                             is_shared_mem=True)


        if is_cuda:
            self._to_cuda()
        if is_shared_mem:
            self._to_shared_mem()
        self.queue_head_pos = 0
        self.lock = mp.Lock() # TODO: could make locks more granular!
        self.len = 0
        pass

    def _to_cuda(self):
        self.buffer._to_cuda()
        pass

    def _to_shared_mem(self):
        self.buffer._share_mem()
        pass

    def put(self, obj):
        """
        Implements deque-like insertion, while being thread and process-safe.
        TODO: propagate unstructured data
        """
        self.lock.acquire()
        assert isinstance(obj, BatchEpisodeBuffer), "obj needs to be of type BatchEpisodeBuffer"
        overflow_pad = len(obj) // len(self.buffer)
        overflow = (self.queue_head_pos + len(obj) % len(self.buffer)) - len(self.buffer)

        if overflow > 0:
            self.buffer.data._transition[self.queue_head_pos:] = obj.data._transition[(overflow_pad-1)*len(self.buffer):-overflow]
            self.buffer.data._episode[self.queue_head_pos:] = obj.data._episode[(overflow_pad - 1) * len(self.buffer):-overflow]
            self.buffer.seq_lens[self.queue_head_pos:] = obj.seq_lens[(overflow_pad-1)*len(self.buffer):-overflow%len(self.buffer)]

            self.buffer.data._transition[:overflow] = obj.data._transition[-overflow:]
            self.buffer.data._episode[:overflow] = obj.data._episode[-overflow:]
            self.buffer.seq_lens[:overflow] = obj.seq_lens[-overflow:]
        else:
            self.buffer.data._transition[self.queue_head_pos:self.queue_head_pos + len(obj)] = obj.data._transition[(overflow_pad-1)*len(self.buffer):]
            self.buffer.data._episode[self.queue_head_pos:self.queue_head_pos + len(obj)] = obj.data._episode[(overflow_pad - 1) * len(self.buffer):]
            self.buffer.seq_lens[self.queue_head_pos:self.queue_head_pos + len(obj)] = obj.seq_lens[(overflow_pad-1)*len(self.buffer):]

        self.queue_head_pos = (self.queue_head_pos + len(obj)) % len(self.buffer)
        self.len = min(len(self.buffer), self.len + len(obj))
        self.lock.release()
        pass

    def __len__(self):
        return self.len

    def sample(self, batch_size, seq_len=0):
        """
        Returns a BatchHistory object of size batch_size that has been sampled using priority_alpha
        """
        self.lock.acquire()

        self.sample_mem = BatchEpisodeBuffer(data_scheme=self.buffer._data_scheme,
                                             n_bs=batch_size,
                                             n_t=seq_len if seq_len !=0 else self.buffer._n_t,
                                             n_agents=self.buffer.n_agents,
                                             is_cuda=self.buffer.is_cuda,
                                             is_shared_mem=False)

        assert self.can_sample(batch_size), "Too few elements in buffer to sample from at given batch size!"

        weights = np.array([self.buffer.seq_lens[_b]+1 for _b in range(len(self))])
        history_ids = np.random.choice(np.arange(0, len(self)),
                                                size=batch_size,
                                                replace=True,
                                                p=weights/np.sum(weights))
        history_id_histogram = Counter(history_ids)

        i = 0
        for _id in set(history_ids.tolist()):

            # replace should almost always be false in practice as batch_size << len(self.buffer)
            if seq_len == 0:
                seqs_per_hist = 1
            else:
                seqs_per_hist = len(self.buffer[_id]) - seq_len + 1
            replace = False if seqs_per_hist >= history_id_histogram[_id] else True
            choices = np.random.choice(np.arange(0, seqs_per_hist),
                                                 size=history_id_histogram[_id],
                                                 replace=replace,
                                                 )
            if seq_len == 0:
                for _i, choice in enumerate(choices):
                    self.sample_mem.data._transition[i+_i] = self.buffer.data._transition[_id]
                    self.sample_mem.data._episode[i+_i] = self.buffer.data._episode[_id]
                    self.sample_mem.seq_lens[i+_i] = self.buffer.seq_lens[_id]
            else:
                for _i, choice in enumerate(choices):
                    assert False, "this bit of code has not been tested yet!"
                    self.sample_mem.data._transition[i+_i] = self.buffer.data[_id, choice:choice + seq_len]
                    self.sample_mem.data._episode[i+_i] = self.buffer.data[_id]
                    self.sample_mem.seq_lens[i+_i] = self.buffer.seq_lens[_id]
            i += len(choices)

        ret = self.sample_mem # important to clone here, else buffer corruption could occur!
        self.lock.release()
        return ret

    def can_sample(self, batch_size):
        return len(self) >= batch_size

    def __del__(self):
        pass

ReplayBuffer = ContiguousReplayBuffer
