import torch as th
from functools import partial

class TensorDict():

    def __init__(self,
                 shape=None,
                 seq_lengths=None,
                 columns=None,
                 index_mask=None,
                 groups=None,
                 storage=None,
                 is_cuda=False,
                 device_id=0,
                 ):

        # init from args
        self.shape = shape
        self.seq_lengths = seq_lengths # byte tensor of dimensions b*t indicating whether element in (1) or outside (2) of sequence
        self.columns = columns
        self.index_mask = index_mask
        self.groups = groups
        self.storage = storage
        self.is_cuda =is_cuda
        self.device_id = device_id
        pass

    def __getitem__(self, key):
        '''
        Indexing and Slicing returns a new TensorDict that shares storage with the original TensorDict.

        Indexing has two different modes:
            - key : list(strings) - column slice
            - key : int/slice/list(int) (,int/slice/list(int)) - slicing along batch and time dimensions

        :param key:
        :return: TensorDict with shared storage
        '''

        # detect which mode
        if not isinstance(key, list):
            key = [key] # simplifies handling
        if all([isinstance(k, str) for k in key]): # column slice

            new_groups = {} # groups accessible in new view
            new_columns = [] # columns accessible in new view

            new_column_set = set()
            for _k in key:
                if _k in self.groups:
                    new_column_set.update(self.groups[_k])
                    new_groups[_k] = self.groups[_k]
                elif _k in self.columns:
                    new_column_set.update(_k)
                else:
                    raise KeyError("Unrecognized column key: {}. Available columns: {}".format(_k, list(self.columns.keys())))

            for _k in self.columns: # enforces original column order
                if _k in new_column_set:
                    new_columns += _k

            return TensorDict(shape=self.shape,
                              seq_lengths=self.seq_lengths,
                              columns=new_columns,
                              index_mask=self.index_mask,
                              groups=new_groups,
                              storage=self.storage) # TODO: give it all the arguments

        elif all([isinstance(k, (int, slice, list)) for k in key]) and len(key) <= 2: # batch / time slice
            new_index_mask = self.index_mask
            new_seq_lengths = self.seq_lengths
            new_shape = self.shape
            for _i, _k in enumerate(key):
                if isinstance(_k, int):
                    _k = [_k] # only deal with lists
                if isinstance(_k, slice):
                    _k = list(range(_k.start, _k.stop, _k.step)) # only deal with lists
                if isinstance(_k, (list, tuple)) and len(_k) > 0:
                    if not all(_k[_j] <= _k[_j + 1] for _j in range(len(_k) - 1)): # indices have to be in ascending order, no duplicates!
                        raise KeyError("Invalid keys {} along {} dimension (not in ascending order, or with duplicates). "
                                       "Valid keys: {}".format(str(_k),
                                        {0: "b", 1: "t"}[_i],
                                        self.index_mask[_i]))
                    if not all([isinstance(__k, int) for __k in _k]):
                        raise IndexError("Unrecognized item {} in index{}.".format(_k, key))
                    invalid_idx = [_j for _j in _k if (_j not in self.index_mask[_i])] # check if all indices are in scope
                    if len(invalid_idx) > 0:
                        raise KeyError("Invalid keys {} along {} dimension. Valid keys: {}".format(str(invalid_idx),
                                                                                                   {0: "b", 1: "t"}[_i],
                                                                                                   self.index_mask[_i]))
                    new_index_mask[_i] = _k
                    new_shape[_i] = len(_k)
                else:
                    raise IndexError("Unrecognized item {} in index{}.".format(_k, key))
                if _i == 0: # batch dimension
                    new_seq_lengths = self.seq_lengths[self._long(new_index_mask[0]), :]
                elif _i == 1: # adjust sequence lengths along time dimension
                    former_idxs_at_seq_end = [ self.index_mask[_b][self.seq_lengths[_b]-1] for _b in new_index_mask[0] ]
                    new_seq_lengths[:] = [ len([_j for _j in new_index_mask[1]
                                                           if _j < former_idxs_at_seq_end[_bi]])
                                                    for _bi in range(len(new_index_mask[0]))]
                    # new_seq_lengths is now indexed by new batch views

            return TensorDict(shape=new_shape,
                              seq_lengths=new_seq_lengths,
                              columns=self.columns,
                              index_mask=new_index_mask,
                              groups=self.groups,
                              storage=self.storage)  # TODO: give it all the arguments
        else:
            raise IndexError("Unrecognized index format: {}".format(key))

        pass

    def __setitem__(self, key, value):

        pass

    def _long(self, item):
        """
        convenience function returning long tensor type corresponding to proper device and device id
        """
        return partial(th.cuda.LongTensor, device=self.device_id) if self.is_cuda else th.LongTensor

    def seq_lengths(self):
        return

    def extend(self, data_scheme=None, td=None):

        pass

    def clone(self, device=None, device_id=None):

        pass

    def dict(self, transforms, gstack=True):

        pass

    def tensor(self, transforms, gstack=True):

        pass

    def mask(self):

        pass

