from copy import deepcopy
import math
import numpy as np
import pandas as pd
from torch.autograd import Variable
import torch as th
from types import SimpleNamespace as SN

from .transforms import TRANSFORMS, _has_gradient, _onehot
from .scheme import Scheme


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except:
        return False

def is_nan(obj):
    try:
        ret = np.isnan(obj)
    except:
        ret = False
    return ret

class BatchEpisodeBuffer():
    """
    Contains a batch of histories and takes care that sequences are padded as appropriate
    New: supports parallelization using subprocess module
    """

    def __init__(self,
                 data_scheme,
                 n_bs,
                 n_t,
                 n_agents,
                 is_cuda=False,
                 is_shared_mem=False,
                 sizes=None):

        self.n_agents = n_agents
        self._data_scheme = data_scheme
        self._n_bs = n_bs
        self._n_t = n_t
        self.is_cuda = is_cuda
        self.is_shared_mem = is_shared_mem
        self.data = SN()
        self.columns = SN()
        self._setup_data(self._data_scheme,
                         sizes,
                         self._n_bs,
                         self._n_t)
        self.seq_lens = [0 for _ in range(len(self))]
        self.format = "bs" # this is ALWAYS the order of contiguous batch histories
        pass

    def _setup_data(self, data_scheme, sizes, n_bs, n_t):
        if sizes is None:
            self.sizes = {_df["name"]:_df["shape"] for _df in data_scheme.scheme_list if _df.get("switch", True)}
        else:
            self.sizes = sizes

        n_transition_cols = sum([np.asscalar(np.prod(self.sizes[_df["name"]]))
                                 for _df in data_scheme.scheme_list
                                 if _df.get("switch", True) and _df.get("scope", "transition")=="transition"])

        n_episode_cols = sum([np.asscalar(np.prod(self.sizes[_df["name"]]))
                                 for _df in data_scheme.scheme_list
                                 if _df.get("switch", True) and _df.get("scope", "transition")=="episode"])

        if not self.is_cuda:
            if n_transition_cols > 0:
                self.data._transition = th.FloatTensor(n_bs, n_t, n_transition_cols)* float("nan")
            if n_episode_cols > 0:
                self.data._episode = th.FloatTensor(n_bs, n_episode_cols) * float("nan")
        else:
            if n_transition_cols > 0:
                self.data._transition = th.cuda.FloatTensor(n_bs, n_t, n_transition_cols) * float("nan")
            if n_episode_cols > 0:
                self.data._episode = th.cuda.FloatTensor(n_bs, n_episode_cols) * float("nan")

        #create a dictionary with column end and start positions - transition data
        self.columns._transition = {}
        col_counter = 0
        for _i, _df in enumerate([_df for _df in data_scheme.scheme_list
                                if _df.get("switch", True) and _df.get("scope", "transition")=="transition"]):
            self.columns._transition[_df["name"]] = (col_counter, col_counter + np.asscalar(np.prod(self.sizes[_df["name"]])))
            col_counter += np.asscalar(np.prod(self.sizes[_df["name"]]))

        #create a dictionary with column end and start positions - episode data
        self.columns._episode = {}
        col_counter = 0
        for _i, _df in enumerate([_df for _df in data_scheme.scheme_list
                                  if _df.get("switch", True) and _df.get("scope", "transition")=="episode"]):
            self.columns._episode[_df["name"]] = (col_counter, col_counter + np.asscalar(np.prod(self.sizes[_df["name"]])))
            col_counter += np.asscalar(np.prod(self.sizes[_df["name"]]))

        #fill in special elements, such as agent_ids
        for _col, _vals in self.columns._transition.items():
            if _col[:15] == "agent_id_onehot":
                self.data._transition[:,:,_vals[0]:_vals[1]] = _onehot(float(_col.split("__agent")[1]),
                                                                       rng=(0, _vals[1]-_vals[0]),
                                                                       is_cuda=self.data._transition.is_cuda)
            elif _col[:8] == "agent_id":
               self.data._transition[:,:,_vals[0]:_vals[1]] = float(_col.split("__agent")[1])
        # set up container for unstructured data that is not being handled any further by episode_buffer
        self.data.unstructured = {}

        if self.is_shared_mem:
            self._share_mem()

        pass

    def get_col(self, col, scope="transition", agent_ids=None, t=None, stack=True, bs=None):
        """
        retrieve a single column from buffer (or a list of columns, labelled by agent_ids)

        indexing efficiently over two dimensions is *** in pytorch
        see https://discuss.pytorch.org/t/how-to-select-index-over-two-dimension/10098/2
        [4x faster to use cat() than to slice twice]

        t: can be slice or None
        bs: can be list of indices or slice or None
        """
        if agent_ids is not None:
            agent_ids = [agent_ids] if not isinstance(agent_ids, (list, tuple)) else agent_ids

        def _get_col(_col, _scope, _t, _bs):
            if scope in ["transition"]:
                bs_slice = self._parse_id(_bs, dim_len=self.data._transition.shape[0], name="bs", allow_lists=True)
                t_slice = self._parse_id(_t, dim_len=self.data._transition.shape[1], name="t", allow_lists=False)
                col_slice = slice(self.columns._transition[_col][0], self.columns._transition[_col][1])
                if isinstance(bs, (tuple, list)):
                    return th.cat([self.data._transition[slice(_bs,_bs+1), t_slice, col_slice] for _bs in bs])
                else:
                    return self.data._transition[bs_slice, t_slice, col_slice]
            elif scope in ["episode"]:
                # we just ignore the t-slice
                # TODO: should we rather throw an assertion if t!=None ?
                bs_slice = self._parse_id(_bs, dim_len=self.data._episode.shape[0], name="bs", allow_lists=True)
                col_slice = slice(self.columns._episode[_col][0], self.columns._episode[_col][1])
                if isinstance(bs, (tuple, list)):
                    return th.cat([self.data._episode[slice(_bs,_bs+1), col_slice] for _bs in bs])
                else:
                    return self.data._episode[bs_slice, col_slice]
            else:
                assert False, "unknown scope ({}) at get_col".format(_scope)

        if agent_ids is not None:
            ret = [ _get_col(_col="{}__agent{}".format(col, agent_id), _scope=scope, _t=t, _bs=bs) for agent_id in agent_ids]
            if scope in ["transition"]:
                tformat = "bs*t*v"
            elif scope in ["episode"]:
                tformat = "bs*v"
            else:
                assert False, "unknown scope!"
            if stack:
                ret = th.stack(ret)
                tformat  = "a*" + tformat
            else:
                tformat = "[a]*" + tformat
        else:
            ret = _get_col(_col=col, _t=t, _bs=bs, _scope=scope)
            if scope in ["transition"]:
                tformat = "bs*t*v"
            elif scope in ["episode"]:
                tformat = "bs*v"
            else:
                assert False, "unknown scope!"

        return ret, tformat

    @staticmethod
    def _parse_id(id, dim_len, name,  allow_lists=False):
            if not allow_lists:
                assert not isinstance(id, (
                tuple, list)), "id {} has to be a slice, integer id or None - lists or tuples of ids are not supported!".format(name)
            if id is None:
                return slice(0, dim_len)
            if isinstance(id, slice):
                return id
            elif isinstance(id, (tuple, list)):
                return id
            else:
                return slice(id, id + 1)

    def set_col(self, col, data, scope="transition", agent_ids=None, t=None, bs=None):
        """
        set a single column from buffer (or a list of columns, labelled by agent_ids)
        """

        assert not _has_gradient(data), "data cannot have variables attached!"

        if agent_ids is not None:
            agent_ids = [agent_ids] if not isinstance(agent_ids, (list, tuple)) else agent_ids

        data = data.cuda() if self.is_cuda else data.cpu()

        def _set_col(_col, _data, _scope, _t, _bs):
            if _scope in ["transition"]:
                bs_slice = self._parse_id(_bs, dim_len=self.data._transition.shape[0], name="bs", allow_lists=True)
                t_slice = self._parse_id(_t, dim_len=self.data._transition.shape[1], name="t", allow_lists=False)
                col_slice = slice(self.columns._transition[_col][0], self.columns._transition[_col][1])
                if not isinstance(bs, (tuple, list)):
                    assert bs_slice.stop <= self.data._transition.shape[0] and t_slice.stop <= self.data._transition.shape[1], "indices out of range!"
                    try:
                        self.data._transition[bs_slice, t_slice, col_slice] = _data
                    except Exception as e:
                        pass

                    # modify sequence lengths
                    for _bs in range(bs_slice.start, bs_slice.stop):
                        self.seq_lens[_bs] = max(self.seq_lens[_bs], t_slice.stop)
                else:
                    assert t_slice.stop <= self.data._transition.shape[1], "indices out of range!"
                    assert len(bs) <= _data.shape[0], "too many batch indices supplied!"

                    for _i, _bs in enumerate(bs): # TODO: This should work with scatter, but will have to see how exactly!
                       self.data._transition[slice(_bs, _bs+1), t_slice, col_slice] = _data[slice(_i,_i+1), :, :]
                       # adapt sequence lengths:
                       self.seq_lens[_bs] = max(self.seq_lens[_bs], t_slice.stop)
            elif _scope in ["episode"]:
                # we just ignore the t-slice
                # TODO: should we rather throw an assertion if t!=None ?
                bs_slice = self._parse_id(_bs, dim_len=self.data._episode.shape[0], name="bs", allow_lists=True)
                col_slice = slice(self.columns._episode[_col][0], self.columns._episode[_col][1])
                if not isinstance(bs, (tuple, list)):
                    assert bs_slice.stop <= self.data._episode.shape[0], "indices out of range!"
                    if self.data._episode.dim() == 2:
                        try:
                            self.data._episode[bs_slice, col_slice] = _data
                        except Exception as e:
                            pass
                    elif self.data._episode.dim() == 3:
                        self.data._episode[bs_slice, :, col_slice] = _data
                    else:
                        assert False, "unknown episode data dim!"
                else:
                    assert len(bs) <= _data.shape[0], "too many batch indices supplied!"
                    for _i, _bs in enumerate(bs): # TODO: This should work with scatter, but will have to see how exactly!
                       if self.data._episode.dim() == 2:
                           self.data._episode[slice(_bs, _bs+1), col_slice] = _data[slice(_i,_i+1), :]
                       elif self.data._episode.dim() == 3:
                           self.data._episode[slice(_bs, _bs + 1), :, col_slice] = _data[slice(_i, _i + 1),:, :]
                       else:
                           assert False, "unknown episode data dim!"
            else:
                assert False, "unknown scope ({}) in set_col".format(_scope)

        if agent_ids is not None:
            #TODO: SPEED THIS UP SOMEHOW!!
            #Note: If we are sure that columns of same type (but with different agentid) are stored next to each other in the right order
            # of agent_ids, then we could probably use that to set the space in batch somehow
            # we can check here whether this is the case by looking at self.columns._transition
            [ _set_col(_col="{}__agent{}".format(col, agent_id), _scope=scope, _t=t, _bs=bs, _data=data[i,:,:,:]) for i, agent_id in enumerate(agent_ids)]
        else:
            _set_col(_col=col, _t=t, _bs=bs, _data=data, _scope=scope)

        return self

    def __getitem__(self, items):
        """
        shortcut to select columns from data._transition
        """
        if isinstance(items, str):
            return self.get_col(col=items, scope="transition")
        elif isinstance(items, int):
            return self.data._transition[items, :, :]
        elif isinstance(items, slice):
            return self.data._transition[items,:,:]

    def __setitem__(self, items, val):
        if isinstance(items, str):
            assert isinstance(val, tuple), "val needs to be tuple (value, time slice)"
            return self.set_col(col=items, scope="transition", data=val[0], t=val[1])

    def get_cols(self, cols):
        """
        select all columns of given names in a batch
        NOTE: This is an efficiency boost over subsequent application of get_col
        (Pytorch slicing is unfortunately really limited)
        """
        indices = [list(range(self.columns._transition[col][0], self.columns._transition[col][1])) for col in cols]
        ttype = th.LongTensor if not self.is_cuda else th.cuda.LongTensor
        tformat = "bs*t*v"
        return self.data._transition[:,:,ttype([item for sublist in indices for item in sublist])], tformat

    def _to_cuda(self):
        self.data._transition = self.data._transition.cuda()
        pass

    def _share_mem(self):
        if hasattr(self.data, "_transition"):
            self.data._transition.share_memory_()
        if hasattr(self.data, "_episode"):
            self.data._episode.share_memory_()
        pass

    def __del__(self):
        pass

    def __len__(self):
        return self.data._transition.shape[0]

    def __iter__(self):
        self.__iter_idx = 0
        return self

    def __next__(self):
        if self.__iter_idx == self.data._transition.shape[0]:
            raise StopIteration
        self.__iter_idx += 1
        return self.data._transition[self.__iter_idx-1, :, :]

    def insert(self, seq_buf_batch_obj, scope="transition", bs_ids=None, t_ids=None, bs_empty=None):
        """
        insert into scope data given bs_ids and t_ids
        """
        assert isinstance(seq_buf_batch_obj, BatchEpisodeBuffer), \
                        "seq_buf_batch_obj needs to be of type BatchEpisodeBuffer"

        t_ids = [t_ids] if not isinstance(t_ids, (list, tuple)) else t_ids
        bs_ids = [bs_ids] if not isinstance(bs_ids, (list, tuple)) else bs_ids

        # perform device copies if necessary (might expand to handling multiple gpu ids)
        if scope in ["transition"]:
            data = seq_buf_batch_obj.data._transition.cuda() if self.is_cuda else seq_buf_batch_obj.data._transition.cpu()
            # now it gets interesting, as pytorch has chronic issues with indexing...
            if not self.data._transition.is_contiguous():
                self.data._transition = self.data._transition.contiguous() # no-op if already contiguous - there is no reason it should ever be non-contiguous

            if list(bs_ids) == list(range(0, self.data._transition.shape[0])):
                self.data._transition[:, t_ids, :] = data
            elif list(t_ids) == list(range(0, self.data._transition.shape[1])):
                self.data._transition[bs_ids, :, :] = data
            else:
                assert False, "insert for complicated id lists not implemented yet!"
                #for _i, _bs in bs_ids: # TODO: Make this nice using scatter!!
                #    self.data._transition[_bs, :, :] = data[_i,:,:]
                # proposed solution by AlbanD: view, and scatter coordinate-wise
                #view = self.data._transition.view(-1, self.data._transition.shape[2])
                #self.data._transition.scatter()

            # update seq_lens
            for bs_id in bs_ids: # maybe list comprehension but it's fast anyway
                #if bs_empty is not None and bs_id not in bs_empty:
                if (bs_empty is not None and bs_id not in bs_empty) or bs_empty is None:
                    self.seq_lens[bs_id] = max(self.seq_lens[bs_id], max(t_ids)+1)

        elif scope in ["episode"]:
            data = seq_buf_batch_obj.data._episode.cuda() if self.is_cuda else seq_buf_batch_obj.data._episode.cpu()
            if not self.data._episode.is_contiguous():
                self.data._episode = self.data._episode.contiguous()  # no-op if already contiguous - there is no reason it should ever be non-contiguous
            if list(bs_ids) == list(range(0, self.data._episode.shape[0])):
                self.data._episode[:, :] = data
            else:
                self.data._episode[bs_ids, :] = data

        return self

    def get_data(self, scope):
        if scope in ["transition"]:
            return self.data._transition
        elif scope in ["episode"]:
            return self.data._episode

    def fill(self, val):
        self.data._transition[:,:,:] = val
        pass

    def view(self, bs_ids, dict_of_schemes, output_format=None, **kwargs):
        input_type_scalar = False
        if isinstance(dict_of_schemes, dict):
            pass
        elif isinstance(dict_of_schemes, Scheme):
            dict_of_schemes = {"_":dict_of_schemes}
            input_type_scalar = True
        else:
            assert False, "unintelligible input format for dict_of_schemes - should be either Scheme or dict(str=Scheme)"

        t_id = kwargs.get("t_id", None)
        fill_zero = kwargs.get("fill_zero", False)

        def _view(scheme, bs_ids=bs_ids, kwargs=kwargs, cache_dict=None):
            # 1. view

            if bs_ids is None:
                bs_ids = list(range(0, len(self)))

            def _apply_transform(scheme_item, _data, _tformat, _scope):
                if scheme_item.get("transforms", None) is not None:
                    for _transform in scheme_item["transforms"]:
                        if callable(_transform[0]):
                            f_transform = _transform[0]
                        elif isinstance(_transform[0], str) and _transform[0] in TRANSFORMS:
                            f_transform = TRANSFORMS[_transform[0]]
                        else:
                            assert False, "Transform unknown!"
                        _data = f_transform(_data, **_transform[1], tformat=_tformat, output_size_only=False, scope=_scope)
                        pass
                return _data
                pass


            # adjust sequence lengths
            if t_id is None:
                adjusted_seq_lens = [ self.seq_lens[_bs] for _bs in bs_ids ]
            else:
                adjusted_seq_lens = [1 if self.seq_lens[_bs] > t_id else 0 for _bs in bs_ids]

            # doing a bit of scheme-fu here: could certainly be done a bit nicer, but it's fast anyway so who cares
            output_sizes = scheme.get_output_sizes(self._data_scheme)
            scheme_renamed = deepcopy(scheme)
            scheme_renamed.scheme_list = [{ _k: _scheme.get("rename", _v) if _k=="name" else _v  for _k, _v in _scheme.items()} for _scheme in scheme.scheme_list]
            cbh = BatchEpisodeBuffer(data_scheme=scheme_renamed,
                                     n_bs=len(bs_ids),
                                     n_agents=self.n_agents,
                                     n_t=1 if t_id is not None else self.data._transition.shape[1],
                                     is_cuda=self.is_cuda,
                                     is_shared_mem=True,
                                     sizes=output_sizes)

            for scheme_item in scheme.scheme_list:

                # ignore scheme_items that are switched off
                switch = scheme_item.get("switch", True)
                if not switch:
                    continue
                scope = scheme_item.get("scope", "transition")

                t_slice = None if t_id is None else slice(max(0, t_id-scheme.t_id_depth), t_id+1)
                _data, _data_format = self.get_col(col=scheme_item["name"],
                                                   scope=scope,
                                                   t=t_slice,
                                                   bs=bs_ids)

                data = _apply_transform(scheme_item=scheme_item,
                                        _data=_data,
                                        _tformat=_data_format,
                                        _scope=scope
                                       )

                # final slice if necessary
                if t_id is not None and scope in ["transition"]:
                    data = data[:, -1:, :]

                cbh.set_col(scheme_item.get("rename", scheme_item.get("name")), data, scope=scope)

            if fill_zero: # fill NaNs with zeros if requested
                if hasattr(cbh.data, "_transition"):
                    cbh.data._transition[cbh.data._transition!=cbh.data._transition] = 0.0
                if hasattr(cbh.data, "_episode"):
                    cbh.data._episode[cbh.data._episode!=cbh.data._episode] = 0.0

            cbh.seq_lens=adjusted_seq_lens

            ret = cbh
            return ret

        ret_dict = {}
        for scheme_name, scheme in  dict_of_schemes.items():
            ret_dict[scheme_name] = _view(scheme=scheme)

        if input_type_scalar:
            return ret_dict["_"], "bs*t*v"
        else:
            return ret_dict, "{?}*bs*t*v"
    pass

    def flush(self):
        self.data._transition.fill_(float("nan"))
        self.seq_lens = [0 for _ in range(len(self))]

        # treat agent_id columns specially
        for _col, _vals in self.columns._transition.items():
            #if _col[:8] == "agent_id":
            #    self.data._transition[:,:,_vals[0]:_vals[1]] = float(_col.split("__agent")[1])
            if _col[:15] == "agent_id_onehot":
                self.data._transition[:,:,_vals[0]:_vals[1]] = _onehot(float(_col.split("__agent")[1]),
                                                                       rng=(0, _vals[1]-_vals[0]),
                                                                       is_cuda=self.data._transition.is_cuda)
            elif _col[:8] == "agent_id":
               self.data._transition[:,:,_vals[0]:_vals[1]] = float(_col.split("__agent")[1])
        pass

    def to_cuda(self):
        self.data._transition = self.data._transition.cuda()
        return self

    def to_pd(self):
        """
        give pandas DataFrame representation of contiguous batch sequence object
        FOR DEBUGGING ONLY - Pandas is slow as f***
        """
        import pandas as pd
        cols_transition = [_entry["name"] for _entry in self._data_scheme.scheme_list if _entry.get("switch", True) and _entry.get("scope", "transition")=="transition"]
        cols_episode = [_entry["name"] for _entry in self._data_scheme.scheme_list if
                           _entry.get("switch", True) and _entry.get("scope", "transition") == "episode"]
        transition_pds =     [ pd.DataFrame(columns=cols_transition,
                              data=[[self.data._transition[_i, _j, self.columns._transition[col][0]:self.columns._transition[col][1]].cpu().numpy().tolist()
                                     if len(self.data._transition[_i, _j, self.columns._transition[col][0]:self.columns._transition[col][1]].cpu().numpy().tolist()) > 1 else self.data._transition[_i, _j, self.columns._transition[col][0]:self.columns._transition[col][1]].cpu().numpy().tolist()[0] for col in cols_transition] for _j in range(self._n_t)],
                              index=list(range(0, self._n_t))) for _i in range(len(self)) ]
        #episode_pds =        [ pd.DataFrame(columns=cols_episode,
        #                      data=[[self.data._episode[_i, _j, self.columns._episode[col][0]:self.columns._episode[col][1]].cpu().numpy().tolist()
        #                             if len(self.data._episode[_i, _j, self.columns._episode[col][0]:self.columns._episode[col][1]].cpu().numpy().tolist()) > 1 else self.data._episode[_i, _j, self.columns._episode[col][0]:self.columns._episode[col][1]].cpu().numpy().tolist()[0] for col in cols_episode] for _j in range(self._n_t)],
        #                      index=list(range(0, self._n_t))) for _i in range(len(self)) ]
        #return [dict(_transition=_tr, episode=_ep) for _tr, _ep in zip(transition_pds, episode_pds) ]
        return [_tr for _tr in transition_pds]

    def get_stat(self, label, **kwargs):
        """
        calculate statistics "label" on all elements of batch set
        note that sometimes, batch-wise operations might speed up invidual history operations.
        also the batch view allows for returned elements to be appropriately padded
        """
        to_cuda = kwargs.get("to_cuda", False)
        to_variable = kwargs.get("to_variable", False)
        bs = kwargs.get("bs", None)


        if label in []:
            pass # batch-wise or custom statistics go in here
        elif label in ["reward_sum"]:
            tmp, _ = self.get_col(bs=bs, col="reward")
            tmp = tmp.cpu().numpy()
            stats = np.nansum(tmp, axis=1).squeeze(1)
            return stats
        elif label in ["episode_length"]:
            return self.seq_lens
        elif label in ["policy_entropy"]:
            policy_label = kwargs.get("policy_label", None)
            if policy_label is None:
                entropies = [ np.nanmean(np.nansum((-th.log(self["policies__agent{}".format(_aid)][0]) *
                                                    self["policies__agent{}".format(_aid)][0]).cpu().numpy(), axis=2)) for _aid in range(self.n_agents)]
            else:
                entropies = [np.nanmean(np.nansum((-th.log(self["{}__agent{}".format(policy_label, _aid)][0]) *
                                                   self["{}__agent{}".format(policy_label, _aid)][0]).cpu().numpy(), axis=2))
                             for _aid in range(self.n_agents)]
            return np.asscalar(np.mean(entropies))
        elif label in ["qvalues_entropy"]:
            entropies = [ np.nanmean(np.nansum((-th.log(self["qvalues__agent{}".format(_aid)][0]) * self["qvalues__agent{}".format(_aid)][0]).cpu().numpy(), axis=2)) for _aid in range(self.n_agents)]
            return np.asscalar(np.mean(entropies))
        elif label in ["td_lambda_targets"]:
            gamma = kwargs.get("gamma")
            td_lambda = kwargs.get("td_lambda")
            n_agents = kwargs.get("n_agents", self.n_agents)
            value_function_values = kwargs.get("value_function_values") # else may get weird variable sharing errors

            stats, tformat = self._td_lambda_targets(value_function_values,
                                                     n_agents=n_agents,
                                                     gamma=gamma,
                                                     td_lambda=td_lambda)
            return stats, tformat
        else:
            assert False, "Unknown batch statistic: {}".format(label)
        pass

    def _td_lambda_targets(self, value_function_values, n_agents, gamma, td_lambda):
        """
        uses efficient update rule for calculating consecutive G_t_n_lambda in reverse (see docs)
        """

        V_tensor = value_function_values
        if isinstance(V_tensor, Variable):
            V_tensor = V_tensor.data
        V_tensor = V_tensor.clone()

        R_tensor, _ = self.get_col(col="reward")
        if isinstance(R_tensor, Variable):
            R_tensor = R_tensor.data
        R_tensor = R_tensor.clone().unsqueeze(0).repeat(n_agents, 1, 1, 1)
        h = self._n_t-1 # horizon (index)

        # create truncation tensor
        truncated, _ = self.get_col(col="truncated")
        truncated = truncated.clone() # clone this because we are de-NaNing in place
        truncated[truncated!=truncated] = 0.0 # need to mask NaNs
        truncated_tensor = truncated.sum(dim=1, keepdim=True).unsqueeze(0).repeat(n_agents,1,1,1)

        seq_lens = self.seq_lens

        def _align_right(tensor, h, lengths):
            for _i, _l in enumerate(lengths):
                if _l < h + 1 and _l > 0:
                    tensor[:, _i, -_l:, :] = tensor[:, _i, :_l, :].clone() # clone is super important as otherwise, cyclical reference!
                    tensor[:, _i, :(h + 1 - _l), :] = float("nan")  # not strictly necessary as will shift back anyway later...
            return tensor

        def _align_left(tensor, h, lengths):
            for _i, _l in enumerate(lengths):
                if _l < h + 1 and _l > 0:
                    tensor[:, _i, :_l, :] = tensor[:, _i, -_l:, :].clone() # clone is super important as otherwise, cyclical reference!
                    tensor[:, _i, -(h + 1 - _l):, :] = float(
                        "nan")  # not strictly necessary as will shift back anyway later...
            return tensor

        R_tensor = _align_right(R_tensor, h, seq_lens)
        V_tensor = _align_right(V_tensor, h, seq_lens)

        G_buffer = R_tensor.clone() * float("nan")
        G_buffer[:, :, h - 1, :] = R_tensor[:, :, h, :]
        G_buffer[:, :, h - 1:h, :] += gamma * V_tensor[:, :, h:, :] * truncated_tensor

        for t in range(h - 1, 0, -1):
            new_G = gamma * td_lambda * G_buffer[:, :, t, :] + gamma*(1-td_lambda) * V_tensor[:, :, t, :] + R_tensor[:, :, t, :]
            G_buffer[:, :, t - 1, :] = new_G

        G_buffer = _align_left(G_buffer, h, seq_lens)
        return G_buffer, "a*bs*t*v"
