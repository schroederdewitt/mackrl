import numpy as np
import torch as th

from torch.autograd import Variable

from utils.mackrel import _joint_actions_2_action_pair


def _has_gradient(tensor):
    """
    pytorch < 0.4 compatibility
    """
    if hasattr(tensor, "requires_grad"):
        return tensor.requires_grad
    else:
        return False

def _onehot(tensor, rng, is_cuda=None):
    if isinstance(tensor, float):
        tt = th.FloatTensor if not is_cuda else th.cuda.FloatTensor
        newt = tt(len(list(range(*rng))))
        newt.zero_()
        newt[int(tensor)] = 1.0
    else:
        newt = tensor.clone().repeat(*([1]*(len(tensor.shape)-1) + [len(list(range(*rng)))]))
        newt.zero_()
        newt.scatter_(len(tensor.shape) - 1, tensor.long(), 1)
    return newt

def fillnan(series, value, **kwargs):
    """
    fill nan values in series with value
    """
    return series.fillna(value)


def _join_dicts(*dicts):
    """
    joins dictionaries together, while checking for key conflicts
    """
    key_pool = set()
    for _d in dicts:
        new_keys = set(_d.keys())
        a = key_pool.intersection(new_keys)
        if key_pool.intersection(new_keys) != set():
            assert False, "ERROR: dicts to be joined have overlapping keys. Common hint: Have you specified your scheme dicts properly?"
        else:
            key_pool.update(new_keys)

    ret = {}
    for _d in dicts:
        ret.update(_d)

    return ret


def _to_batch(item, tformat):
    assert tformat in ["a*bs*t*v", "bs*t*v"], "unknown format: {}".format(tformat)
    if isinstance(item, dict):
        res_dict = {}
        for _k, _v in item.items():
            res_dict[_k], params, tformat = _to_batch(_v, tformat)
        return res_dict, params, tformat
    else:
        if tformat in ["a*bs*t*v"]:
            a, bs, t, v = item.shape
            return item.view(a * bs * t, v), (a, bs, t, v), tformat
        elif tformat in ["bs*t*v"]:
            bs, t, v = item.shape
            return item.view(bs * t, v), (bs, t, v), tformat


def _from_batch(item, params, tformat):
    assert tformat in ["a*bs*t*v", "bs*t*v"], "unknown format: {}".format(tformat)
    if isinstance(item, dict):
        res_dict = {}
        for _k, _v in item.items():
            res_dict[_k] = _from_batch(_v, params, tformat)
        return res_dict
    else:
        if tformat in ["a*bs*t*v"]:
            a, bs, t, _ = params
            return item.view(a, bs, t, -1)
        elif tformat in ["bs*t*v"]:
            bs, t, _ = params
            return item.view(bs, t, -1)


def _check_inputs_validity(inputs, input_shapes, formats, allow_nonseq=False):
    # Check validity of key set supplied
    assert set(inputs.keys()) == set(input_shapes.keys()), \
        "unexpected set of inputs keys supplied: {}, {}".format(str(inputs.keys()), str(input_shapes.keys()))

    # Check format is known
    assert formats in ["a*bs*t*v"], "unknown format: {}".format(formats)

    # Check validity of individual input regions
    for _k, _v in inputs.items():
        # assert _v.dim() == 4, \
        #     "incorrect input dimensionality for {},(expected: {}, supplied: {}), " \
        #     "have you forgotten to add batch dimension or " \
        #     "have you erroneously added a sequence dimension?".format(_k,
        #                                                                4,
        #                                                                _v.dim())
        # assert _v.shape[-1] == input_shapes[_k] , \
        #     "incorrect input shape (expected: {}, supplied: {}".format(input_shapes[_k], _v[0].shape)

        assert (_v.data != _v.data).sum() == 0, "FATAL: np.nan detected in model input {}".format(_k)
        assert abs(_v.data.sum()) != np.inf, "FATAL: +/- np.inf detected in model input {}".format(_k)


def _check_nan_inf(series):
    is_nan = any((series != series))
    is_inf = any([str(_s) in ["inf", "-inf"] for _s in series])  # only reliable way I found!
    return is_nan, is_inf


def _naninfmean(tensor):
    _tensor = tensor.clone()
    if isinstance(tensor, Variable):
        _tensor = tensor.data
    _tensor[_tensor != _tensor] = 0.0
    _tensor[_tensor == float("nan")] = 0.0
    _tensor[_tensor == float("-nan")] = 0.0
    return _tensor.mean().item()


def _to_numpy_cpu(item):
    if issubclass(type(item), Variable):
        item = item.data
        if item.is_cuda:
            item = item.cpu()
        item = item.numpy()
        return item
    elif isinstance(item, np.ndarray):
        return item
    else:
        assert False, "unexpected input type: {}".format(str(item))


def _build_input(columns, inputs):
    model_inputs = {}
    for _k, _v in columns.items():
        model_inputs[_k] = th.cat([inputs[_c] for _c in _v], 2)
    return model_inputs


def _build_inputs(list_of_columns, inputs, format):
    model_inputs = []
    for _i, columns in enumerate(list_of_columns):
        model_inputs.append(_build_input(columns=columns,
                                         inputs=inputs))
    # join model inputs together
    inputs_dic = {}
    for _input in model_inputs:
        for _k, _v in _input.items():
            if _k not in inputs_dic:
                inputs_dic[_k] = [_v]
            else:
                inputs_dic[_k].append(_v)
    for _k in inputs_dic.keys():
        inputs_dic[_k] = th.cat(inputs_dic[_k], 1)
    return inputs_dic, format


def _stack_by_key(inputs, format):
    dic = {}
    for _k, _v in inputs[0].items():
        dic[_k] = []
        for item in inputs:
            dic[_k].append(item[_k])
        dic[_k] = th.cat(dic[_k], dim=0)
    return dic, format


def _split_batch(tensor, n_agents, format):
    if format in ["bs"]:
        return tensor.split(int(tensor.shape[0] / n_agents), dim=0), format
    elif format in ["sb"]:
        return tensor.split(int(tensor.shape[1] / n_agents), dim=1), format
    else:
        assert False, "unknown format!"


def _bsdim(tformat):
    _f = tformat.split("*")
    bsidx = [i for i, x in enumerate(_f) if x == "bs"]
    assert len(bsidx) == 1, "invalid tensor format string!"
    return bsidx[0]


def _tdim(tformat):
    _f = tformat.split("*")
    tidx = [i for i, x in enumerate(_f) if x == "t"]
    assert len(tidx) == 1, "invalid tensor format string!"
    return tidx[0]


def _adim(tformat):
    _f = tformat.split("*")
    aidx = [i for i, x in enumerate(_f) if x == "a"]
    assert len(aidx) == 1, "invalid tensor format string!"
    return aidx[0]


def _vdim(tformat):
    _f = tformat.split("*")
    aidx = [i for i, x in enumerate(_f) if x == "v"]
    assert len(aidx) == 1, "invalid tensor format string!"
    return aidx[0]


def _check_nan(input, silent_fail=True):
    retval = False
    from torch import nn
    if isinstance(input, dict):
        for _k1, _v1 in input.items():
            for _k2, _v2 in _v1.items():
                if th.sum(_v2.data != _v2.data):
                    retval = True
                    if not silent_fail:
                        assert False, "NaNs in {}:{}".format(_k1, _k2)
                    else:
                        _v2[_v2!=_v2] = 0.0 # fill NaNs with zeroes
                        print("NaNs in {}:{}".format(_k1, _k2))
    elif isinstance(input, (th.FloatTensor, th.DoubleTensor, th.HalfTensor, th.ByteTensor, th.CharTensor, th.ShortTensor, th.IntTensor, th.LongTensor,
                                  th.cuda.FloatTensor, th.cuda.DoubleTensor, th.cuda.HalfTensor, th.cuda.ByteTensor, th.cuda.CharTensor,
                                  th.cuda.ShortTensor, th.cuda.IntTensor, th.cuda.LongTensor)):
        if th.sum(input != input):
            retval = True
            if not silent_fail:
                assert False, "NaNs in tensor!"
            else:
                input[input != input] = 0.0 # fill NaNs with zeroes
                print("NaNs in tensor!")
    elif isinstance(input, th.autograd.Variable):
        if th.sum(input.data != input.data):
            retval = True
            if not silent_fail:
                assert False, "NaNs in Variable!"
            else:
                input[input!=input] = 0.0 # fill NaNs with zeroes
                print("NaNs in Variable!")
    elif issubclass(type(input), nn.Module):
        for i, p in enumerate(input.parameters()):
            print("grad:", p.grad)
            if th.sum(p.data != p.data):
                retval = True
                if not silent_fail:
                    assert False, "NaNs in parameter {}!".format(i)
                else:
                    p[p!=p] = 0.0 # fill NaNs with zeroes
                    print("NaNs in parameter {}!".format(i))
            if th.sum(p.grad != p.grad):
                retval=True
                if not silent_fail:
                    assert False, "NaNs in parameter gradient {}!".format(i)
                else:
                    p.grad[p.grad != p.grad] = 0.0 # fill NaNs with zeroes
                    print("NaNs in parameter gradient {}!".format(i))
    elif isinstance(input, list): # expect a list of parameters
        for p in input:
            if p is not None and p.grad is not None:
                if th.sum(p.data != p.data):
                    retval = True
                    if not silent_fail:
                        assert False, "NaNs in parameter {}!".format(p)
                    else:
                        p[p!=p] = 0.0 # fill NaNs with zeroes
                        print("NaNs in parameter {}!".format(p))
                if th.sum(p.grad.data != p.grad.data):
                    retval = True
                    if not silent_fail:
                        assert False, "NaNs in parameter gradient {}!".format(p)
                    else:
                        p.grad[p.grad != p.grad] = 0.0  # fill NaNs with zeroes
                        print("NaNs in parameter gradient {}!".format(p))

    return retval


def _pick_keys(dic, keys):
    """
    convenience function picking only certain keys from a dict and return as new dict with those keys only
    """
    return {_k: _v for _k, _v in dic.items() if _k in keys}


def _build_model_inputs(column_dict, inputs, inputs_tformat, to_variable=True, fill_zero=True, stack=True):
    """
    Takes in:
        - dict of CTSBs (i.e. input data per agent)
        - dict of dict of Schemes (i.e. input region scheme per agent per model input region)

    fills inputs into the relevant scheme fields and, if stack=True, stacks results along agent ids

    returns:
        {per-source-per-agent}{per-input region}[bs*t*v] (if not stacked)
        {per-source}{per-input region}[a*bs*t*v]

    NOTE: If inputs is passed in as autograd.Variable, then

    # TODO: Could be sped up if input regions are "glomped" together first - but this requires very sensitive handling!
    """

    ret_dict = {}
    assert inputs_tformat in ["{?}*bs*t*v"], "invalid input format!"
    # output_tformat = "[a]*bs*t*v"
    for _source_name, _input_regions in column_dict.items():
        if _source_name not in ret_dict:
            ret_dict[_source_name] = {}
        for _input_region, _scheme in _input_regions.items():
            input_region_list = []
            for _scheme_list_entry in _scheme.scheme_list:

                scope = _scheme_list_entry.get("scope", "transition")
                switch = _scheme_list_entry.get("switch", True)
                if not switch:
                    continue
                _data, _data_format = inputs[_source_name].get_col(col=_scheme_list_entry["name"],
                                                                   scope=scope)
                input_region_list.append(_data)

            try:
                input_region_list_cat = th.cat(input_region_list, dim=_vdim(_data_format))
            except RuntimeError as e:
               assert False, "Triggered Runtime error <{}>. Have you defined a scheme that features " \
                             "episode and transition data in the same input column and does NOT " \
                             "expand-transform the episode data along the t dimension?".format(e)

            if to_variable and not isinstance(input_region_list_cat, Variable):
                requires_grad = _scheme_list_entry.get("requires_grad", True)
                input_region_list_cat = Variable(input_region_list_cat, requires_grad=requires_grad)

            ret_dict[_source_name][_input_region] = input_region_list_cat

    if stack:
        # cluster source names by agent ids
        output_tformat = "a*bs*t*v"
        tmp_dic = {}
        for _k in ret_dict.keys():
            _s = _k.split("__agent")
            if len(_s) > 1:
                if not _s[0] in tmp_dic:
                    tmp_dic[_s[0]] = {}
                tmp_dic[_s[0]][int(_s[1])] = None
            else:
                tmp_dic[_s[0]] = {}
            pass

        for _k in list(tmp_dic.keys()):
            if _k in tmp_dic and tmp_dic[_k] is not {}:
                id_keys = list(tmp_dic[_k].keys())
                if len(id_keys) > 0:
                    # sort keys and "glom" results together
                    id_keys = sorted(id_keys)
                    if _k in ret_dict:
                        assert False, "error: cannot have global key that is also name of a per-agent key"
                    else:
                        ret_dict[_k] = {}
                    # for _input_region_k in column_dict[list(column_dict.keys())[0]].keys():
                    for _input_region_k in column_dict[_k + "__agent{}".format(0)].keys():
                        _vec = th.stack(
                            [ret_dict[_k + "__agent{}".format(_id_key)][_input_region_k] for _id_key in id_keys])
                        ret_dict[_k][_input_region_k] = _vec
                    for _id_key in id_keys:
                        del ret_dict[_k + "__agent{}".format(_id_key)]
    else:
        output_tformat = "bs*t*v"
    return ret_dict, output_tformat


def _agent_flatten(lst):
    pass


def _generate_scheme_shapes(transition_scheme, dict_of_schemes):
    """
    returns a dict of the same structure as schemes, but with each value being the (scalar) 1D length of the scheme element
    """
    scheme_shapes = {}
    for _k, scheme in dict_of_schemes.items():
        scheme_shapes[_k] = scheme.get_output_sizes(transition_scheme)
    return scheme_shapes


def _generate_input_shapes(input_columns, scheme_shapes):
    """
    generates 1D model input shape sizes given scheme_shapes and input_columns
    """
    input_shapes = {}
    for _input_column_k, _input_column_v in input_columns.items():
        input_shapes[_input_column_k] = {}
        for _input_column_region_k, _input_column_region_v in _input_column_v.items():
            input_shapes[_input_column_k][_input_column_region_k] = sum(
                [scheme_shapes[_input_column_k][_scheme_list_entry["name"]]
                 for _scheme_list_entry in _input_column_region_v.scheme_list
                 if _scheme_list_entry.get("switch", True)])

    return input_shapes


def _one_hot(ndarray_or_tensor, **kwargs):
    """
    Transforms each element of a pandas series into a one-hot encoding of itself
    One-hot dimensionality is specified by rng=(low, high) parameter
    """
    rng = kwargs.get("range", None)
    output_size_only = kwargs.get("output_size_only", False)

    if output_size_only:
        return (rng[1] - rng[0] + 1) * ndarray_or_tensor

    tensor = ndarray_or_tensor
    if not tensor.is_cuda:
        y_onehot = th.FloatTensor(*tensor.shape[:-1], (rng[1] - rng[0] + 1)).zero_()
    else:
        y_onehot = th.cuda.FloatTensor(*tensor.shape[:-1], (rng[1] - rng[0] + 1)).zero_()
    nan_mask = (tensor != tensor)
    tensor[nan_mask] = 0  # mask nans the simple way # DEBUG
    # try:
    y_onehot.scatter_(len(tensor.shape) - 1, tensor.long(), 1)
    # except Exception as e:
    #     a = y_onehot.cpu().numpy()
    #     b = tensor.cpu().numpy()
    #     pass
    if len(nan_mask.shape) > 0:
        y_onehot[nan_mask.repeat(1, 1, y_onehot.shape[2])] = 0  # set nans to zero
    return y_onehot


def _one_hot_pairwise(tensor, **kwargs):
    rng = kwargs.get("range", None)
    output_size_only = kwargs.get("output_size_only", False)
    if output_size_only:
        return 2 * (rng[1] - rng[0] + 1) * tensor
    # TODO: need to handle delegation action (set all zero)
    action1, action2 = _joint_actions_2_action_pair(tensor, (rng[1] - rng[0] + 1))
    one_hot_action1 = _one_hot(action1, range=rng)
    one_hot_action2 = _one_hot(action2, range=rng)
    assert tensor.dim() == 3, "wrong tensor dim"
    tmp = th.cat([one_hot_action1, one_hot_action2], dim=-1) # DEBUG
    return tmp


def _mask(tensor, **kwargs):
    output_size_only = kwargs.get("output_size_only", False)
    if output_size_only:
        return tensor
    fill = kwargs.get("fill", None)
    assert fill is not None, "require fill attribute"
    ret = tensor.clone()
    ret.fill_(fill)
    return ret


def _shift(ndarray_or_tensor, **kwargs):
    """
    Returns series with index shifted such that what used to be one time step behind now aligns with current time step
    Missing values are filled according to fill_missing
    """
    output_size_only = kwargs.get("output_size_only", False)
    tformat = kwargs.get("tformat")

    if output_size_only:
        return ndarray_or_tensor  # series[0] depicts the length of the series
    steps = kwargs.get("steps", None)
    fill = kwargs.get("fill", float("nan"))
    assert steps is not None, "Param nsteps not specified"
    tensor = ndarray_or_tensor.clone()
    if tensor.shape[_tdim(tformat)] <= abs(steps):  # t-dimension must be along axis 1!
        tensor[:, :, :] = fill
        return tensor
    if steps > 0:
        tensor[:, steps:, :] = tensor[:, 0:tensor.shape[1] - steps, :]
        tensor[:, :steps, :] = fill
    elif steps < 0:
        tensor[:, :steps, :] = tensor[:, -steps:, :]
        tensor[:, steps:, :] = fill
    elif steps == 0:
        return tensor
    return tensor


def _seq_mean(seq, length=100):
    """
    return mean of last <length> sequence elements. if sequence is shorter than <length> elements,
    just return average over everything that's there and return a warning
    """
    assert len(seq) > 0, "empty sequence! fix this."
    warning_msg = None if len(seq) >= length else "seq ({}) is shorter than length ({})".format(len(seq), length)
    ret = np.mean(seq[max(-length, -len(seq)):])
    return ret  # , warning_msg


def _underscore_to_cap(_str):
    """
    convenience function taking in an underscore-separated string and turns it into capitalized string
    e.g "policy_loss" -> "Policy loss:"
    """
    spl = _str.split("_")
    spl[0] = spl[0].capitalize()
    return " ".join(spl)


def _copy_remove_keys(dic, keys):
    """
    convenience function returning the copy a dict missing keys
    """
    new_dic = {_k: _v for _k, _v in dic.items() if _k not in keys}
    return new_dic


def _make_logging_str(dic):
    """
    convenience function turning a dic into a logging string
    """
    keys = sorted(dic.keys())
    logging_str = ", ".join(["{}={:g}".format(_k, dic[_k]) for _k in keys])
    return logging_str


TRANSFORMS = {"one_hot": _one_hot,
              "shift": _shift,
              "mask": _mask,
              "one_hot_pairwise": _one_hot_pairwise}


# "t_repeat": _t_repeat}

def _merge_dicts(a, b, path=None, overwrite=True):
    """
    like _join_dicts, but works for any nested levels
    copied from: https://stackoverflow.com/questions/7204805/dictionaries-of-dictionaries-merge
    """
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                _merge_dicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                if not overwrite:
                    raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
                else:
                    a[key] = b[key]
        else:
            a[key] = b[key]
    return a


def _unpack_random_seed(seeds, output_shape, gen_fn):
    """
    takes a random seed and generates and unpacks it into a tensor of random variables
    """
    # # generate epsilon of necessary
    # if is_seed: #self.args.pomace_use_epsilon_seed:

    assert len(output_shape) == 2, "output_shape must be of dim 2!"

    if seeds.is_cuda:  # inputs["pomace_epsilon_seeds"]
        _initial_rng_state_all = th.cuda.get_rng_state_all()
        # epsilon_variances.shape[_bsdim(tformat)],
        #                                            self.args.pomace_epsilon_size
        epsilons = th.cuda.FloatTensor(*output_shape)  # not sure if can do this directly on GPU using pytorch.dist...
        for _bs in range(output_shape[0]):  # (epsilon_variances.shape[0]):
            # inputs["pomace_epsilon_seeds"]
            th.cuda.manual_seed_all(int(seeds.data[_bs, 0]))
            # epsilons[_bs].normal_(mean=0.0, std=epsilon_variances.data[_bs, 0])
            epsilons[_bs] = gen_fn(out=epsilons[_bs],
                                   bs=_bs)
        th.cuda.set_rng_state_all(_initial_rng_state_all)
    else:
        _initial_rng_state = th.get_rng_state()
        epsilons = th.FloatTensor(*output_shape)
        for _bs in range(output_shape[0]):  # could use pytorch dist
            th.manual_seed(int(seeds.data[_bs, 0]))
            epsilons[_bs] = gen_fn(out=epsilons[_bs],
                                   bs=_bs)
        th.set_rng_state(_initial_rng_state)
    epsilons = Variable(epsilons, requires_grad=False)
    return epsilons

    # else:
    #     epsilons = inputs["epsilon"] * epsilon_variances
    #
    # else:
    # if inputs["pomace_epsilon_seeds"].is_cuda:
    #     epsilons = Variable(th.cuda.FloatTensor(epsilon_variances.shape[_bsdim(tformat)],
    #                                             self.args.pomace_epsilon_size).zero_(), requires_grad=False)
    # else:
    #     epsilons = Variable(th.FloatTensor(epsilon_variances.shape[_bsdim(tformat)],
    #                                        self.args.pomace_epsilon_size).zero_(), requires_grad=False)
    #     pass

def _n_step_return(values, rewards, truncated, terminated, n, gamma, horizon, seq_lens):
    """
    return n-step returns

    values expected to be in format a*bs*t*v
    all others as bs*t*v
    """

    def _align_right(tensor, h, lengths):
        for _i, _l in enumerate(lengths):
            if _l < h + 1 and _l > 0:
                tensor[:, _i, -_l:, :] = tensor[:, _i, :_l,
                                         :].clone()  # clone is super important as otherwise, cyclical reference!
                tensor[:, _i, :(h + 1 - _l), :] = float(
                    "nan")  # not strictly necessary as will shift back anyway later...
        return tensor

    def _align_left(tensor, h, lengths):
        for _i, _l in enumerate(lengths):
            if _l < h + 1 and _l > 0:
                tensor[:, _i, :_l, :] = tensor[:, _i, -_l:,:].clone()  # clone is super important as otherwise, cyclical reference!
                tensor[:, _i, -(h + 1 - _l):, :] = float("nan")  # not strictly necessary as will shift back anyway later...
        return tensor

    ttype = th.FloatTensor if not values.is_cuda else th.cuda.FloatTensor
    V_tensor = values.clone()
    R_tensor = rewards.clone().unsqueeze(0).repeat(V_tensor.shape[0], 1, 1, 1)
    TR_tensor = truncated.clone().unsqueeze(0).repeat(V_tensor.shape[0], 1, 1, 1)
    TE_tensor = terminated.clone().unsqueeze(0).repeat(V_tensor.shape[0], 1, 1, 1)
    V_tensor[(TR_tensor + TE_tensor)==1.0] = 0.0 # set values of terminal states to 0
    R_tensor = _align_right(R_tensor, horizon, seq_lens)
    V_tensor = _align_right(V_tensor, horizon, seq_lens)
    N_tensor = V_tensor.clone().fill_(float("nan"))
    # for _t in range(1, V_tensor.shape[2]):
    #     realizable_length = min(n, V_tensor.shape[2] - _t)
    #     R = R_tensor[:,:,_t:_t+realizable_length,:]
    #     power = ttype(list(range(realizable_length))).unsqueeze(0).unsqueeze(0).unsqueeze(3)
    #     G = ttype(V_tensor.shape[0], V_tensor.shape[1], realizable_length, 1).fill_(gamma) ** power
    #     V = (gamma ** (realizable_length-1)) * V_tensor[:,:,_t+realizable_length-1:_t+realizable_length,:]
    #     tmp = (R * G).sum(dim=2, keepdim=True)
    #     N_tensor[:,:,_t-1:_t,:] = (R * G).sum(dim=2, keepdim=True) + V
    #     x = 5
    for _t in range(V_tensor.shape[2]-1, 0, -1):
        realizable_length = min(n, V_tensor.shape[2] - _t)
        R = R_tensor[:,:,_t:_t+realizable_length,:]
        power = ttype(list(range(realizable_length))).unsqueeze(0).unsqueeze(0).unsqueeze(3)
        G = ttype(V_tensor.shape[0], V_tensor.shape[1], realizable_length, 1).fill_(gamma) ** power
        V = (gamma ** (realizable_length)) * V_tensor[:,:,_t+realizable_length-1:_t+realizable_length,:]
        tmp = (R * G).sum(dim=2, keepdim=True)
        N_tensor[:,:,_t-1:_t,:] = (R * G).sum(dim=2, keepdim=True) + V
        # x = 5
    N_tensor = _align_left(N_tensor, horizon, seq_lens)
    return N_tensor

def _pad(tensor, tformat, seq_lens, val):
    """
    Fill dummy states beyond episode limit with val (in_place)
    """
    assert tformat in ["a*bs*t*v", "bs*t*v"], "invalid tformat"
    bs_ids = range(tensor.shape[_bsdim(tformat)])
    for _bs in bs_ids:
        first_nan_t = seq_lens[_bs]
        if first_nan_t < tensor.shape[_tdim(tformat)]:
            if tformat in ["a*bs*t*v"]:
                tensor[:, _bs, first_nan_t:, :] = val
            elif tformat in ["bs*t*v"]:
                tensor[_bs, first_nan_t:, :] = val
    return tensor


def _pad_nan(tensor, tformat, seq_lens):
    """
    Fill dummy states beyond episode limit with NaNs (in_place)
    """
    return _pad(tensor, tformat, seq_lens, float("nan"))

def _pad_zero(tensor, tformat, seq_lens):
    """
    Fill dummy states beyond episode limit with zeros (in_place)
    """
    return _pad(tensor, tformat, seq_lens, 0.0)

