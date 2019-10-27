from copy import deepcopy
from .transforms import TRANSFORMS

SCHEME_CACHE  = {} # global scheme registry

class Scheme():

    def __init__(self, scheme_list, agent_flatten=True):
        self.scheme_list = scheme_list
        if agent_flatten:
            self.agent_flatten()  # NEW!
        self.t_id_depth = self._get_t_id_depth(self.scheme_list)
        return

    def _get_t_id_depth(self, scheme_list):
        # calculates required minimum sequence depth when t_id is set (based on shift transforms)
        min_depth = 0
        for _scheme in scheme_list:
            for _transf in _scheme.get("transforms", []):
                if _transf[0] in ["shift"]:
                    min_depth = max(_transf[1].get("steps", 0), min_depth)
        return min_depth

    def _agent_flatten_dict(self, dic):
        """
        flattens dict values that are lists into flat entries labelled by __agent<id>
        """
        dic = dict(dic) # copy
        for k in list(dic.keys()):
            if isinstance(dic[k], (tuple, list)):
                for i, v in enumerate(dic[k]):
                    dic[k+"__agent{:d}".format(i)] = v
                del dic[k]
        return dic

    def _check_scheme_history_compatibility(self, scheme_list, history):
        """
        check whether all history columns are found in the scheme
        """
        scheme_list_keys = [_s["name"] for _s in scheme_list if _s.get("switch", True)]
        history_keys = history.get_keys()
        for _k in scheme_list_keys:
            assert _k in history_keys, "key {} not in scheme_list".format(_k)
        pass

    def __contains__(self, item): # "in" operator
        _name_dict = { _s["name"]:None for _s in self.scheme_list}
        return item in _name_dict

    def get_by_name(self, name):
        """
        return scheme_list element with given name (list if there is more than one)
        """
        ret = []
        for _scheme_entry in self.scheme_list:
            if _scheme_entry["name"] == name:
                if isinstance(ret, (list, tuple)):
                    ret.append(_scheme_entry)
        return ret[0] if len(ret) == 1 else ret
    pass

    def agent_flatten(self):
        flat_scheme_list = []
        for _scheme in self.scheme_list:
            if _scheme.get("select_agent_ids", None) is not None:
                # remove agent data entries that are not applicable
                for _agent_id in _scheme.get("select_agent_ids"):
                    tmp_scheme = deepcopy(_scheme) #deepcopy(_scheme)
                    tmp_scheme["name"] = _scheme["name"] + "__agent{}".format(_agent_id)
                    if tmp_scheme.get("rename", None) is not None:
                        tmp_scheme["rename"] = _scheme["rename"] + "__agent{}".format(_agent_id)
                    del tmp_scheme["select_agent_ids"]
                    # remove transforms that are not applicable
                    if tmp_scheme.get("transforms", None) is not None:
                        for _id, transform in enumerate(tmp_scheme.get("transforms", None)):
                            if transform[1].get("select_agent_ids", None):
                                if _agent_id not in transform[1]["select_agent_ids"]:
                                    del tmp_scheme["transforms"][_id]
                                else:
                                    del tmp_scheme["transforms"][_id][1]["select_agent_ids"]
                    flat_scheme_list.append(tmp_scheme)
            else:
                flat_scheme_list.append(_scheme)

        self.scheme_list = flat_scheme_list
        return Scheme(flat_scheme_list, agent_flatten=False)

    def get_output_sizes(self, transition_scheme):
        """
        calculates output sizes given a transition scheme
        """
        def _apply_transform(scheme_item, _data):
            if scheme_item.get("transforms", None) is not None:
                for _transform in scheme_item["transforms"]:
                    if callable(_transform[0]):
                        f_transform = _transform[0]
                    elif isinstance(_transform[0], str) and _transform[0] in TRANSFORMS:
                        f_transform = TRANSFORMS[_transform[0]]
                    else:
                        assert False, "Transform unknown!"
                    _data = f_transform(_data, **_transform[1], output_size_only=True)
                    pass
            return _data

        transition_scheme_dic = {_item["name"]:_item for _item in transition_scheme.scheme_list}
        output_size_dic = {}
        for scheme_entry in self.scheme_list:
            if not scheme_entry.get("switch", True):
                continue
            if not scheme_entry["name"] in transition_scheme:
                assert False, "cannot find '{}' in transition_scheme - have you misspelled it?".format(scheme_entry["name"])
            input_size = transition_scheme_dic[scheme_entry["name"]]["shape"][0]
            output_size = _apply_transform(scheme_entry, input_size)
            output_size_dic[scheme_entry.get("rename", scheme_entry["name"])] = output_size
        return output_size_dic
