from itertools import combinations
#from torch.autograd import Variable
import torch as th
import numpy as np


def _n_agent_pair_samples(n_agents):
    return n_agents // 2

def _ordered_agent_pairings(n_agents):
    return sorted(combinations(list(range(n_agents)), 2))

def all_pairs(lst, pair_map, max_iter=None, shuffle_it=False, rand_state=None):
   if len(lst) < 2:
       yield []
       return
   if len(lst) % 2 == 1:
       # Handle odd length list
       _id_lst = list(range(len(lst)))
       if max_iter:
           _id_lst = _id_lst[:max_iter]
       if shuffle_it:
           rand_state.shuffle(_id_lst)
       for i in _id_lst:
           for result in all_pairs(lst[:i] + lst[i+1:], pair_map, max_iter, shuffle_it, rand_state):
               yield result
   else:
       a = lst[0] # TODO: might want to shuffle initial index as well!
       _id_lst = list(range(1, len(lst)))
       if max_iter:
           _id_lst = _id_lst[:max_iter]
       if shuffle_it:
           rand_state.shuffle(_id_lst)
       for i in _id_lst:
           pair = (a,lst[i])
           for rest in all_pairs(lst[1:i]+lst[i+1:], pair_map, max_iter, shuffle_it, rand_state):
               yield [pair_map[pair]] + rest

rnd_seed = None
def _ordered_2_agent_pairings(n_agents, args=None):
    global rnd_seed
    lst = _ordered_agent_pairings(n_agents)
    pair_map = {pair: i for i, pair in enumerate(lst)}
    shuffle_it = getattr(args, "pair_partition_shuffle", False)
    max_iter = getattr(args, "pair_partition_max_iter", None)
    fix_partition_size = getattr(args, "fix_partition_size", False)
    if rnd_seed is None:
        rnd_seed = np.random.randint(0, np.iinfo(np.uint32).max, 1, np.uint32)[0]
    rnd_state = np.random.RandomState(rnd_seed)
    if not fix_partition_size:
        partn = list(all_pairs(list(range(n_agents)),
                               pair_map,
                               max_iter=max_iter,
                               shuffle_it=shuffle_it,
                               rand_state=rnd_state))
        return partn
    else:
        partn = list(all_pairs(list(range(n_agents)),
                               pair_map,
                               max_iter=False,
                               shuffle_it=False,
                               rand_state=None))
        rnd_state.shuffle(partn)
        return partn[:fix_partition_size]

def _excluded_pair_ids(n_agents, sampled_pair_ids):
    pairings = _ordered_agent_pairings(n_agents)
    tmp = [_i for _i, _pair in enumerate(pairings) if not any([{*pairings[_s]} & {*_pair} for _s in sampled_pair_ids ])]
    return tmp

def _n_agent_pairings(n_agents, args=None):
    # Number of ways to select 1 pair from n_agents
    return int((n_agents * (n_agents-1)) / 2)

def _n_2_agent_pairings(n_agents):
    # Number of ways to select 2 pairs from n_agents
    if n_agents > 3:
        return int((n_agents) * (n_agents - 1) * (n_agents - 2) * (n_agents - 3) / 8)
    else:
        return int((n_agents * (n_agents-1)) / 2)

def _joint_actions_2_action_pair(joint_action, n_actions,use_delegate_action=True):
    if isinstance(joint_action, int):
        if use_delegate_action:
            if joint_action != 0.0:
                _action1 = (joint_action - 1.0) // n_actions
                _action2 = (joint_action - 1.0) % n_actions
            else:
                _action1 = float("nan")
                _action2 = float("nan")
        else:
            _action1 = th.floor(joint_action / n_actions)
            _action2 = (joint_action) % n_actions
        return _action1, _action2
    else:
        if use_delegate_action:
            mask = (joint_action == 0.0)
            joint_action[mask] = 1.0
            _action1 = th.floor((joint_action-1.0) / n_actions)
            _action2 = (joint_action-1.0) % n_actions
            _action1[mask] = float("nan")
            _action2[mask] = float("nan")
        else:
            _action1 = th.floor(joint_action / n_actions)
            _action2 = (joint_action) % n_actions
        return _action1, _action2

def _joint_actions_2_action_pair_aa(joint_action, n_actions, avail_actions1, avail_actions2, use_delegate_action=True):
    joint_action = joint_action.clone()
    if use_delegate_action:
        mask = (joint_action == 0.0)
        joint_action[mask] = 1.0
        _action1 = th.floor((joint_action-1.0) / n_actions)
        _action2 = (joint_action-1.0) % n_actions
        _action1[mask] = float("nan")
        _action2[mask] = float("nan")
    else:
        _action1 = th.floor(joint_action / n_actions)
        _action2 = (joint_action) % n_actions

    aa_m1 = _action1 != _action1
    aa_m2 = _action2 != _action2
    _action1[aa_m1] = 0
    _action2[aa_m2] = 0
    aa1 = avail_actions1.data.gather(-1, ( _action1.long() ))
    aa2 = avail_actions2.data.gather(-1, ( _action2.long() ))
    _action1[aa1 == 0] = float("nan")
    _action2[aa2 == 0] = float("nan")
    _action1[aa_m1] = float("nan")
    _action2[aa_m2] = float("nan")
    return _action1, _action2

def _action_pair_2_joint_actions(action_pair, n_actions):
    assert action_pair[0].max() < n_actions and action_pair[1].max() < n_actions , "Input outside action range: {}, {} but should be < {}".format(action_pair[0].max(),
                                                                                                                                                  action_pair[1].max(),
                                                                                                                                                  n_actions)
    return action_pair[0] * n_actions + action_pair[1]

def _pairing_id_2_agent_ids(pairing_id, n_agents):
    all_pairings = _ordered_agent_pairings(n_agents)
    return all_pairings[pairing_id]

def _pairing_id_2_agent_ids__tensor(pairing_id, n_agents, tformat):
    assert tformat in ["a*bs*t*v"], "invalid tensor input format"
    pairing_list = _ordered_agent_pairings(n_agents)
    ttype = th.cuda.LongTensor if pairing_id.is_cuda else th.LongTensor
    ids1 = ttype(pairing_list)[:, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(pairing_id.shape[0], pairing_id.shape[1], pairing_id.shape[2],1)
    ids2 = ttype(pairing_list)[:, 1].unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(pairing_id.shape[0], pairing_id.shape[1], pairing_id.shape[2],1)
    ret0 = ids1.gather(-1, pairing_id.long())
    ret1 = ids2.gather(-1, pairing_id.long())
    return ret0, ret1

def _agent_ids_2_pairing_id(agent_ids, n_agents):
    agent_ids = tuple(agent_ids)
    all_pairings = _ordered_agent_pairings(n_agents)
    assert agent_ids in all_pairings, "agent_ids is not of proper format!"
    return all_pairings.index(agent_ids)

# simple tests to establish correctness of encoding functions
if __name__ == "__main__":
    print(len(_ordered_2_agent_pairings(11)))
    print(len(_ordered_2_agent_pairings(12)))
