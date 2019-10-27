from copy import deepcopy
from functools import partial
from itertools import combinations
from models import REGISTRY as mo_REGISTRY
import numpy as np
from numpy.random import randint
import os
import torch as th
from torch import nn
from torch.autograd import Variable
from torch.optim import RMSprop

from debug.debug import IS_PYCHARM_DEBUG
from components.scheme import Scheme
from components.transforms import _adim, _bsdim, _tdim, _vdim, \
    _generate_input_shapes, _generate_scheme_shapes, _build_model_inputs, \
    _join_dicts, _seq_mean, _copy_remove_keys, _make_logging_str, _underscore_to_cap, _check_nan, _n_step_return, _pad_zero, _pad
from components.losses import EntropyRegularisationLoss
from components.transforms import _to_batch, \
    _from_batch, _naninfmean
from utils.mackrel import _n_agent_pairings, _agent_ids_2_pairing_id, _pairing_id_2_agent_ids, _n_agent_pair_samples, _agent_ids_2_pairing_id

from .basic import BasicLearner
from .centralV import CentralVLearner

class MACKRLPolicyLoss(nn.Module):

    def __init__(self):
        super(MACKRLPolicyLoss, self).__init__()

    def forward(self, policies, advantages, tformat, seq_lens, *args, **kwargs):
        assert tformat in ["a*bs*t*v"], "invalid input format!"
        n_agents = kwargs["n_agents"]

        policies = policies.clone()
        advantages = advantages.clone().detach().unsqueeze(0).repeat(policies.shape[0], 1, 1, 1)

        _pad_zero(policies, tformat, seq_lens)
        # last elements of advantages are NaNs
        mask = advantages.clone().fill_(1.0).byte()
        _pad_zero(mask, tformat, seq_lens)
        mask[:, :, :-1, :] = mask[:, :, 1:, :] # account for terminal NaNs of targets
        mask[:, :, -1, :] = 0.0  # handles case of seq_len=limit_len
        advantages[~mask] = 0.0

        _pad(policies, tformat, seq_lens, 1.0)
        policy_mask = (policies < 10e-40)
        log_policies = th.log(policies)
        log_policies[policy_mask] = 0.0

        nan_mask = policies.clone().fill_(1.0)
        _pad_zero(nan_mask, tformat, seq_lens)
        loss = - log_policies * advantages * nan_mask.float()
        norm = nan_mask.sum(_bsdim(tformat), keepdim=True)
        norm[norm == 0.0] = 1.0
        loss_mean = loss.sum(_bsdim(tformat), keepdim=True) / norm.float()
        loss_mean = loss_mean.squeeze(_vdim(tformat)).squeeze(_bsdim(tformat))

        loss_mean = loss_mean / n_agents

        output_tformat = "a*t"
        return loss_mean, output_tformat

class MACKRLLearner(CentralVLearner):

    def __init__(self, multiagent_controller, logging_struct=None, args=None):
        super().__init__(multiagent_controller, logging_struct, args)
        self.policy_loss_class = MACKRLPolicyLoss

