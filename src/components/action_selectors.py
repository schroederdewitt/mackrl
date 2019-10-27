import numpy as np
import torch as th
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn.functional import softmax
from .transforms import _to_batch, _from_batch, _adim, _vdim, _bsdim, _check_nan

REGISTRY = {}

class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args
        self.output_type = "policies"
        pass

    def select_action(self, inputs, avail_actions, tformat, test_mode=False):
        assert tformat in ["a*bs*t*v"], "invalid format!"

        if isinstance(inputs["policies"], Variable):
            agent_policies = inputs["policies"].data.clone()
        else:
            agent_policies = inputs["policies"].clone()  # might not be necessary

        if avail_actions is not None:
            """
            NOTE: MULTINOMIAL ACTION SELECTION  is usually performed by on-policy algorithms.
            ON-POLICY mean that avail_actions have to be handled strictly within the model, and need to form part
            of the backward pass.
            However, sometimes, numerical instabilities require to use non-zero masking (i.e. using tiny values) of the
            unavailable actions in the model - else the backward might erratically return NaNs.
            In this case, non-available actions may be hard-set to 0 in the action selector. The off-policy shift that
            this creates can usually be assumed to be extremely tiny.
            """
            _sum = th.sum(agent_policies * avail_actions, dim=_vdim(tformat), keepdim=True)
            _sum_mask = (_sum == 0.0)
            _sum.masked_fill_(_sum_mask, 1.0)
            masked_policies = agent_policies * avail_actions / _sum

            # if no action is available, choose an action uniformly anyway...
            masked_policies.masked_fill_(_sum_mask.repeat(1, 1, 1, avail_actions.shape[_vdim(tformat)]),
                                         1.0 / avail_actions.shape[_vdim(tformat)])
            # throw debug message
            if th.sum(_sum_mask) > 0:
                if self.args.debug_verbose:
                    print('Warning in MultinomialActionSelector.available_action(): some input policies sum up to 0!')
        else:
            masked_policies = agent_policies
        masked_policies_batch, params, tformat = _to_batch(masked_policies, tformat)

        _check_nan(masked_policies_batch)
        mask = (masked_policies_batch != masked_policies_batch)
        masked_policies_batch.masked_fill_(mask, 0.0)
        assert th.sum(masked_policies_batch < 0) == 0, "negative value in masked_policies_batch"

        a = masked_policies_batch.cpu().numpy()
        try:
            if not test_mode:
                _samples = Categorical(masked_policies_batch).sample().unsqueeze(1).float()
            else:
                _samples = th.argmax(masked_policies_batch, dim=1).unsqueeze(1).float()
        except RuntimeError as e:
            print('Warning in MultinomialActionSelector.available_action(): Categorical throws error {}!'.format(e))
            masked_policies_batch.random_(0, 2)
            _samples = Categorical(masked_policies_batch).sample().unsqueeze(1).float()
            pass

        _samples = _samples.masked_fill_(mask.long().sum(dim=1, keepdim=True) > 0, float("nan"))

        samples = _from_batch(_samples, params, tformat)
        _check_nan(samples)

        return samples, masked_policies, tformat

REGISTRY["multinomial"] = MultinomialActionSelector

class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args
        self.output_type = "qvalues"

    def _get_epsilons(self):
        assert False, "function _get_epsilon must be overwritten by user in runner!"
        pass

    def select_action(self, inputs, avail_actions, tformat, test_mode=False):
        assert tformat in ["a*bs*t*v"], "invalid format!"

        if isinstance(inputs["qvalues"], Variable):
            agent_qvalues = inputs["qvalues"].data.clone()
        else:
            agent_qvalues = inputs["qvalues"].clone() # might not be necessary

        # greedy action selection
        assert avail_actions.sum(dim=_vdim(tformat)).prod() > 0.0, \
            "at least one batch entry has no available action!"

        # mask actions that are excluded from selection
        agent_qvalues[avail_actions == 0.0] = -float("inf") # should never be selected!

        masked_qvalues_batch, params, tformat = _to_batch(agent_qvalues, tformat)
        _, _argmaxes = masked_qvalues_batch.max(dim=1, keepdim=True)

        if not test_mode: # normal epsilon-greedy action selection
            epsilons, epsilons_tformat = self._get_epsilons()
            random_numbers = epsilons.clone().uniform_()
            _avail_actions, params, tformat = _to_batch(avail_actions, tformat)
            random_actions = Categorical(_avail_actions).sample().unsqueeze(1)
            epsilon_pos = (random_numbers < epsilons).repeat(agent_qvalues.shape[_adim(tformat)], 1) # sampling uniformly from actions available
            epsilon_pos = epsilon_pos[:random_actions.shape[0], :]
            _argmaxes[epsilon_pos] = random_actions[epsilon_pos]
            eps_argmaxes = _from_batch(_argmaxes, params, tformat)
            return eps_argmaxes, agent_qvalues, tformat
        else: # don't use epsilon!
            # sanity check: there always has to be at least one action available.
            argmaxes = _from_batch(_argmaxes, params, tformat)
            return argmaxes, agent_qvalues, tformat

REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
