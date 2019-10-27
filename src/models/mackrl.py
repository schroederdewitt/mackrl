from itertools import combinations
import numpy as np
from torch.autograd import Variable
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import operator
import itertools
from functools import reduce

from components.transforms import _check_inputs_validity, _to_batch, _from_batch, \
    _adim, _bsdim, _tdim, _vdim, _check_nan, _pad
from models.basic import RNN as RecurrentAgent, DQN as NonRecurrentAgent
from utils.mackrel import _n_agent_pairings, _agent_ids_2_pairing_id, _ordered_agent_pairings, \
    _action_pair_2_joint_actions, _ordered_2_agent_pairings, _ordered_2_agent_pairings

class MACKRLVFunction(nn.Module):

    def __init__(self, input_shapes, n_agents, n_actions, output_shapes={}, layer_args={}, args=None):

        super(MACKRLVFunction, self).__init__()

        self.args = args
        self.n_agents = n_agents
        self.n_actions = n_actions

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        assert set(input_shapes.keys()) == {"main"}, \
            "set of input_shapes does not coincide with model structure!"
        self.input_shapes.update(input_shapes)

        # Set up output_shapes automatically if required
        self.output_shapes = {}
        self.output_shapes["vvalue"] = 1 # qvals
        self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["fc1"] = {"in":self.input_shapes["main"], "out":64}
        self.layer_args["fc2"] = {"in":self.layer_args["fc1"]["out"], "out":self.output_shapes["vvalue"]}
        self.layer_args.update(layer_args)

        # Set up network layers
        self.fc1 = nn.Linear(self.layer_args["fc1"]["in"], self.layer_args["fc1"]["out"])
        self.fc2 = nn.Linear(self.layer_args["fc2"]["in"], self.layer_args["fc2"]["out"])

        # DEBUG
        # self.fc2.weight.data.zero_()
        # self.fc2.bias.data.zero_()

    def init_hidden(self):
        """
        There's no hidden state required for this model.
        """
        pass

    def forward(self, inputs, tformat):
        # _check_inputs_validity(inputs, self.input_shapes, tformat, allow_nonseq=True)

        main, params, m_tformat = _to_batch(inputs.get("main"), tformat)
        x = F.relu(self.fc1(main))
        vvalue = self.fc2(x)
        return _from_batch(vvalue, params, m_tformat), m_tformat


class MACKRLCritic(nn.Module):

    """
    Concats MACKRLQFunction and MACKRLAdvantage together to an advantage and qvalue function
    """

    def __init__(self, input_shapes, n_actions, n_agents, output_shapes={}, layer_args={}, args=None):
        """
        This model contains no network layers but only sub-models
        """

        super(MACKRLCritic, self).__init__()
        self.args = args
        self.n_agents = n_agents
        self.n_actions = n_actions

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        self.input_shapes["avail_actions"] = self.n_actions
        self.input_shapes.update(input_shapes)

        # Set up output_shapes automatically if required
        self.output_shapes = {}
        self.output_shapes["advantage"] = 1
        self.output_shapes["vvalue"] = 1
        self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["vfunction"] = {}
        self.layer_args.update(layer_args)

        self.MACKRLVFunction = MACKRLVFunction(input_shapes={"main":self.input_shapes["vfunction"]},
                                                     output_shapes={},
                                                     layer_args={"main":self.layer_args["vfunction"]},
                                                     n_agents = self.n_agents,
                                                     n_actions = self.n_actions,
                                                     args=self.args)

        # self.MACKRLAdvantage = MACKRLAdvantage(input_shapes={"avail_actions":self.input_shapes["avail_actions"],
        #                                                "qvalues":self.MACKRLQFunction.output_shapes["qvalues"],
        #                                                "agent_action":self.input_shapes["agent_action"],
        #                                                "agent_policy":self.input_shapes["agent_policy"]},
        #                                  output_shapes={},
        #                                  n_actions=self.n_actions,
        #                                  args=self.args)

        pass

    def init_hidden(self):
        """
        There's no hidden state required for this model.
        """
        pass


    def forward(self, inputs, actions, tformat, baseline=True):
        #_check_inputs_validity(inputs, self.input_shapes, tformat)

        vvalue, vvalue_tformat = self.MACKRLVFunction(inputs={"main":inputs["vfunction"],
                                                                 "actions":actions,},
                                                         tformat=tformat)

        # advantage, qvalue, _ = self.MACKRLAdvantage(inputs={"avail_actions":qvalues.clone().fill_(1.0),
        #                                                  # critic level1 has all actions available forever
        #                                                  "qvalues":qvalues,
        #                                                  "agent_action":inputs["agent_action"],
        #                                                  "agent_policy":inputs["agent_policy"]},
        #                                          tformat=tformat,
        #                                          baseline=baseline)
        return {"vvalue":vvalue}, vvalue_tformat

class MLPEncoder(nn.Module):
    def __init__(self, input_shapes, output_shapes={}, layer_args={}, args=None):
        super(MLPEncoder, self).__init__()
        self.args = args

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        assert set(input_shapes.keys()) == {"main"}, \
            "set of input_shapes does not coincide with model structure!"
        self.input_shapes.update(input_shapes)

        # Set up layer_args automatically if required
        self.output_shapes = {}
        self.output_shapes["fc1"] = 64 # output
        self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["fc1"] = {"in":input_shapes["main"], "out":output_shapes["main"]}
        self.layer_args.update(layer_args)

        #Set up network layers
        self.fc1 = nn.Linear(self.input_shapes["main"], self.output_shapes["main"])
        pass

    def forward(self, inputs, tformat):

        x, n_seq, tformat = _to_batch(inputs["main"], tformat)
        x = F.relu(self.fc1(x))
        return _from_batch(x, n_seq, tformat), tformat


class MACKRLNonRecurrentAgentLevel1(NonRecurrentAgent):

    def forward(self, inputs, n_agents, tformat, loss_fn=None, hidden_states=None, **kwargs):
        test_mode = kwargs["test_mode"]

        avail_actions, params_aa, tformat_aa = _to_batch(inputs["avail_actions"], tformat)
        x, params, tformat = _to_batch(inputs["main"], tformat)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # mask policy elements corresponding to unavailable actions
        x = th.exp(x)
        x_sum = x.sum(dim=1, keepdim=True)
        second_mask = (x_sum <= np.sqrt(float(np.finfo(np.float32).tiny)) * x.shape[1])
        x_sum = x_sum.masked_fill(second_mask, 1.0)
        x = th.div(x, x_sum)

        # throw debug warning if second masking was necessary
        if th.sum(second_mask) > 0:
            if self.args.debug_verbose:
                print('Warning in MACKRLNonRecurrentAgentLevel1.forward(): some sum during the softmax has been 0!')

        # add softmax exploration (if switched on)
        # if self.args.mackrl_exploration_mode_level1 in ["softmax"] and not test_mode:
        #     epsilons = inputs["epsilons_central_level1"].unsqueeze(_tdim(tformat)).unsqueeze(0)
        #     epsilons, _, _ = _to_batch(epsilons, tformat)
        #     x = epsilons / _n_agent_pairings(n_agents) + x * (1 - epsilons)

        x = _from_batch(x, params, tformat)

        if loss_fn is not None:
            losses, _ = loss_fn(x, tformat=tformat)

        return x, hidden_states, losses, tformat

class MACKRLRecurrentAgentLevel1(nn.Module):

    def __init__(self, input_shapes, n_agents, output_type=None, output_shapes={}, layer_args={}, args=None, **kwargs):
        super(MACKRLRecurrentAgentLevel1, self).__init__()

        self.args = args
        self.n_agents = n_agents
        assert output_type is not None, "you have to set an output_type!"
        # self.output_type=output_type

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        # assert set(input_shapes.keys()) == {"main"}, \
        #     "set of input_shapes does not coincide with model structure!"
        self.input_shapes.update(input_shapes)

        # Set up layer_args automatically if required
        self.output_shapes = {}
        self.output_shapes["output"] = len(_ordered_2_agent_pairings(self.n_agents, self.args))
        if self.output_shapes is not None:
            self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["encoder"] = {"in":self.input_shapes["main"], "out":64}
        self.layer_args["gru"] = {"in":self.layer_args["encoder"]["out"], "hidden":64}
        self.layer_args["output"] = {"in":self.layer_args["gru"]["hidden"], "out":self.output_shapes["output"]}
        self.layer_args.update(layer_args)

        # Set up network layers
        self.encoder = MLPEncoder(input_shapes=dict(main=self.layer_args["encoder"]["in"]),
                                    output_shapes=dict(main=self.layer_args["encoder"]["out"]))
        self.gru = nn.GRUCell(self.layer_args["gru"]["in"], self.layer_args["gru"]["hidden"])
        self.output = nn.Linear(self.layer_args["output"]["in"], self.layer_args["output"]["out"])


    def forward(self, inputs, hidden_states, tformat, loss_fn=None, **kwargs):
        try:
            _check_inputs_validity(inputs, self.input_shapes, tformat)
        except Exception as e:
            print("Exception {} - have replaced NaNs with zeros".format(e))
            for _k, _v in inputs.items():
                inputs[_k][inputs[_k]!=inputs[_k]] = 0.0
            pass

        test_mode = kwargs["test_mode"]
        n_agents = kwargs["n_agents"]

        if len(inputs["main"].shape) == 3:
            _inputs = inputs["main"].unsqueeze(0) # as agent dimension is lacking
        else:
            _inputs = inputs["main"]
        #_inputs_aa = inputs["avail_actions"]

        loss = None
        t_dim = _tdim(tformat)
        assert t_dim == 2, "t_dim along unsupported axis"
        t_len = _inputs.shape[t_dim]

        x_list = []
        h_list = [hidden_states]

        for t in range(t_len):

            x = _inputs[:, :, slice(t, t + 1), :].contiguous()
            #avail_actions = _inputs_aa[:, :, slice(t, t + 1), :].contiguous()
            x, tformat = self.encoder({"main":x}, tformat)

            x, params_x, tformat_x = _to_batch(x, tformat)
            #avail_actions, params_aa, tformat_aa = _to_batch(avail_actions, tformat)
            h, params_h, tformat_h = _to_batch(h_list[-1], tformat)

            h = self.gru(x, h)
            x = self.output(h)

            # mask policy elements corresponding to unavailable actions
            #n_available_actions = avail_actions.detach().sum(dim=1, keepdim=True)

            # DEBUG
            x = F.softmax(x, 1)

            # Alternative variant
            #x = th.nn.functional.softmax(x).clone()
            #x.masked_fill_(avail_actions.long() == 0, float(np.finfo(np.float32).tiny))
            #x = th.div(x, x.sum(dim=1, keepdim=True))


            if self.args.mackrl_exploration_mode_level1 in ["softmax"] and not test_mode:
               epsilons = inputs["epsilons_central_level1"].unsqueeze(_tdim("bs*t*v")).detach()
               epsilons, _, _ = _to_batch(epsilons, "bs*t*v")
               x = epsilons /  x.shape[-1] + x * (1 - epsilons)

            if hasattr(self.args, "mackrl_fix_level1_pair") and self.args.mackrl_fix_level1_pair:
                #x.fill_(0.0)
                x[:,1:] = 0.0
                x[:, 0] = 1.0

            h = _from_batch(h, params_h, tformat_h)
            x = _from_batch(x, params_x, tformat_x)


            h_list.append(h)
            x_list.append(x)

        if loss_fn is not None:
            _x = th.cat(x_list, dim=_tdim(tformat))
            loss = loss_fn(_x, tformat=tformat)[0]

        return th.cat(x_list, t_dim), \
               th.cat(h_list[1:], t_dim), \
               loss, \
               tformat

class MACKRLNonRecurrentAgentLevel2(NonRecurrentAgent):

    def __init__(self, input_shapes, n_actions, output_type=None, output_shapes={}, layer_args={}, args=None, **kwargs):
        super().__init__(input_shapes,
        n_actions,
        output_type = output_type,
        output_shapes = dict(output=n_actions*n_actions + 2),
        layer_args = layer_args,
        args = args, ** kwargs)  # need to expand using no-op action

    def forward(self, inputs, tformat, loss_fn=None, hidden_states=None, **kwargs):
        test_mode = kwargs["test_mode"]

        avail_actions, params_aa, tformat_aa = _to_batch(inputs["avail_actions"], tformat)
        x, params, tformat = _to_batch(inputs["main"], tformat)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # mask policy elements corresponding to unavailable actions
        # n_available_actions = avail_actions.detach().sum(dim=1, keepdim=True)
        # x = th.exp(x)
        # x = x.masked_fill(avail_actions == 0, float(np.finfo(np.float32).tiny))
        # x = th.div(x, x.sum(dim=1, keepdim=True))

        n_available_actions = avail_actions.sum(dim=1, keepdim=True)
        x = th.exp(x)
        x = x.masked_fill(avail_actions == 0, np.sqrt(float(np.finfo(np.float32).tiny)))
        x_sum = x.sum(dim=1, keepdim=True)
        second_mask = (x_sum <= np.sqrt(float(np.finfo(np.float32).tiny)) * avail_actions.shape[1])
        x_sum = x_sum.masked_fill(second_mask, 1.0)
        x = th.div(x, x_sum)

        # throw debug warning if second masking was necessary
        if th.sum(second_mask) > 0:
            if self.args.debug_verbose:
                print('Warning in MACKRLNonRecurrentAgentLevel2.forward(): some sum during the softmax has been 0!')

        # add softmax exploration (if switched on)
        if self.args.coma_exploration_mode in ["softmax"] and not test_mode:
            epsilons = inputs["epsilons"].unsqueeze(_tdim(tformat))
            epsilons, _, _ = _to_batch(epsilons, tformat)
            x = avail_actions * epsilons / n_available_actions + x * (1 - epsilons)

        x = _from_batch(x, params, tformat)

        if loss_fn is not None:
            losses, _ = loss_fn(x, tformat=tformat)

        return x, hidden_states, losses, tformat

class MACKRLRecurrentAgentLevel2(nn.Module):

    def __init__(self, input_shapes, n_actions, output_type=None, output_shapes={}, layer_args={}, args=None, **kwargs):
        super(MACKRLRecurrentAgentLevel2, self).__init__()

        self.args = args
        self.n_actions = n_actions
        assert output_type is not None, "you have to set an output_type!"
        # self.output_type=output_type

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        # assert set(input_shapes.keys()) == {"main"}, \
        #     "set of input_shapes does not coincide with model structure!"
        self.input_shapes.update(input_shapes)

        # Set up layer_args automatically if required
        self.output_shapes = {}
        self.output_shapes["output"] = 2 + self.n_actions*self.n_actions # includes delegation action and no-op
        if self.output_shapes is not None:
            self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["encoder"] = {"in":self.input_shapes["main"], "out":64}
        self.layer_args["gru"] = {"in":self.layer_args["encoder"]["out"], "hidden":64}
        self.layer_args["output"] = {"in":self.layer_args["gru"]["hidden"], "out":self.output_shapes["output"]}
        self.layer_args.update(layer_args)

        # Set up network layers
        self.encoder = MLPEncoder(input_shapes=dict(main=self.layer_args["encoder"]["in"]),
                                    output_shapes=dict(main=self.layer_args["encoder"]["out"]))
        self.gru = nn.GRUCell(self.layer_args["gru"]["in"], self.layer_args["gru"]["hidden"])
        self.output = nn.Linear(self.layer_args["output"]["in"], self.layer_args["output"]["out"])

    def forward(self, inputs, hidden_states, tformat, loss_fn=None, **kwargs):
        seq_lens = kwargs["seq_lens"]

        try:
            _check_inputs_validity(inputs, self.input_shapes, tformat)
        except Exception as e:
            print("Exception {} - have replaced NaNs with zeros".format(e))
            for _k, _v in inputs.items():
                inputs[_k][inputs[_k]!=inputs[_k]] = 0.0
            pass

        test_mode = kwargs["test_mode"]
        pairwise_avail_actions = _pad(kwargs["pairwise_avail_actions"].detach(), tformat, seq_lens, 1.0)
        pairwise_avail_actions.requires_grad = False

        _inputs = inputs["main"]

        loss = None
        t_dim = _tdim(tformat)
        assert t_dim == 2, "t_dim along unsupported axis"
        t_len = _inputs.shape[t_dim]

        x_list = []
        h_list = [hidden_states]

        for t in range(t_len):

            x = _inputs[:, :, slice(t, t + 1), :].contiguous()
            avail_actions = pairwise_avail_actions[:, :, slice(t, t + 1), :].contiguous().detach()
            x, tformat = self.encoder({"main":x}, tformat)

            x, params_x, tformat_x = _to_batch(x, tformat)
            avail_actions, params_aa, tformat_aa = _to_batch(avail_actions, tformat)
            h, params_h, tformat_h = _to_batch(h_list[-1], tformat)

            h = self.gru(x, h)
            x = self.output(h)

            if getattr(self.args, "mackrl_logit_bias", 0.0) != 0.0:
                x = th.cat([x[:, 0:1] + self.args.mackrl_logit_bias, x[:, 1:]], dim=1)

            n_available_actions = avail_actions.sum(dim=1, keepdim=True)
            x = x - (1 - avail_actions) * 1e30
            x = F.softmax(x, 1)

            # add softmax exploration (if switched on)
            if self.args.mackrl_exploration_mode_level2 in ["softmax"] and not test_mode:
               epsilons = inputs["epsilons_central_level2"].unsqueeze(_tdim(tformat)).detach()
               epsilons, _, _ = _to_batch(epsilons, tformat)
               n_available_actions[n_available_actions==0.0] = 1.0
               x = avail_actions.detach() * epsilons / n_available_actions + x * (1 - epsilons)

            h = _from_batch(h, params_h, tformat_h)
            x = _from_batch(x, params_x, tformat_x)

            h_list.append(h)
            x_list.append(x)

        x_cat = th.cat(x_list, t_dim)

        if hasattr(self.args, "mackrl_always_delegate") and self.args.mackrl_always_delegate:
            x_cat[:, :, :, 0] = 1.0
            x_cat[:, :, :, 1:] = 0.0

        if loss_fn is not None:
            loss = loss_fn(x_cat, tformat=tformat)[0]

        return x_cat, \
               th.cat(h_list[1:], t_dim), \
               loss, \
               tformat

class MACKRLNonRecurrentAgentLevel3(NonRecurrentAgent):

    def __init__(self, input_shapes, n_actions, output_type=None, output_shapes={}, layer_args={}, args=None, **kwargs):
        super().__init__(input_shapes,
        n_actions,
        output_type = output_type,
        output_shapes = dict(output=n_actions + 1),
        layer_args = layer_args,
        args = args, ** kwargs)  # need to expand using no-op action


    def forward(self, inputs, tformat, loss_fn=None, hidden_states=None, **kwargs):
        test_mode = kwargs["test_mode"]

        avail_actions, params_aa, tformat_aa = _to_batch(inputs["avail_actions"], tformat)
        x, params, tformat = _to_batch(inputs["main"], tformat)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # mask policy elements corresponding to unavailable actions
        n_available_actions = avail_actions.sum(dim=1, keepdim=True)
        x = th.exp(x)
        x = x.masked_fill(avail_actions == 0, np.sqrt(float(np.finfo(np.float32).tiny)))
        x_sum = x.sum(dim=1, keepdim=True)
        second_mask = (x_sum <= np.sqrt(float(np.finfo(np.float32).tiny)) * avail_actions.shape[1])
        x_sum = x_sum.masked_fill(second_mask, 1.0)
        x = th.div(x, x_sum)

        # throw debug warning if second masking was necessary
        if th.sum(second_mask.data) > 0:
            if self.args.debug_verbose:
                print('Warning in MACKRLNonRecurrentAgentLevel3.forward(): some sum during the softmax has been 0!')

        # add softmax exploration (if switched on)
        if self.args.coma_exploration_mode in ["softmax"] and not test_mode:
            epsilons = inputs["epsilons"].unsqueeze(_tdim(tformat))
            epsilons, _, _ = _to_batch(epsilons, tformat)
            x = avail_actions * epsilons / n_available_actions + x * (1 - epsilons)

        x = _from_batch(x, params, tformat)

        if loss_fn is not None:
            losses, _ = loss_fn(x, tformat=tformat)

        return x, hidden_states, losses, tformat

class MACKRLRecurrentAgentLevel3(RecurrentAgent):

    def __init__(self, input_shapes, n_actions, output_type=None, output_shapes={}, layer_args={}, args=None, **kwargs):
        super().__init__(input_shapes,
        n_actions,
        output_type = output_type,
        output_shapes = dict(output=1+n_actions), # do include no-op!
        layer_args = layer_args,
        args = args, ** kwargs)

    def forward(self, inputs, hidden_states, tformat, loss_fn=None, **kwargs):
        seq_lens = kwargs["seq_lens"]

        try:
            _check_inputs_validity(inputs, self.input_shapes, tformat)
        except Exception as e:
            print("Exception {} - have replaced NaNs with zeros".format(e))
            for _k, _v in inputs.items():
                inputs[_k][inputs[_k]!=inputs[_k]] = 0.0
            pass

        test_mode = kwargs["test_mode"]

        _inputs = inputs["main"]
        _inputs_aa = _pad(inputs["avail_actions"], tformat, seq_lens, 1.0)

        loss = None
        t_dim = _tdim(tformat)
        assert t_dim == 2, "t_dim along unsupported axis"
        t_len = _inputs.shape[t_dim]

        x_list = []
        h_list = [hidden_states]

        for t in range(t_len):

            x = _inputs[:, :, slice(t, t + 1), :].contiguous()
            avail_actions = _inputs_aa[:, :, slice(t, t + 1), :].contiguous()
            x, tformat = self.encoder({"main":x}, tformat)

            x, params_x, tformat_x = _to_batch(x, tformat)
            avail_actions, params_aa, tformat_aa = _to_batch(avail_actions, tformat)
            h, params_h, tformat_h = _to_batch(h_list[-1], tformat)

            h = self.gru(x, h)
            x = self.output(h)

            # # mask policy elements corresponding to unavailable actions
            # n_available_actions = avail_actions.sum(dim=1, keepdim=True)
            # x = th.exp(x)
            # x = x.masked_fill(avail_actions == 0, np.sqrt(float(np.finfo(np.float32).tiny)))
            # x_sum = x.sum(dim=1, keepdim=True)
            # second_mask = (x_sum <= np.sqrt(float(np.finfo(np.float32).tiny))*avail_actions.shape[1])
            # x_sum = x_sum.masked_fill(second_mask, 1.0)
            # x = th.div(x, x_sum)

            n_available_actions = avail_actions.sum(dim=1, keepdim=True)
            x = x - (1 - avail_actions) * 1e30
            x = F.softmax(x, 1)

            # throw debug warning if second masking was necessary
            # if th.sum(second_mask.data) > 0:
            #     if self.args.debug_verbose:
            #         print('Warning in MACKRLRecurrentAgentLevel3.forward(): some sum during the softmax has been 0!')

            # add softmax exploration (if switched on)
            if self.args.mackrl_exploration_mode_level3 in ["softmax"] and not test_mode:
               epsilons = inputs["epsilons_central_level3"].unsqueeze(_tdim(tformat)).detach()
               epsilons, _, _ = _to_batch(epsilons, tformat)
               # n_available_actions[n_available_actions==0.0] = 1 #np.sqrt(float(np.finfo(np.float32).tiny))
               #if th.sum(th.sum(n_available_actions, dim=-1) > 0):
               #    a = 6
               #    pass
               #if th.sum(n_available_actions == 0.0) > 0:
               #    a = 5
               #    pass
               x = avail_actions.detach() * epsilons / n_available_actions + x * (1 - epsilons) # avail_actions * epsilons / n_available_actions + x * (1 - epsilons)

            h = _from_batch(h, params_h, tformat_h)
            x = _from_batch(x, params_x, tformat_x)

            h_list.append(h)
            x_list.append(x)

        if loss_fn is not None:
            _x = th.cat(x_list, dim=_tdim(tformat))
            loss = loss_fn(_x, tformat=tformat)[0]

        return th.cat(x_list, t_dim), \
               th.cat(h_list[1:], t_dim), \
               loss, \
               tformat

class MACKRLAgent(nn.Module):

    def __init__(self,
                 input_shapes,
                 n_agents,
                 n_actions,
                 args,
                 model_classes,
                 agent_output_type="policies",
                 **kwargs):
        super().__init__()

        self.args = args
        self.n_agents = n_agents
        self.n_actions = n_actions

        self.ordered_agent_pairings = _ordered_agent_pairings(self.n_agents)
        self.ordered_2_agent_pairings = _ordered_2_agent_pairings(self.n_agents, self.args)
        # _ordered_2_agent_pairings(self.n_agents)

        self.models = {}
        # set up models level 1 - these output p(u_a | tau_a)
        self.model_level1 = model_classes["level1"](input_shapes=input_shapes["level1"],
                                                    n_agents=n_agents,
                                                    output_type=agent_output_type,
                                                    args=args)
        if self.args.use_cuda:
            self.model_level1 = self.model_level1.cuda()

        # set up models level 2 - these output p(u_a, u_b|CK_01)
        if self.args.agent_level2_share_params:
            model_level2 = model_classes["level2"](input_shapes=input_shapes["level2"],
                                                   n_actions=n_actions,
                                                   output_type=agent_output_type,
                                                   args=self.args)
            if self.args.use_cuda:
                model_level2 = model_level2.cuda()

            for _agent_id1, _agent_id2 in sorted(combinations(list(range(n_agents)), 2)):
                self.models["level2_{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))] = model_level2
        else:
            assert False, "TODO"

        # set up models level 3 - these output p(ps=a,b)
        self.models_level3 = {}
        if self.args.agent_level3_share_params:
            model_level3 = model_classes["level3"](input_shapes=input_shapes["level3"],
                                                   n_actions=n_actions,
                                                   output_type=agent_output_type,
                                                   args=args)
            if self.args.use_cuda:
                model_level3 = model_level3.cuda()

            for _agent_id in range(n_agents):
                self.models["level3_{}".format(_agent_id)] = model_level3
        else:
            assert False, "TODO"

        # set up models for the pair probability modules - these output p(u_a, u_b|tau_0, tau_1, CK_01)

        # add models to parameter hook
        for _k, _v in self.models.items():
            setattr(self, _k, _v)
        pass


    def forward(self, inputs, actions, hidden_states, tformat, loss_fn=None, **kwargs):
        seq_lens = kwargs["seq_lens"]
        # TODO: How do we handle the loss propagation for recurrent layers??

        # generate level 1-3 outputs
        out_level1, hidden_states_level1, losses_level1, tformat_level1 = self.model_level1(inputs=inputs["level1"]["agent_input_level1"],
                                                                                            hidden_states=hidden_states["level1"],
                                                                                            loss_fn=None, #loss_fn,
                                                                                            tformat=tformat["level1"],
                                                                                            #n_agents=self.n_agents,
                                                                                            **kwargs)

        pairwise_avail_actions = inputs["level2"]["agent_input_level2"]["avail_actions_pair"]
        ttype = th.cuda.FloatTensor if pairwise_avail_actions.is_cuda else th.FloatTensor
        delegation_avails = Variable(ttype(pairwise_avail_actions.shape[0],
                                           pairwise_avail_actions.shape[1],
                                           pairwise_avail_actions.shape[2], 1).fill_(1.0), requires_grad=False)
        pairwise_avail_actions = th.cat([delegation_avails, pairwise_avail_actions], dim=_vdim(tformat["level2"]))
        out_level2, hidden_states_level2, losses_level2, tformat_level2 = self.models["level2_{}".format(0)](inputs=inputs["level2"]["agent_input_level2"],
                                                                                                            hidden_states=hidden_states["level2"],
                                                                                                            loss_fn=None, #loss_fn,
                                                                                                            tformat=tformat["level2"],
                                                                                                            pairwise_avail_actions=pairwise_avail_actions,
                                                                                                            **kwargs)

        out_level3, hidden_states_level3, losses_level3, tformat_level3 = self.models["level3_{}".format(0)](inputs["level3"]["agent_input_level3"],
                                                                                                            hidden_states=hidden_states["level3"],
                                                                                                            loss_fn=None,
                                                                                                            tformat=tformat["level3"],
                                                                                                            #seq_lens=seq_lens,
                                                                                                            **kwargs)


        # for each agent pair (a,b), calculate p_a_b = p(u_a, u_b|tau_a tau_b CK_ab) = p(u_d|CK_ab)*pi(u_a|tau_a)*pi(u_b|tau_b) + p(u_ab|CK_ab)
        # output dim of p_a_b is (n_agents choose 2) x bs x t x n_actions**2

        # Bulk NaN masking
        # out_level1[out_level1 != out_level1] = 0.0
        # out_level2[out_level2 != out_level2] = 0.0
        # out_level3[out_level3 != out_level3] = 0.0
        # actions = actions.detach()
        # actions[actions!=actions] = 0.0

        p_d = out_level2[:, :, :, 0:1]
        p_ab = out_level2[:, :, :, 1:-1]

        _actions = actions.clone()
        _actions[actions != actions] = 0.0
        pi = out_level3
        pi_actions_selected = out_level3.gather(_vdim(tformat_level3), _actions.long()).clone()
        # pi_actions_selected[pi_actions_selected  != pi_actions_selected ] = 0.0 #float("nan")
        pi_actions_selected_mask = (actions != actions)
        avail_actions_level3 = inputs["level3"]["agent_input_level3"]["avail_actions"].clone().data
        avail_actions_selected = avail_actions_level3.gather(_vdim(tformat_level3), _actions.long()).clone()

        # RENORMALIZE THIS STUFF
        pi_a_cross_pi_b_list = []
        pi_ab_list = []
        pi_corr_list = []

        # DEBUG: if prob is 0 throw a debug!
        # debug
        # 0 (0, 1)
        # 1 (0, 2)
        # 2 (1, 2)

        for _i, (_a, _b) in enumerate(self.ordered_agent_pairings): #_ordered_agent_pairings(self.n_agents)[:self.args.n_pair_samples] if hasattr("n_pair_samples", self.args) else _ordered_agent_pairings(self.n_agents)): #(_ordered_agent_pairings(self.n_agents)):
            # calculate pi_a_cross_pi_b # TODO: Set disallowed joint actions to NaN!
            pi_a = pi[_a:_a+1].clone()
            pi_b = pi[_b:_b+1].clone()
            x, params_x, tformat_x = _to_batch(out_level3[_a:_a+1], tformat["level3"])
            y, params_y, tformat_y  = _to_batch(out_level3[_b:_b+1], tformat["level3"])
            actions_masked = actions.clone()
            actions_masked[actions!=actions] = 0.0
            _actions_x, _actions_params, _actions_tformat = _to_batch(actions_masked[_a:_a+1].clone(), tformat["level3"])
            _actions_y, _actions_params, _actions_tformat = _to_batch(actions_masked[_b:_b+1].clone(), tformat["level3"])
            _x = x.gather(1, _actions_x.long())
            _y = y.gather(1, _actions_y.long())
            z = _x * _y
            u = _from_batch(z, params_x, tformat_x)
            pi_a_cross_pi_b_list.append(u)
            # calculate p_ab_selected
            _p_ab = p_ab[_i:_i+1].clone()

            joint_actions = _action_pair_2_joint_actions((actions_masked[_a:_a+1], actions_masked[_b:_b+1]), self.n_actions)
            _z = _p_ab.gather(_vdim(tformat_level2), joint_actions.long())
            # Set probabilities corresponding to jointly-disallowed actions to 0.0
            avail_flags = pairwise_avail_actions[_i:_i+1].gather(_vdim(tformat_level2), joint_actions.long() + 1).clone()
            _z[avail_flags==0.0] = 0.0 # TODO: RENORMALIZE?
            pi_ab_list.append(_z)
            if not ((hasattr(self.args, "mackrl_always_delegate") and self.args.mackrl_always_delegate) or \
                    (hasattr(self.args, "mackrl_delegate_if_zero_ck") and self.args.mackrl_delegate_if_zero_ck)):
                # Calculate corrective delegation (when unavailable actions are selected)
                aa_a, _, _ = _to_batch(avail_actions_level3[_a:_a+1].clone(), tformat_level3)
                aa_b, _, _ = _to_batch(avail_actions_level3[_b:_b+1].clone(), tformat_level3)
                paa, params_paa, tformat_paa = _to_batch(pairwise_avail_actions[_i:_i+1].clone(), tformat_level3)
                _pi_a, _, _ = _to_batch(pi_a, tformat_level3)
                _pi_b, _, _ = _to_batch(pi_b, tformat_level3)
                # x = th.bmm(th.unsqueeze(aa_a, 2),  th.unsqueeze(aa_b, 1))
                # At least one action unavailable.
                diff_matrix =  th.relu(paa[:, 1:-1].view(-1, self.n_actions, self.n_actions) -
                                       th.bmm(th.unsqueeze(aa_a[:, :-1], 2),  th.unsqueeze(aa_b[:, :-1], 1)))

                diff_matrix = diff_matrix * _p_ab.view(-1, self.n_actions, self.n_actions)

                both_unavailable = th.relu(paa[:, 1:-1].view(-1, self.n_actions, self.n_actions) -
                                      th.add(th.unsqueeze(aa_a[:, :-1], 2), th.unsqueeze(aa_b[:, :-1], 1)))

                both_unavailable = both_unavailable * _p_ab.view(-1, self.n_actions, self.n_actions)
                both_unavailable_weight = th.sum(both_unavailable.view(-1, self.n_actions*self.n_actions), -1, keepdim=True)

                # If neither component of the joint action is available both get resampled, with the probability of the independent actors.
                correction = both_unavailable_weight * pi_actions_selected[_a:_a + 1].view(-1,1) * pi_actions_selected[_b:_b + 1].view(-1,1)

                act_a = actions_masked[_a:_a + 1].clone().view(-1, 1,1).long()
                act_b = actions_masked[_b:_b + 1].clone().view(-1, 1,1).long()
                b_resamples = th.sum(th.gather(diff_matrix, 1, act_a.repeat([1,1, self.n_actions])),-1)
                b_resamples = b_resamples * pi_actions_selected[_b:_b + 1].view(-1,1)

                a_resamples = th.sum(th.gather(diff_matrix, 2, act_b.repeat([1, self.n_actions,1])), 1)
                a_resamples = a_resamples * pi_actions_selected[_a:_a + 1].view(-1,1)

                correction = correction + b_resamples + a_resamples
                pi_corr_list.append(_from_batch(correction, params_paa, tformat_paa))

        pi_a_cross_pi_b = th.cat(pi_a_cross_pi_b_list, dim=0)
        pi_ab_selected = th.cat(pi_ab_list, dim=0)

        p_a_b = p_d * pi_a_cross_pi_b + pi_ab_selected
        agent_pairs = self.ordered_agent_pairings
        if not ( (hasattr(self.args, "mackrl_always_delegate") and self.args.mackrl_always_delegate) or \
               (hasattr(self.args, "mackrl_delegate_if_zero_ck") and self.args.mackrl_delegate_if_zero_ck) ):
            pi_corr = th.cat(pi_corr_list, dim=0)
            p_a_b += pi_corr # DEBUG

        pi_c_prod_list = []
        p_a_b_pair_list = []

        for pairs in self.ordered_2_agent_pairings:
            p_a_b_pair_list.append(reduce(operator.mul, [p_a_b[pair] for pair in pairs]))

            agents_in_pairs = list(itertools.chain.from_iterable([agent_pairs[pair] for pair in pairs]))

            # calculate pi_c_prod
            _pi_actions_selected = pi_actions_selected.clone()
            _pi_actions_selected[pi_actions_selected_mask] = 0.0
            for _agent in agents_in_pairs:
                _pi_actions_selected[_agent:_agent + 1] = 1.0
            _k = th.prod(_pi_actions_selected, dim=_adim(tformat_level3), keepdim=True)
            pi_c_prod_list.append(_k)

        pi_ab_pair = th.stack(p_a_b_pair_list, dim=0)
        pi_c_prod = th.cat(pi_c_prod_list, dim=0)

        # next, calculate p_a_b * prod(p, -a-b)
        p_prod = pi_ab_pair * pi_c_prod

        # now, calculate p_a_b_c
        _tmp =  out_level1.transpose(_adim(tformat_level1), _vdim(tformat_level1))
        _tmp[_tmp!=_tmp] = 0.0
        p_a_b_c = (p_prod * _tmp).sum(dim=_adim(tformat_level1), keepdim=True)

        if self.args.debug_mode in ["check_probs"]:
            if not hasattr(self, "action_table"):
                self.action_table = {}
            if not hasattr(self, "actions_sampled"):
                self.actions_sampled = 0
            actions_flat = actions.view(self.n_agents, -1)
            for id in range(actions_flat.shape[1]):
                act = tuple(actions_flat[:, id].tolist())
                if act in self.action_table:
                    self.action_table[act] += 1
                else:
                    self.action_table[act] = 0
            self.actions_sampled += actions_flat.shape[0]

        if self.args.debug_mode in ["check_probs"]:
            actions_flat = actions.view(self.n_agents, -1)
            for id in range(actions_flat.shape[1]):
                print("sampled: ",
                      self.action_table[tuple(actions_flat[:, id].tolist())] / self.actions_sampled,
                      " pred: ",
                      p_a_b_c.view(-1)[id])

        # DEBUG MODE HERE!
        # _check_nan(pi_c_prod)
        # _check_nan(p_d)
        # _check_nan(pi_ab_selected)
        # _check_nan(pi_a_cross_pi_b)
        # _check_nan(p_a_b)
        # _check_nan(p_a_b_c)

        # agent_parameters = list(self.parameters())
        # pi_c_prod.sum().backward(retain_graph=True) # p_prod throws NaN
        # _check_nan(agent_parameters)
        # p_a_b.sum().backward(retain_graph=True)
        # _check_nan(agent_parameters)
        # p_prod.sum().backward(retain_graph=True)
        # _check_nan(agent_parameters)

        hidden_states = {"level1": hidden_states_level1,
                         "level2": hidden_states_level2,
                         "level3": hidden_states_level3
                        }

        loss = loss_fn(policies=p_a_b_c, tformat=tformat_level3)

        # loss = p_a_b_c.sum(), "a*bs*t*v"
        #loss[0].sum().backward(retain_graph=True)
        # loss[0].backward(retain_graph=True)

        # try:
        #     _check_nan(agent_parameters)
        # except Exception as e:
        #     for a, b in self.named_parameters():
        #         print("{}:{}".format(a, b.grad))
        #     a = 5
        #     pass

        return p_a_b_c, hidden_states, loss, tformat_level3 # note: policies can have NaNs in it!!
