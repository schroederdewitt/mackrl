from itertools import combinations
import numpy as np
from torch.autograd import Variable
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from components.transforms import _check_inputs_validity, _to_batch, _from_batch, _adim, _bsdim, _tdim, _vdim, _check_nan
from models.basic import RNN as RecurrentAgent, DQN as NonRecurrentAgent
from utils.mackrel import _n_agent_pairings, _agent_ids_2_pairing_id, _ordered_agent_pairings, _action_pair_2_joint_actions

class CentralVFunction(nn.Module):

    def __init__(self, input_shapes, n_agents, n_actions, output_shapes={}, layer_args={}, args=None):

        super(CentralVFunction, self).__init__()

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

        if getattr(self.args, "critic_is_recurrent", False):
            self.layer_args["fc2"]["in"] = 64
            self.gru = nn.GRUCell(self.layer_args["fc1"]["out"], 64)

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
        if getattr(self.args, "critic_is_recurrent", False):
            return
        pass

    def forward(self, inputs, tformat, **kwargs):
        # _check_inputs_validity(inputs, self.input_shapes, tformat, allow_nonseq=True)

        if getattr(self.args, "critic_is_recurrent", False):
            _inputs = inputs.get("main")

            t_dim = _tdim(tformat)
            assert t_dim == 1, "t_dim along unsupported axis"
            t_len = _inputs.shape[t_dim]

            try:
                hidden_states = kwargs["hidden_states"]
            except:
                pass

            x_list = []
            h_list = [hidden_states]
            for t in range(t_len):
                x = _inputs[:, slice(t, t + 1), :].contiguous()

                x, params_x, tformat_x = _to_batch(x, tformat)
                h, params_h, tformat_h = _to_batch(h_list[-1], tformat)

                x = F.relu(self.fc1(x))
                h = self.gru(x, h)
                x = self.fc2(x)

                h = _from_batch(h, params_h, tformat_h)
                x = _from_batch(x, params_x, tformat_x)

                h_list.append(h)
                x_list.append(x)

            return th.cat(x_list, t_dim), \
                   tformat
        else:
            main, params, m_tformat = _to_batch(inputs.get("main"), tformat)
            x = F.relu(self.fc1(main))
            vvalue = self.fc2(x)

        return _from_batch(vvalue, params, m_tformat), m_tformat

class CentralVCritic(nn.Module):

    """
    Concats MACKRLQFunction and MACKRLAdvantage together to an advantage and qvalue function
    """

    def __init__(self, input_shapes, n_actions, n_agents, output_shapes={}, layer_args={}, args=None):
        """
        This model contains no network layers but only sub-models
        """

        super(CentralVCritic, self).__init__()
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

        self.CentralVFunction= CentralVFunction(input_shapes={"main":self.input_shapes["vfunction"]},
                                                   output_shapes={},
                                                   layer_args={"main":self.layer_args["vfunction"]},
                                                   n_agents = self.n_agents,
                                                   n_actions = self.n_actions,
                                                   args=self.args)

        pass

    def init_hidden(self):
        """
        There's no hidden state required for this model.
        """
        pass


    def forward(self, inputs, tformat, baseline=True, **kwargs):

        vvalue, vvalue_tformat = self.CentralVFunction(inputs={"main":inputs["vfunction"]},
                                                       tformat=tformat,
                                                       **kwargs)

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
