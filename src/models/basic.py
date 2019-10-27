from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch as th

from components.transforms import _to_batch, _from_batch, _check_inputs_validity, _tdim, _vdim

class DQN(nn.Module):
    def __init__(self, input_shapes, n_actions, output_type=None, output_shapes=None, layer_args=None, args=None):
        super(DQN, self).__init__()
        self.args = args
        self.n_actions = n_actions

        assert output_type is not None, "you have to set an output_type!"
        self.output_type = output_type

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}

        self.input_shapes.update(input_shapes)

        # Set up output_shapes automatically if required
        self.output_shapes = {}
        self.output_shapes["fc2"] = self.n_actions  # output
        if output_shapes is not None:
            self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["fc1"] = {"in": self.input_shapes["main"], "out":64}
        self.layer_args["fc2"] = {"in": self.layer_args["fc1"]["out"], "out": self.output_shapes["fc2"]}
        if layer_args is not None:
            self.layer_args.update(layer_args)

        # Set up network layers
        self.fc1 = nn.Linear(self.layer_args["fc1"]["in"], self.layer_args["fc1"]["out"])
        self.fc2 = nn.Linear(self.layer_args["fc2"]["in"], self.layer_args["fc2"]["out"])

    def init_hidden(self, batch_size, *args, **kwargs):
        """
        model has no hidden state, but we will pretend otherwise for consistency
        """
        vbl = Variable(th.zeros(batch_size, 1, 1))
        tformat = "bs*t*v"
        return vbl.cuda() if self.args.use_cuda else vbl, tformat

    def forward(self, inputs, tformat, loss_fn=None, hidden_states=None, **kwargs):
        _check_inputs_validity(inputs, self.input_shapes, tformat)

        # Execute model branch "main"
        x, params, tformat = _to_batch(inputs["main"], tformat)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = _from_batch(x, params, tformat)

        losses = None
        if self.output_type in ["policies"]:
            log_softmax = kwargs.get("log_softmax", False)
            if log_softmax:
                x = F.log_softmax(x, dim=_vdim(tformat))
            else:
                x = F.softmax(x, dim=_vdim(tformat))
        if loss_fn is not None:
            losses, _ = loss_fn(x, tformat=tformat)

        return x, hidden_states, losses, tformat  # output, hidden state, losses



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

class RNN(nn.Module):

    def __init__(self, input_shapes, n_actions, output_type=None, output_shapes={}, layer_args={}, args=None, **kwargs):
        super(RNN, self).__init__()
        self.args = args
        self.n_actions = n_actions
        assert output_type is not None, "you have to set an output_type!"
        # self.output_type=output_type

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        self.input_shapes.update(input_shapes)

        # Set up layer_args automatically if required
        self.output_shapes = {}
        self.output_shapes["output"] = self.n_actions # output
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

    def init_hidden(self, batch_size=1):
        vbl = Variable(th.zeros(batch_size, 1, self.layer_args["gru"]["hidden"]))
        tformat = "bs*t*v"
        return vbl.cuda() if self.args.use_cuda else vbl, tformat

    def forward(self, inputs, hidden_states, tformat, loss_fn=None, **kwargs):
        """
        If data contains whole sequences, can pass loss_fn to forward pass in order to generate all losses
        automatically.
        Can either be operated in sequence mode, or operated step-by-step
        """
        _check_inputs_validity(inputs, self.input_shapes, tformat)
        _inputs = inputs["main"]

        loss = None
        t_dim = _tdim(tformat)
        assert t_dim == 2, "t_dim along unsupported axis"
        t_len = _inputs.shape[t_dim]

        loss_x = []
        output_x = []
        h_list = [hidden_states]

        for t in range(t_len):

            x = _inputs[:, :, slice(t, t + 1), :].contiguous()
            x, tformat = self.encoder({"main":x}, tformat)
            x, params_x, tformat_x = _to_batch(x, tformat)
            h, params_h, tformat_h = _to_batch(h_list[-1], tformat)

            h = self.gru(x, h)
            x = self.output(h)

            h = _from_batch(h, params_h, tformat_h)
            x = _from_batch(x, params_x, tformat_x)
            h_list.append(h)
            loss_x.append(x)

            # we will not branch the variables if loss_fn is set - instead return only tensor values for x in that case
            output_x.append(x) if loss_fn is None else output_x.append(x.clone())

        if loss_fn is not None:
            _x = th.cat(loss_x, dim=_tdim(tformat))
            loss = loss_fn(_x, tformat=tformat)[0]

        return th.cat(output_x, t_dim), \
           th.cat(h_list[1:], t_dim), \
           loss, \
           tformat

class FCEncoder(nn.Module):
    def __init__(self, input_shapes, output_shapes=None, layer_args=None, args=None):
        super(FCEncoder, self).__init__()
        self.args = args

        # Set up input regions automatically if required (if sensible)
        self.input_shapes = {}
        assert set(input_shapes.keys()) == {"main"}, \
            "set of input_shapes does not coincide with model structure!"
        self.input_shapes.update(input_shapes)

        # Set up layer_args automatically if required
        self.output_shapes = {}
        self.output_shapes["fc1"] = 64
        if output_shapes is not None:
            self.output_shapes.update(output_shapes)

        # Set up layer_args automatically if required
        self.layer_args = {}
        self.layer_args["fc1"] = {"in":input_shapes["main"], "out":output_shapes["main"]}
        if layer_args is not None:
            self.layer_args.update(layer_args)

        #Set up network layers
        self.fc1 = nn.Linear(self.input_shapes["main"], self.output_shapes["main"])
        pass

    def forward(self, inputs, tformat):

        x, n_seq, tformat = _to_batch(inputs["main"], tformat)
        x = F.relu(self.fc1(x))
        return _from_batch(x, n_seq, tformat), tformat
