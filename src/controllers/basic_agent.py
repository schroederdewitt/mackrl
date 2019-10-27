import torch as th

from models import REGISTRY as mo_REGISTRY
from components.scheme import Scheme
from components.transforms import _generate_input_shapes, _generate_scheme_shapes

class BasicAgentController():

    def __init__(self,
                 n_agents,
                 n_actions,
                 args,
                 agent_id=None,
                 model=None,
                 output_type="policies",
                 scheme_fn=None,
                 input_columns=None):
        self.args = args
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.output_type = output_type

        # Set up schemes
        if scheme_fn is None:
            self.scheme_fn = lambda _agent_id: Scheme([dict(name="agent_id",
                                                            transforms=[("one_hot",dict(range=(0, self.n_agents-1)))],
                                                            select_agent_ids=[_agent_id],),
                                                       dict(name="observations",
                                                            rename="agent_observation",
                                                            select_agent_ids=[_agent_id],
                                                            switch=not self.args.use_full_observability),
                                                       dict(name="actions",
                                                            rename="past_action",
                                                            select_agent_ids=[_agent_id],
                                                            transforms=[("shift", dict(steps=1)),
                                                                        ("one_hot", dict(range=(0, self.n_actions-1)))], # DEBUG!
                                                            switch=self.args.obs_last_action),
                                                       dict(name="state",
                                                            switch=self.args.use_full_observability)
                                                      ]).agent_flatten()
        else:
            self.scheme_fn = scheme_fn

        self.scheme = {}
        self.scheme["main"] = self.scheme_fn(self.agent_id)

        if input_columns is None:
            # construct model-specific input regions
            self.input_columns = {}
            self.input_columns["main"] = {}
            self.input_columns["main"]["main"] = Scheme([dict(name="agent_id", select_agent_ids=[self.agent_id]),
                                                         dict(name="agent_observation",
                                                              select_agent_ids=[self.agent_id],
                                                              switch=not self.args.use_full_observability),
                                                         dict(name="state", switch=self.args.use_full_observability),
                                                         dict(name="past_action",
                                                              select_agent_ids=[self.agent_id],
                                                              switch=self.args.obs_last_action)]).agent_flatten()
        else:
            self.input_columns = input_columns


        if model is not None:
            self.model = model
        else:
            self.model = mo_REGISTRY[args.agent_model]

        self.args = args
        self.use_coma = True if getattr(args, "learner", "") == "coma" else False

    def get_parameters(self):
        return self.model.parameters()

    def create_model(self, transition_scheme):
        # obtain shapes of all schemes

        self.scheme_shapes = _generate_scheme_shapes(transition_scheme=transition_scheme,
                                                     dict_of_schemes=self.scheme)

        self.input_shapes = _generate_input_shapes(input_columns=self.input_columns,
                                                   scheme_shapes=self.scheme_shapes)

        self.model = self.model(input_shapes=self.input_shapes["main"],
                                n_actions=self.n_actions,
                                output_type=self.output_type,
                                args=self.args)
        if self.args.use_cuda:
            self.model = self.model.cuda()
        return

    def generate_initial_hidden_states(self, batch_size=1, **kwargs):
        return self.model.init_hidden(batch_size)

    def get_outputs(self, inputs, hidden_state, loss_fn=None, mode=None, softmax=False, log_softmax=False, to_cpu=False, to_numpy=False):
        """
        While agents supply a set of schemes, they do not necessarily apply them.
        This adds flexibility to the user, who can perform arbitrary modifications in this way.
        """

        assert isinstance(inputs, dict), "inputs needs to be dict"
        assert all([isinstance(_v.data, (th.FloatTensor, th.cuda.FloatTensor)) for _v in inputs.values()]), \
            "inputs elements need to be of type th.FloatTensor"

        # construct the model inputs
        model_inputs = {}
        for _k, _v in self.input_columns.items():
            model_inputs[_k] = th.cat([inputs[_c] for _c in _v], 2)

        if self.use_coma:
            if not log_softmax:
                softmax = True
            pi, hidden_state, losses = self.model(model_inputs, hidden_state=hidden_state, loss_fn=loss_fn, softmax=softmax, log_softmax=log_softmax)
            if to_cpu or to_numpy:
                pi = pi.cpu()
                hidden_state = hidden_state if hidden_state is None else hidden_state.cpu()
                losses = losses if losses is None else losses.cpu()
            if to_numpy:
                pi = pi.data.numpy()
                hidden_state = hidden_state if hidden_state is None else hidden_state.data.numpy()
                losses = losses if losses is None else losses.data.numpy()
            return {"policies": pi, "hidden_state": hidden_state, "losses": losses} # return policy instead
        else:
            q, hidden_state, losses = self.model(model_inputs, hidden_state=hidden_state, loss_fn=loss_fn, softmax=softmax, log_softmax=log_softmax)
            if to_cpu or to_numpy:
                q = q.cpu()
                hidden_state = hidden_state if hidden_state is None else hidden_state.cpu()
                losses = losses if losses is None else losses.cpu()
            if to_numpy:
                q = q.data.numpy()
                hidden_state = hidden_state if hidden_state is None else hidden_state.data.numpy()
                losses = losses if losses is None else losses.data.numpy()
            return {"q_values": q, "hidden_state": hidden_state, "losses":losses}




