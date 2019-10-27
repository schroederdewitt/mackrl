from copy import deepcopy
from functools import partial
import numpy as np
from numpy.random import randint
import torch as th
from torch import nn
from torch.autograd import Variable
from torch.optim import RMSprop

from debug.debug import IS_PYCHARM_DEBUG
from components.scheme import Scheme
from components.transforms import _adim, _bsdim, _tdim, _vdim, \
    _generate_input_shapes, _generate_scheme_shapes, _build_model_inputs, \
    _join_dicts, _seq_mean, _copy_remove_keys, _make_logging_str, _underscore_to_cap, \
    _n_step_return
from components.losses import EntropyRegularisationLoss
from components.transforms import _to_batch, _from_batch, _naninfmean, _pad_zero, _pad, _check_nan
from models.centralV import CentralVCritic

from .basic import BasicLearner

class CentralVPolicyLoss(nn.Module):

    def __init__(self):
        super(CentralVPolicyLoss, self).__init__()

    def forward(self, policies, advantages, actions, seq_lens, tformat, *args, **kwargs):

        assert tformat in ["a*bs*t*v"], "invalid input format!"

        policies = policies.clone()
        advantages = advantages.clone().detach().unsqueeze(0).repeat(policies.shape[0],1,1,1)
        actions = actions.clone()

        # last elements of advantages are NaNs
        mask = advantages.clone().fill_(1.0).byte()
        _pad_zero(mask, tformat, seq_lens)
        mask[:, :, :-1, :] = mask[:, :, 1:, :] # account for terminal NaNs of targets
        mask[:, :, -1, :] = 0.0  # handles case of seq_len=limit_len
        _pad_zero(policies, tformat, seq_lens)
        _pad_zero(actions, tformat, seq_lens)
        advantages[~mask] = 0.0

        pi_taken = th.gather(policies, _vdim(tformat), actions.long())
        pi_taken_mask = (pi_taken < 10e-40)
        log_pi_taken = th.log(pi_taken)
        log_pi_taken[pi_taken_mask] = 0.0

        loss = - log_pi_taken * advantages * mask.float()
        norm = mask.sum(_bsdim(tformat), keepdim=True)
        norm[norm == 0.0] = 1.0
        loss_mean = loss.sum(_bsdim(tformat), keepdim=True) / norm.float()
        loss_mean = loss_mean.squeeze(_vdim(tformat)).squeeze(_bsdim(tformat))

        output_tformat = "a*t"
        return loss_mean, output_tformat

class CentralVCriticLoss(nn.Module):

    def __init__(self):
        super(CentralVCriticLoss, self).__init__()

    def forward(self, inputs, target, mask, tformat):
        assert tformat in ["bs*t*v"], "invalid input format!"

        # calculate mean-square loss
        ret = ((inputs[mask] - target.detach()[mask])**2).mean()

        output_tformat = "s" # scalar
        return ret, output_tformat

class CentralVLearner(BasicLearner):

    def __init__(self, multiagent_controller, logging_struct=None, args=None):
        self.args = args
        self.multiagent_controller = multiagent_controller
        self.n_agents = multiagent_controller.n_agents
        self.n_actions = self.multiagent_controller.n_actions
        for _i in range(1, 4):
            setattr(self, "T_policy_level{}".format(_i), 0)
            setattr(self, "T_critic_level{}".format(_i), 0)

        self.stats = {}
        self.logging_struct = logging_struct

        self.critic_class = CentralVCritic

        self.critic_scheme = Scheme([dict(name="actions_onehot",
                                          rename="past_actions",
                                          select_agent_ids=list(range(self.n_agents)),
                                          transforms=[("shift", dict(steps=1)),
                                                     ],
                                          switch=self.args.critic_use_past_actions),
                                     dict(name="state")
                                   ])
        self.target_critic_scheme = self.critic_scheme

        # Set up schemes
        self.scheme = {}
        # level 1

        self.scheme["critic"] = self.critic_scheme
        self.scheme["target_critic"] = self.critic_scheme

        # create joint scheme from the critic scheme
        self.joint_scheme_dict = _join_dicts(self.scheme,
                                             self.multiagent_controller.joint_scheme_dict)

        # construct model-specific input regions
        self.input_columns = {}
        self.input_columns["critic"] = {"vfunction":Scheme([{"name":"state"},
                                                           ])}
        self.input_columns["target_critic"] = self.input_columns["critic"]

        # for _i in range(self.n_agents):
        #     self.input_columns["critic__agent{}".format(_i)] = {"vfunction":Scheme([{"name":"state"},
        #                                                                             #{"name":"past_actions",
        #                                                                             # "select_agent_ids":list(range(self.n_agents))},
        #                                                                             #{"name": "actions",
        #                                                                             # "select_agent_ids": list(
        #                                                                             #     range(self.n_agents))}
        #                                                                             ])}
        #
        # for _i in range(self.n_agents):
        #     self.input_columns["target_critic__agent{}".format(_i)] = self.input_columns["critic__agent{}".format(_i)]


        self.last_target_update_T_critic = 0
        self.T_critic = 0
        self.T_policy = 0

        self.policy_loss_class = CentralVPolicyLoss
        pass


    def create_models(self, transition_scheme):

        self.scheme_shapes = _generate_scheme_shapes(transition_scheme=transition_scheme,
                                                     dict_of_schemes=self.scheme)

        self.input_shapes = _generate_input_shapes(input_columns=self.input_columns,
                                                   scheme_shapes=self.scheme_shapes)

        # set up critic model
        self.critic_model = self.critic_class(input_shapes=self.input_shapes["critic"],
                                              n_agents=self.n_agents,
                                              n_actions=self.n_actions,
                                              args=self.args)
        if self.args.use_cuda:
            self.critic_model = self.critic_model.cuda()
        self.target_critic_model = deepcopy(self.critic_model)


        # set up optimizers
        if self.args.share_agent_params:
            self.agent_parameters = self.multiagent_controller.get_parameters()
        else:
            assert False, "TODO"
        self.agent_optimiser = RMSprop(self.agent_parameters, lr=self.args.lr_agent)

        self.critic_parameters = []
        if not (hasattr(self.args, "critic_share_params") and not self.args.critic_share_params):
            self.critic_parameters.extend(self.critic_model.parameters())
        else:
            assert False, "TODO"
        self.critic_optimiser = RMSprop(self.critic_parameters, lr=self.args.lr_critic)

        # this is used for joint retrieval of data from all schemes
        self.joint_scheme_dict = _join_dicts(self.scheme, self.multiagent_controller.joint_scheme_dict)

        self.args_sanity_check() # conduct MACKRL sanity check on arg parameters
        pass

    def args_sanity_check(self):
        """
        :return:
        """
        pass

    def train(self,
              batch_history,
              T_env=None):


        # Update target if necessary
        if (self.T_critic - self.last_target_update_T_critic) / self.args.target_critic_update_interval > 1.0:
            self.update_target_nets()
            self.last_target_update_T_critic = self.T_critic
            print("updating target net!")

        # Retrieve and view all data that can be retrieved from batch_history in a single step (caching efficient)

        # create one single batch_history view suitable for all
        data_inputs, data_inputs_tformat = batch_history.view(dict_of_schemes=self.joint_scheme_dict,
                                                              to_cuda=self.args.use_cuda,
                                                              to_variable=True,
                                                              bs_ids=None,
                                                              fill_zero=True) # DEBUG: Should be True

        actions, actions_tformat = batch_history.get_col(bs=None,
                                                         col="actions",
                                                         agent_ids=list(range(0, self.n_agents)),
                                                         stack=True)

        # do single forward pass in critic
        coma_model_inputs, coma_model_inputs_tformat = _build_model_inputs(column_dict=self.input_columns,
                                                                           inputs=data_inputs,
                                                                           inputs_tformat=data_inputs_tformat,
                                                                           to_variable=True)

        critic_loss_arr = []
        critic_mean_arr = []
        target_critic_mean_arr = []
        critic_grad_norm_arr = []



        # construct target-critic targets and carry out necessary forward passes
        # same input scheme for both target critic and critic!
        inputs_target_critic = coma_model_inputs["target_critic"]
        hidden_states = None
        if getattr(self.args, "critic_is_recurrent", False):
            hidden_states = Variable(
                th.zeros(inputs_target_critic["vfunction"].shape[0], 1, self.args.agents_hidden_state_size))
            if self.args.use_cuda:
                hidden_states = hidden_states.cuda()
        output_target_critic, output_target_critic_tformat = self.target_critic_model.forward(inputs_target_critic,
                                                                                              tformat="bs*t*v",
                                                                                              hidden_states=hidden_states)



        target_critic_td_targets, \
        target_critic_td_targets_tformat = batch_history.get_stat("td_lambda_targets",
                                                                  bs_ids=None,
                                                                  td_lambda=self.args.td_lambda,
                                                                  gamma=self.args.gamma,
                                                                  value_function_values=output_target_critic[
                                                                      "vvalue"].unsqueeze(0).detach(),
                                                                  to_variable=True,
                                                                  n_agents=1,
                                                                  to_cuda=self.args.use_cuda)

        # targets for terminal state are always NaNs, so mask these out of loss as well!
        mask = _pad_zero(inputs_target_critic["vfunction"][:,:,-1:].clone().fill_(1.0),
                         "bs*t*v",
                         batch_history.seq_lens).byte()
        mask[:, :-1, :] = mask[:, 1:, :] # account for terminal NaNs of targets
        mask[:, -1, :] = 0.0  # handles case of seq_len=limit_len

        output_critic_list = []
        def _optimize_critic(**kwargs):
            inputs_critic= kwargs["coma_model_inputs"]["critic"]
            inputs_target_critic=kwargs["coma_model_inputs"]["target_critic"]
            inputs_critic_tformat=kwargs["tformat"]
            inputs_target_critic_tformat = kwargs["tformat"]
            t = kwargs["t"]
            do_train = kwargs["do_train"]
            _inputs_critic = inputs_critic
            vtargets = target_critic_td_targets.squeeze(0)

            hidden_states = None
            if getattr(self.args, "critic_is_recurrent", False):
                hidden_states = Variable(th.zeros(output_target_critic["vvalue"].shape[0], 1, self.args.agents_hidden_state_size))
                if self.args.use_cuda:
                    hidden_states = hidden_states.cuda()

            output_critic, output_critic_tformat = self.critic_model.forward({_k:_v[:, t:t+1] for _k, _v in _inputs_critic.items()},
                                                                             tformat="bs*t*v",
                                                                             hidden_states=hidden_states)
            output_critic_list.append({_k:_v.clone() for _k, _v in output_critic.items()})

            if not do_train:
                return output_critic
            critic_loss, \
            critic_loss_tformat = CentralVCriticLoss()(inputs=output_critic["vvalue"],
                                                       target=Variable(vtargets[:, t:t+1], requires_grad=False),
                                                       tformat="bs*t*v",
                                                       mask=mask[:, t:t+1])
                                                       # seq_lens=batch_history.seq_lens)

            # optimize critic loss
            self.critic_optimiser.zero_grad()
            critic_loss.backward()

            critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_parameters,
                                                           10)
            self.critic_optimiser.step()

            # Calculate critic statistics and update
            target_critic_mean = _naninfmean(output_target_critic["vvalue"])

            critic_mean = _naninfmean(output_critic["vvalue"])

            critic_loss_arr.append(np.asscalar(critic_loss.data.cpu().numpy()))
            critic_mean_arr.append(critic_mean)
            target_critic_mean_arr.append(target_critic_mean)
            critic_grad_norm_arr.append(critic_grad_norm)

            self.T_critic += 1
            return output_critic



        output_critic = None
        # optimize the critic as often as necessary to get the critic loss down reliably
        for _i in reversed(range(batch_history._n_t)): #range(self.args.n_critic_learner_reps):
            _ = _optimize_critic(coma_model_inputs=coma_model_inputs,
                                 tformat=coma_model_inputs_tformat,
                                 actions=actions,
                                 t=_i,
                                 do_train=(_i < max(batch_history.seq_lens) - 1))


        hidden_states = None
        if getattr(self.args, "critic_is_recurrent", False):
            hidden_states = Variable(th.zeros(coma_model_inputs["critic"]["vfunction"].shape[0], 1, self.args.agents_hidden_state_size))
            if self.args.use_cuda:
                hidden_states = hidden_states.cuda()

        # get advantages
        # output_critic, output_critic_tformat = self.critic_model.forward(coma_model_inputs["critic"],
        #                                                                   tformat="bs*t*v",
        #                                                                   hidden_states=hidden_states)

        values = th.cat([ x["vvalue"] for x in reversed(output_critic_list)], dim=1)

        # advantages = output_critic["advantage"]
        advantages = _n_step_return(values=values.unsqueeze(0), #output_critic["vvalue"].unsqueeze(0),
                                    rewards=batch_history["reward"][0],
                                    terminated=batch_history["terminated"][0],
                                    truncated=batch_history["truncated"][0],
                                    seq_lens=batch_history.seq_lens,
                                    horizon=batch_history._n_t-1,
                                    n=1 if not hasattr(self.args, "n_step_return_n") else self.args.n_step_return_n,
                                    gamma=self.args.gamma) - values #output_critic["vvalue"]

        advantages = advantages.squeeze(0)

        # only train the policy once in order to stay on-policy!
        policy_loss_function = partial(self.policy_loss_class(),
                                       actions = actions,
                                       advantages=advantages.detach(),
                                       seq_lens=batch_history.seq_lens,
                                       n_agents=self.n_agents)

        hidden_states, hidden_states_tformat = self.multiagent_controller.generate_initial_hidden_states(
            len(batch_history), caller="learner")

        agent_controller_output, \
        agent_controller_output_tformat = self.multiagent_controller.get_outputs(data_inputs,
                                                                                 hidden_states=hidden_states,
                                                                                 loss_fn=policy_loss_function,
                                                                                 tformat=data_inputs_tformat,
                                                                                 test_mode=False,
                                                                                 actions=actions)
        CentralV_loss = agent_controller_output["losses"]
        CentralV_loss = CentralV_loss.mean()

        if hasattr(self.args, "coma_use_entropy_regularizer") and self.args.coma_use_entropy_regularizer:
            CentralV_loss += self.args.coma_entropy_loss_regularization_factor * \
                         EntropyRegularisationLoss()(policies=agent_controller_output["policies"],
                                                     tformat="a*bs*t*v").sum()

        # carry out optimization for agents
        self.agent_optimiser.zero_grad()
        CentralV_loss.backward()

        policy_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_parameters, 10)
        try:
            _check_nan(self.agent_parameters, silent_fail=False)
            self.agent_optimiser.step()  # DEBUG
            self._add_stat("Agent NaN gradient", 0.0, T_env=T_env)
        except Exception as e:
            self.logging_struct.py_logger.warning("NaN in agent gradients! Gradient not taken. ERROR: {}".format(e))
            self._add_stat("Agent NaN gradient", 1.0, T_env=T_env)

        # increase episode counter (the fastest one is always)
        self.T_policy += len(batch_history) * batch_history._n_t

        # Calculate policy statistics
        advantage_mean = _naninfmean(advantages)
        self._add_stat("advantage_mean", advantage_mean, T_env=T_env)
        self._add_stat("policy_grad_norm", policy_grad_norm, T_env=T_env)
        self._add_stat("policy_loss", CentralV_loss.data.cpu().numpy(), T_env=T_env)
        self._add_stat("critic_loss", np.mean(critic_loss_arr), T_env=T_env)
        self._add_stat("critic_mean", np.mean(critic_mean_arr), T_env=T_env)
        self._add_stat("target_critic_mean", np.mean(target_critic_mean_arr), T_env=T_env)
        self._add_stat("critic_grad_norm", np.mean(critic_grad_norm_arr), T_env=T_env)
        self._add_stat("T_policy", self.T_policy, T_env=T_env)
        self._add_stat("T_critic", self.T_critic, T_env=T_env)

        pass

    def update_target_nets(self):
        self.target_critic_model.load_state_dict(self.critic_model.state_dict())

    def get_stats(self):
        if hasattr(self, "_stats"):
            return self._stats
        else:
            return []

    def log(self, log_directly = True):
        """
        Each learner has it's own logging routine, which logs directly to the python-wide logger if log_directly==True,
        and returns a logging string otherwise

        Logging is triggered in run.py
        """
        stats = self.get_stats()
        logging_dict =  dict(advantage_mean = _seq_mean(stats["advantage_mean"]),
                             critic_grad_norm = _seq_mean(stats["critic_grad_norm"]),
                             critic_loss =_seq_mean(stats["critic_loss"]),
                             policy_grad_norm = _seq_mean(stats["policy_grad_norm"]),
                             policy_loss = _seq_mean(stats["policy_loss"]),
                             target_critic_mean = _seq_mean(stats["target_critic_mean"]),
                             T_critic=self.T_critic,
                             T_policy=self.T_policy
                            )
        logging_str = "T_policy={:g}, T_critic={:g}, ".format(logging_dict["T_policy"], logging_dict["T_critic"])
        logging_str += _make_logging_str(_copy_remove_keys(logging_dict, ["T_policy", "T_critic"]))

        if log_directly:
            self.logging_struct.py_logger.info("{} LEARNER INFO: {}".format(self.args.learner.upper(), logging_str))

        return logging_str, logging_dict

    def save_models(self, path, token, T):
        import os
        if not os.path.exists("results/models/{}".format(self.args.learner)):
            os.makedirs("results/models/{}".format(self.args.learner))

        self.multiagent_controller.save_models(path=path, token=token, T=T)
        th.save(self.critic_model.state_dict(),"results/models/{}/{}_critic__{}_T.weights".format(self.args.learner,
                                                                                            token,
                                                                                            T))
        th.save(self.target_critic_model.state_dict(), "results/models/{}/{}_target_critic__{}_T.weights".format(self.args.learner,
                                                                                                           token,
                                                                                                           T))
        pass
