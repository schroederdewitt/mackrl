import numpy as np
from components.scheme import Scheme
from components.transforms import _seq_mean, _vdim, _check_nan
from copy import deepcopy
from itertools import combinations, chain
from scipy.stats.stats import pearsonr
import torch as th
from torch.distributions import Normal

from components.epsilon_schedules import FlatThenDecaySchedule
from runners import REGISTRY as r_REGISTRY
from utils.mackrel import _n_agent_pair_samples, _n_agent_pairings, _ordered_agent_pairings, _n_2_agent_pairings, \
            _ordered_2_agent_pairings

NStepRunner = r_REGISTRY["nstep"]

class MACKRLRunner(NStepRunner):

    def _setup_data_scheme(self, data_scheme):

        scheme_list = [dict(name="observations",
                            shape=(self.env_obs_size,),
                            select_agent_ids=range(0, self.n_agents),
                            dtype=np.float32,
                            missing=np.nan,),
                       dict(name="state",
                            shape=(self.env_state_size,),
                            dtype = np.float32,
                            missing=np.nan,
                            size=self.env_state_size),
                       *[dict(name="actions_level1__sample{}".format(_i), # stores ids of pairs that are sampled from
                            shape=(1,),
                            dtype=np.int32,
                            missing=-1,) for _i in range(_n_agent_pair_samples(self.n_agents))],
                       *[dict(name="actions_level2__sample{}".format(_i), # stores joint action for each sampled pair
                              shape=(1,), # i.e. just one number for a pair of actions!
                              dtype=np.int32,
                              missing=-1,) for _i in range(_n_agent_pair_samples(self.n_agents))],
                       dict(name="actions_level2",  # stores action for each agent that was chosen individually
                            shape=(1,),
                            select_agent_ids=range(0, _n_agent_pairings(self.n_agents)),
                            dtype=np.int32,
                            missing=-1, ),
                       dict(name="actions_level3", # stores action for each agent that was chosen individually
                            shape=(1,),
                            select_agent_ids=range(0, self.n_agents),
                            dtype=np.int32,
                            missing=-1, ),
                       dict(name="actions", # contains all agent actions - this is what env.step is based on!
                            shape=(1,),
                            select_agent_ids=range(0, self.n_agents),
                            dtype=np.int32,
                            missing=-1, ),
                       dict(name="actions_onehot",  # contains all agent actions - this is what env.step is based on!
                            shape=(self.n_actions,),
                            select_agent_ids=range(0, self.n_agents),
                            dtype=np.int32,
                            missing=-1, ),
                       dict(name="avail_actions_active",
                            shape=(self.n_actions + 1,), # include no-op
                            select_agent_ids=range(0, self.n_agents),
                            dtype=np.int32,
                            missing=-1,),
                       dict(name="avail_actions",
                            shape=(self.n_actions + 1,),  # include no-op
                            select_agent_ids=range(0, self.n_agents),
                            dtype=np.int32,
                            missing=-1, ),
                       dict(name="reward",
                            shape=(1,),
                            dtype=np.float32,
                            missing=np.nan),
                       dict(name="agent_id",
                            shape=(1,),
                            dtype=np.int32,
                            select_agent_ids=range(0, self.n_agents),
                            missing=-1),
                       dict(name="agent_id_onehot",
                            shape=(self.n_agents,),
                            dtype=np.int32,
                            select_agent_ids=range(0, self.n_agents),
                            missing=-1),
                       # dict(name="policies_",
                       #      shape=(self.n_actions+1,), # includes no-op
                       #      select_agent_ids=range(0, self.n_agents),
                       #      dtype=np.float32,
                       #      missing=np.nan),
                       dict(name="policies_level1",
                            shape=(len(_ordered_2_agent_pairings(self.n_agents, self.args)),),
                            dtype=np.float32,
                            missing=np.nan),
                       *[dict(name="policies_level2__sample{}".format(_i),
                              shape=(2 + self.n_actions * self.n_actions,),  # includes delegation and no-op
                              dtype=np.int32,
                              missing=-1, ) for _i in range(_n_agent_pair_samples(self.n_agents))],
                       dict(name="policies_level3",
                            shape=(self.n_actions+1,),  # does include no-op
                            select_agent_ids=range(0, self.n_agents),
                            dtype=np.float32,
                            missing=np.nan),
                       dict(name="terminated",
                            shape=(1,),
                            dtype=np.bool,
                            missing=False),
                       dict(name="truncated",
                            shape=(1,),
                            dtype=np.bool,
                            missing=False),
                       dict(name="reset",
                            shape=(1,),
                            dtype=np.bool,
                            missing=False),
                       dict(name="mackrl_epsilons_central",
                            scope="episode",
                            shape=(1,),
                            dtype=np.float32,
                            missing=float("nan")),
                       dict(name="mackrl_epsilons_central_level1",
                            scope="episode",
                            shape=(1,),
                            dtype=np.float32,
                            missing=float("nan")),
                       dict(name="mackrl_epsilons_central_level2",
                            scope="episode",
                            shape=(1,),
                            dtype=np.float32,
                            missing=float("nan")),
                       dict(name="mackrl_epsilons_central_level3",
                            scope="episode",
                            shape=(1,),
                            dtype=np.float32,
                            missing=float("nan"))
                       ]



        if self.args.mackrl_use_obs_intersections:
            obs_intersect_pair_size = self.env_setup_info[0]["obs_intersect_pair_size"]
            obs_intersect_all_size = self.env_setup_info[0]["obs_intersect_all_size"]
            scheme_list.extend([dict(name="obs_intersection_all",
                                     shape=(self.env_state_size if self.args.env_args["intersection_global_view"] else obs_intersect_all_size,),
                                     dtype=np.float32,
                                     missing=np.nan,
                                ),
                                *[dict(name="obs_intersection__pair{}".format(_i),
                                       shape=(self.env_state_size if self.args.env_args["intersection_global_view"] else obs_intersect_pair_size,),
                                       dtype=np.float32,
                                       missing=np.nan,
                                 )
                                for _i in range(_n_agent_pairings(self.n_agents))],
                                *[dict(name="avail_actions__pair{}".format(_i),
                                       shape=(self.n_actions*self.n_actions + 1,), # do include no-op
                                       dtype=np.float32,
                                       missing=np.nan,
                                       )
                                  for _i in range(_n_agent_pairings(self.n_agents))]
                                ])

        if self.args.obs_noise is not None:
            obs_intersect_pair_size = self.env_setup_info[0]["obs_intersect_pair_size"]
            obs_intersect_all_size = self.env_setup_info[0]["obs_intersect_all_size"]
            scheme_list.extend([*[dict(name="obs_intersection_obsid{}_all".format(_a),
                                       shape=(self.env_state_size if self.args.env_args[
                                           "intersection_global_view"] else obs_intersect_all_size,),
                                       dtype=np.float32,
                                       missing=np.nan,
                                       ) for _a in range(self.n_agents)],
                                *list(chain.from_iterable([[dict(name="obs_intersection_obsid{}__pair{}".format(_p, _i),
                                                                 shape=(self.env_state_size if self.args.env_args[
                                                                     "intersection_global_view"] else obs_intersect_pair_size,),
                                                                 dtype=np.float32,
                                                                 missing=np.nan,
                                                                 )
                                                            for _i in range(_n_agent_pairings(self.n_agents))] for _p in
                                                           range(2)])),
                                ])

        self.data_scheme = Scheme(scheme_list)
        pass

    def _add_episode_stats(self, T_env, **kwargs):
        super()._add_episode_stats(T_env, **kwargs)

        test_suffix = "" if not self.test_mode else "_test"
        if self.args.obs_noise and self.test_mode:
            obs_noise_std = kwargs.get("obs_noise_std", 0.0)
            test_suffix = "_noise{}_test".format(obs_noise_std)

        tmp = self.episode_buffer["policies_level1"][0]
        entropy1 = np.nanmean(np.nansum((-th.log(tmp)*tmp).cpu().numpy(), axis=2))
        self._add_stat("policy_level1_entropy",
                       entropy1,
                       T_env=T_env,
                       suffix=test_suffix)

        for _i in range(_n_agent_pair_samples(self.n_agents)):
            tmp = self.episode_buffer["policies_level2__sample{}".format(_i)][0]
            entropy2 = np.nanmean(np.nansum((-th.log(tmp) * tmp).cpu().numpy(), axis=2))
            self._add_stat("policy_level2_entropy_sample{}".format(_i),
                           entropy2,
                           T_env=T_env,
                           suffix=test_suffix)

        #entropy3 = np.nanmean(np.nansum((-th.log(tmp) * tmp).cpu().numpy(), axis=2))
        self._add_stat("policy_level3_entropy",
                        self.episode_buffer.get_stat("policy_entropy", policy_label="policies_level3"),
                        T_env=T_env,
                        suffix=test_suffix)

        actions_level2 = []
        for i in range(self.n_agents // 2):
            actions_level2_tmp, _ = self.episode_buffer.get_col(col="actions_level2__sample{}".format(i))
            actions_level2.append(actions_level2_tmp)
        actions_level2 = th.cat(actions_level2, 0)
        delegation_rate = (th.sum(actions_level2==0.0).float() / (actions_level2.nelement() - th.sum(actions_level2!=actions_level2)).float()).item()
        self._add_stat("level2_delegation_rate",
                       delegation_rate,
                       T_env=T_env,
                       suffix=test_suffix)

        # common knowledge overlap between all agents
        overlap_all = th.sum((self.episode_buffer["obs_intersection_all"][0] > 0.0), dim=_vdim("bs*t*v")).float().mean().item()
        self._add_stat("obs_intersection_all_rate",
                       overlap_all,
                       T_env=T_env,
                       suffix=test_suffix)

        # common knowledge overlap between agent pairs
        overlaps = []
        for _pair_id, (id_1, id_2) in enumerate(_ordered_agent_pairings(self.n_agents)):
            overlap_pair = th.sum(self.episode_buffer["obs_intersection__pair{}".format(_pair_id)][0] > 0.0, dim=_vdim("bs*t*v")).float().mean().item()
            overlaps.append(overlap_pair)
            self._add_stat("obs_intersection_pair{}".format(_pair_id),
                           overlap_pair,
                           T_env=T_env,
                           suffix=test_suffix)

        self._add_stat("obs_intersection_pairs_all".format(_pair_id),
                       sum(overlaps) / float(len(overlaps)),
                       T_env=T_env,
                       suffix=test_suffix)

        # TODO: Policy entropy across levels! (Use suffix)
        return

    def reset(self, test_mode=False, obs_noise_std=None):
        super().reset(test_mode = test_mode, obs_noise_std= obs_noise_std)

        # if no test_mode, calculate fresh set of epsilons/epsilon seeds and update epsilon variance
        if not self.test_mode:
            ttype = th.cuda.FloatTensor if self.episode_buffer.is_cuda else th.FloatTensor
            # calculate MACKRL_epsilon_schedules
            if not hasattr(self, "mackrl_epsilon_decay_schedule_level1"):
                 self.mackrl_epsilon_decay_schedule_level1 = FlatThenDecaySchedule(start=self.args.mackrl_epsilon_start_level1,
                                                                                finish=self.args.mackrl_epsilon_finish_level1,
                                                                                time_length=self.args.mackrl_epsilon_time_length_level1,
                                                                                decay=self.args.mackrl_epsilon_decay_mode_level1)

            epsilons = ttype(self.batch_size, 1).fill_(self.mackrl_epsilon_decay_schedule_level1.eval(self.T_env))
            self.episode_buffer.set_col(col="mackrl_epsilons_central_level1",
                                        scope="episode",
                                        data=epsilons)

            if not hasattr(self, "mackrl_epsilon_decay_schedule_level2"):
                 self.mackrl_epsilon_decay_schedule_level2 = FlatThenDecaySchedule(start=self.args.mackrl_epsilon_start_level2,
                                                                                finish=self.args.mackrl_epsilon_finish_level2,
                                                                                time_length=self.args.mackrl_epsilon_time_length_level2,
                                                                                decay=self.args.mackrl_epsilon_decay_mode_level2)

            epsilons = ttype(self.batch_size, 1).fill_(self.mackrl_epsilon_decay_schedule_level2.eval(self.T_env))
            self.episode_buffer.set_col(col="mackrl_epsilons_central_level2",
                                        scope="episode",
                                        data=epsilons)

            if not hasattr(self, "mackrl_epsilon_decay_schedule_level3"):
                 self.mackrl_epsilon_decay_schedule_level3 = FlatThenDecaySchedule(start=self.args.mackrl_epsilon_start_level3,
                                                                         finish=self.args.mackrl_epsilon_finish_level3,
                                                                         time_length=self.args.mackrl_epsilon_time_length_level3,
                                                                         decay=self.args.mackrl_epsilon_decay_mode_level3)

            epsilons = ttype(self.batch_size, 1).fill_(self.mackrl_epsilon_decay_schedule_level3.eval(self.T_env))
            self.episode_buffer.set_col(col="mackrl_epsilons_central_level3",
                                        scope="episode",
                                        data=epsilons)

        pass

    def run(self, test_mode, **kwargs):
        self.test_mode = test_mode
        obs_noise_std = kwargs.get("obs_noise_std", None) # bad hack

        # don't reset at initialization as don't have access to hidden state size then
        self.reset(test_mode=test_mode, obs_noise_std=obs_noise_std)

        self.T_env = kwargs.get("T_env", self.T_env)

        terminated = False
        while not terminated:
            # increase episode time counter
            self.t_episode += 1

            # retrieve ids of all envs that have not yet terminated.
            # NOTE: for efficiency reasons, will perform final action selection in terminal state
            ids_envs_not_terminated = [_b for _b in range(self.batch_size) if not self.envs_terminated[_b]]
            ids_envs_not_terminated_tensor = th.cuda.LongTensor(ids_envs_not_terminated) \
                                                if self.episode_buffer.is_cuda \
                                                else th.LongTensor(ids_envs_not_terminated)


            if self.t_episode > 0:

                # flush transition buffer before next step
                self.transition_buffer.flush()

                # get selected actions from last step
                selected_actions, selected_actions_tformat = self.episode_buffer.get_col(col="actions",
                                                                                         t=self.t_episode-1,
                                                                                         agent_ids=list(range(self.n_agents))
                                                                                         )

                ret = self.step(actions=selected_actions[:, ids_envs_not_terminated_tensor.cuda()
                                                     if selected_actions.is_cuda else ids_envs_not_terminated_tensor.cpu(), :, :],
                            ids=ids_envs_not_terminated,
                            obs_noise_std=obs_noise_std)

                # retrieve ids of all envs that have not yet terminated.
                # NOTE: for efficiency reasons, will perform final action selection in terminal state
                ids_envs_not_terminated = [_b for _b in range(self.batch_size) if not self.envs_terminated[_b]]
                ids_envs_not_terminated_tensor = th.cuda.LongTensor(ids_envs_not_terminated) \
                    if self.episode_buffer.is_cuda \
                    else th.LongTensor(ids_envs_not_terminated)

                # update which envs have terminated
                for _id, _v in ret.items():
                    self.envs_terminated[_id] = _v["terminated"]

                # insert new data in transition_buffer into episode buffer
                self.episode_buffer.insert(self.transition_buffer,
                                           bs_ids=list(range(self.batch_size)),
                                           t_ids=self.t_episode,
                                           bs_empty=[_i for _i in range(self.batch_size) if _i not in ids_envs_not_terminated])


                # update episode time counter
                if not self.test_mode and kwargs.get("T_env", None) is None:
                    self.T_env += len(ids_envs_not_terminated)


            # generate multiagent_controller inputs for policy forward pass
            action_selection_inputs, \
            action_selection_inputs_tformat = self.episode_buffer.view(dict_of_schemes=self.multiagent_controller.get_joint_scheme_dict(test_mode),
                                                                       to_cuda=self.args.use_cuda,
                                                                       to_variable=True,
                                                                       bs_ids=ids_envs_not_terminated,
                                                                       t_id=self.t_episode,
                                                                       fill_zero=True, # TODO: DEBUG!!!
                                                                       )

            # retrieve avail_actions from episode_buffer
            avail_actions, avail_actions_format = self.episode_buffer.get_col(bs=ids_envs_not_terminated,
                                                                              col="avail_actions",
                                                                              t = self.t_episode,
                                                                              agent_ids=list(range(self.n_agents)))


            # select actions and retrieve related objects
            if isinstance(self.hidden_states, dict):
                hidden_states = {_k:_v[:, ids_envs_not_terminated_tensor, :, :] for _k, _v in self.hidden_states.items()}
            else:
                hidden_states = self.hidden_states[:, ids_envs_not_terminated_tensor, :,:]


            hidden_states, selected_actions, action_selector_outputs, selected_actions_format = \
                self.multiagent_controller.select_actions(inputs=action_selection_inputs,
                                                          avail_actions=avail_actions,

                                                          tformat=action_selection_inputs_tformat,
                                                          info=dict(T_env=self.T_env),
                                                          hidden_states=hidden_states,
                                                          test_mode=test_mode)

            if isinstance(hidden_states, dict):
                for _k, _v in hidden_states.items():
                    self.hidden_states[_k][:, ids_envs_not_terminated_tensor, :, :] = _v
            else:
                self.hidden_states[:, ids_envs_not_terminated_tensor, :, :] = hidden_states

            for _sa in action_selector_outputs:
                self.episode_buffer.set_col(bs=ids_envs_not_terminated,
                                            col=_sa["name"],
                                            t=self.t_episode,
                                            agent_ids=_sa.get("select_agent_ids", None),
                                            data=_sa["data"])

            # write selected actions to episode_buffer
            if isinstance(selected_actions, list):
               for _sa in selected_actions:
                   try:
                       self.episode_buffer.set_col(bs=ids_envs_not_terminated,
                                                   col=_sa["name"],
                                                   t=self.t_episode,
                                                   agent_ids=_sa.get("select_agent_ids", None),
                                                   data=_sa["data"])
                   except Exception as e:
                       pass
            else:
                self.episode_buffer.set_col(bs=ids_envs_not_terminated,
                                            col="actions",
                                            t=self.t_episode,
                                            agent_ids=list(range(self.n_agents)),
                                            data=selected_actions)

            #Check for termination conditions
            #Check for runner termination conditions
            if self.t_episode == self.max_t_episode:
                terminated = True
            # Check whether all envs have terminated
            if all(self.envs_terminated):
                terminated = True
            # Check whether envs may have failed to terminate
            if self.t_episode == self.env_episode_limit+1 and not terminated:
                assert False, "Envs seem to have failed returning terminated=True, thus not respecting their own episode_limit. Please fix envs."

            pass

        # calculate episode statistics
        kwargs["T_env"] = self.T_env
        self._add_episode_stats(**kwargs)
        return self.episode_buffer


    def log(self, log_directly=True, **kwargs):
        stats = self.get_stats()
        self._stats = deepcopy(stats)
        log_str, log_dict = super().log(log_directly=False, **kwargs)
        if not self.test_mode:
            log_str += ", MACKRL_epsilon_level1={:g}".format(self.mackrl_epsilon_decay_schedule_level1.eval(self.T_env))
            log_str += ", MACKRL_epsilon_level2={:g}".format(self.mackrl_epsilon_decay_schedule_level2.eval(self.T_env))
            log_str += ", MACKRL_epsilon_level3={:g}".format(self.mackrl_epsilon_decay_schedule_level3.eval(self.T_env))
            log_str += ", level2_delegation_rate={:g}".format(_seq_mean(stats["level2_delegation_rate"]))
            log_str += ", policies_level1_entropy={:g}".format(_seq_mean(stats["policy_level1_entropy"]))
            for _i in range(_n_agent_pair_samples(self.n_agents)):
                log_str += ", policies_level2_entropy_sample{}={:g}".format(_i, _seq_mean(stats["policy_level2_entropy_sample{}".format(_i)]))
            log_str += ", policies_level3_entropy={:g}".format(_seq_mean(stats["policy_level3_entropy"]))

        else:
            try:
                log_str += ", level2_delegation_rate={:g}".format(_seq_mean(stats["level2_delegation_rate_test"]))
                log_str += ", policies_level1_entropy={:g}".format(_seq_mean(stats["policy_level1_entropy_test"]))
                for _i in range(_n_agent_pair_samples(self.n_agents)):
                    log_str += ", policies_level2_entropy_sample{}={:g}".format(_i, _seq_mean(stats["policy_level2_entropy_sample{}_test".format(_i)]))
                log_str += ", policies_level3_entropy={:g}".format(_seq_mean(stats["policy_level3_entropy_test"]))
            except:
                log_str += "some info missing - are you testing and noising obs?"

        log_str += ", {}".format(self.multiagent_controller.log(T_env=self.T_env, test_mode=self.test_mode, log_directly=False)[0])

        if not self.test_mode:
            self.logging_struct.py_logger.info("TRAIN RUNNER INFO: {}".format(log_str))
        else:
            self.logging_struct.py_logger.info("TEST RUNNER INFO: {}".format(log_str))

        return log_str, log_dict

    pass


    @staticmethod
    def _loop_worker(envs,
                     in_queue,
                     out_queue,
                     buffer_insert_fn,
                     subproc_id=None,
                     args=None,
                     msg=None,
                     kwargs=None):
        obs_noise_std = kwargs.get("obs_noise_std", None)

        if in_queue is None:
            id, chosen_actions, output_buffer, column_scheme, kwargs = msg
            env_id = id
        else:
            id, chosen_actions, output_buffer, column_scheme, kwargs = in_queue.get() # timeout=1)
            env_id_offset = len(envs) * subproc_id  # TODO: Adjust for multi-threading!
            env_id = id - env_id_offset

        _env = envs[env_id]

        if chosen_actions == "SCHEME":
            env_dict = dict(obs_size=_env.get_obs_size(),
                            state_size=_env.get_state_size(),
                            episode_limit=_env.episode_limit,
                            n_agents = _env.n_agents,
                            n_actions=_env.get_total_actions())
            if args.mackrl_use_obs_intersections:
                env_dict["obs_intersect_pair_size"]= _env.get_obs_intersect_pair_size()
                env_dict["obs_intersect_all_size"] = _env.get_obs_intersect_all_size()
            # Send results back
            ret_msg = dict(id=id, payload=env_dict)
            if out_queue is None:
                return ret_msg
            out_queue.put(ret_msg)
            return

        elif chosen_actions == "RESET":
            _env.reset(obs_noise_std=obs_noise_std) # reset the env!

            # perform environment steps and insert into transition buffer
            observations = _env.get_obs(obs_noise=args.obs_noise)
            state = _env.get_state()
            avail_actions = [_aa + [0] for _aa in _env.get_avail_actions()] #_env.get_avail_actions()  # add place for noop action
            ret_dict = dict(state=state)  # TODO: Check that env_info actually exists
            for _i, _obs in enumerate(observations):
                ret_dict["observations__agent{}".format(_i)] = observations[_i]
            for _i, _obs in enumerate(observations):
                ret_dict["avail_actions__agent{}".format(_i)] = avail_actions[_i]

                # handle observation intersections
                if args.mackrl_use_obs_intersections:
                    ret_dict["obs_intersection_all"], _ = _env.get_obs_intersection(tuple(range(_env.n_agents)))
                    if args.obs_noise:
                        lst = _env.get_obs_intersection_noisy(tuple(range(_env.n_agents)))
                        for _l, lelem in enumerate(lst):
                            ret_dict["obs_intersection_obsid{}_all".format(_l)] = lelem[0]
                    for _i, (_a1, _a2) in enumerate(_ordered_agent_pairings(_env.n_agents)):
                        ret_dict["obs_intersection__pair{}".format(_i)], \
                        ret_dict["avail_actions__pair{}".format(_i)] = _env.get_obs_intersection((_a1, _a2))
                        ret_dict["avail_actions__pair{}".format(_i)] = ret_dict["avail_actions__pair{}".format(_i)].flatten().tolist() + [0]

                        if args.obs_noise:
                            lst = _env.get_obs_intersection_noisy((_a1, _a2))
                            for _l, lelem in enumerate(lst):
                                ret_dict["obs_intersection_obsid{}__pair{}".format(_l, _i)] = lelem[0]

            buffer_insert_fn(id=id, buffer=output_buffer, data_dict=ret_dict, column_scheme=column_scheme)

            # Signal back that queue element was finished processing
            ret_msg = dict(id=id, payload=dict(msg="RESET DONE"))
            if out_queue is None:
                return ret_msg
            out_queue.put(ret_msg)
            return

        elif chosen_actions == "STATS":
            env_stats = _env.get_stats()
            env_dict = dict(env_stats=env_stats)
            # Send results back
            ret_msg = dict(id=id, payload=env_dict)
            if out_queue is None:
                return ret_msg
            out_queue.put(ret_msg)
            return

        else:

            reward, terminated, env_info = \
                _env.step([int(_i) for _i in chosen_actions],
                          obs_noise_std=obs_noise_std)

            # perform environment steps and add to transition buffer
            observations = _env.get_obs(obs_noise=args.obs_noise)
            state = _env.get_state()
            avail_actions = [_aa + [0] for _aa in _env.get_avail_actions()] # _env.get_avail_actions()  # add place for noop action
            terminated = terminated
            truncated = terminated and env_info.get("episode_limit", False)
            ret_dict = dict(state=state,
                            reward=reward,
                            terminated=terminated,
                            truncated=truncated,
                            )
            for _i, _obs in enumerate(observations):
                ret_dict["observations__agent{}".format(_i)] = observations[_i]
            for _i, _obs in enumerate(observations):
                ret_dict["avail_actions__agent{}".format(_i)] = avail_actions[_i]

            if args.mackrl_use_obs_intersections:
                # handle observation intersections
                ret_dict["obs_intersection_all"], _= _env.get_obs_intersection(tuple(range(_env.n_agents)))
                if args.obs_noise:
                    lst = _env.get_obs_intersection_noisy(tuple(range(_env.n_agents)))
                    for _l, lelem in enumerate(lst):
                        ret_dict["obs_intersection_obsid{}_all".format(_l)] = lelem[0]
                for _i, (_a1, _a2) in enumerate(_ordered_agent_pairings(_env.n_agents)):
                    ret_dict["obs_intersection__pair{}".format(_i)],\
                    ret_dict["avail_actions__pair{}".format(_i)] = _env.get_obs_intersection((_a1, _a2))
                    ret_dict["avail_actions__pair{}".format(_i)] = ret_dict["avail_actions__pair{}".format(_i)].flatten().tolist() + [0]

                    if args.obs_noise:
                        lst = _env.get_obs_intersection_noisy((_a1, _a2))
                        for _l, lelem in enumerate(lst):
                            ret_dict["obs_intersection_obsid{}__pair{}".format(_l, _i)] = lelem[0]

            buffer_insert_fn(id=id, buffer=output_buffer, data_dict=ret_dict, column_scheme=column_scheme)

            # Signal back that queue element was finished processing
            ret_msg = dict(id=id, payload=dict(msg="STEP DONE", terminated=terminated))
            if out_queue is None:
                return ret_msg
            else:
                out_queue.put(ret_msg)
            return

        return