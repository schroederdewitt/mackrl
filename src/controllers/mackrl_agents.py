from copy import deepcopy
import numpy as np
import os
from torch.autograd import Variable
import torch as th

from components.action_selectors import REGISTRY as as_REGISTRY
from components import REGISTRY as co_REGISTRY
from components.scheme import Scheme
from components.episode_buffer import BatchEpisodeBuffer
from components.transforms import _build_model_inputs, _join_dicts, \
    _generate_scheme_shapes, _generate_input_shapes, _adim, _bsdim, _tdim, _vdim, _agent_flatten, _check_nan, \
    _to_batch, _from_batch, _vdim, _join_dicts, _underscore_to_cap, _copy_remove_keys, _make_logging_str, _seq_mean, \
    _pad_nan, _onehot

from itertools import combinations
from models import REGISTRY as mo_REGISTRY
from utils.mackrel import _n_agent_pair_samples, _agent_ids_2_pairing_id, _joint_actions_2_action_pair, \
    _pairing_id_2_agent_ids, _pairing_id_2_agent_ids__tensor, _n_agent_pairings, \
    _agent_ids_2_pairing_id, _joint_actions_2_action_pair_aa, _ordered_agent_pairings, _excluded_pair_ids, \
    _ordered_2_agent_pairings

class MACKRLMultiagentController():
    """
    container object for a set of independent agents
    TODO: may need to propagate test_mode in here as well!
    """

    def __init__(self, runner, n_agents, n_actions, action_selector=None, args=None, logging_struct=None):
        self.args = args
        self.runner = runner
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.agent_output_type = "policies"
        self.logging_struct = logging_struct

        self._stats = {}

        self.model_class = mo_REGISTRY[args.mackrl_agent_model]

        # # Set up action selector
        if action_selector is None:
            self.action_selector = as_REGISTRY[args.action_selector](args=self.args)
        else:
            self.action_selector = action_selector

        # b = _n_agent_pairings(self.n_agents)
        self.agent_scheme_level1 = Scheme([#dict(name="actions",
        #                                         rename="past_actions",
        #                                         select_agent_ids=list(range(self.n_agents)),
        #                                         transforms=[("shift", dict(steps=1)),
        #                                                       ("one_hot", dict(range=(0, self.n_actions)))],
        #                                         switch=self.args.mackrl_agent_use_past_actions),
                                           dict(name="mackrl_epsilons_central_level1",
                                                scope="episode"),
                                           #dict(name="observations",
                                           #     select_agent_ids=list(range(self.n_agents)))
                                           #if not self.args.mackrl_use_obs_intersections else
                                           dict(name="obs_intersection_all"),
                                           ])
        self.agent_scheme_level1_noisy = Scheme(self.agent_scheme_level1.scheme_list +
                                                [dict(name="obs_intersection_obsid{}_all".format(_i)) for _i in
                                                 range(self.n_agents)]
                                                )

        self.agent_scheme_level2_fn = lambda _agent_id1, _agent_id2: Scheme([dict(name="agent_id_onehot",
                                                                                  rename="agent_ids",
                                                                                  #transforms=[("one_hot",dict(range=(0, self.n_agents-1)))],
                                                                                  select_agent_ids=[_agent_id1, _agent_id2],),
                                                                             dict(name="agent_id", rename="agent_id__flat", select_agent_ids=[_agent_id1, _agent_id2]),
                                                                             dict(name="mackrl_epsilons_central",
                                                                                  scope="episode"),
                                                                             dict(name="avail_actions",
                                                                                  select_agent_ids=[_agent_id1, _agent_id2]),
                                                                             # dict(name="observations",
                                                                             #      select_agent_ids=[_agent_id1, _agent_id2],
                                                                             #      switch=not self.args.mackrl_use_obs_intersections),
                                                                             dict(name="obs_intersection__pair{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)),
                                                                                  ),
                                                                             dict(name="avail_actions__pair{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)),
                                                                                  ),
                                                                             dict(name="mackrl_epsilons_central_level2",
                                                                                  scope="episode"),
                                                                             dict(name="actions_onehot",
                                                                                  rename="past_actions",
                                                                                  select_agent_ids=list(range(self.n_agents)),
                                                                                  transforms=[("shift", dict(steps=1)),
                                                                                              #("one_hot", dict(range=(0, self.n_actions)))],
                                                                                              ],
                                                                                  switch=self.args.mackrl_agent_use_past_actions),
                                                                             ])

        self.agent_scheme_level2_noisy_fn = \
            lambda _agent_id1, _agent_id2: Scheme(self.agent_scheme_level2_fn(_agent_id1, _agent_id2).scheme_list +
                                                  [dict(
                                                      name="obs_intersection_obsid{}__pair{}".format(_j,
                                                          _agent_ids_2_pairing_id(
                                                              (_agent_id1, _agent_id2),
                                                              self.n_agents)),
                                                  ) for _j in range(2)]
                                                  )

        self.agent_scheme_level3_fn = lambda _agent_id: Scheme([dict(name="agent_id_onehot",
                                                                     rename="agent_id",
                                                                     #transforms=[("one_hot",dict(range=(0, self.n_agents-1)))],
                                                                     select_agent_ids=[_agent_id],),
                                                                dict(name="observations",
                                                                     select_agent_ids=[_agent_id]),
                                                                dict(name="agent_id", rename="agent_id__flat", select_agent_ids=[_agent_id]),
                                                                dict(name="avail_actions", select_agent_ids=[_agent_id]),
                                                                dict(name="mackrl_epsilons_central_level3",
                                                                     scope="episode"),
                                                                dict(name="actions_onehot",
                                                                     rename="past_actions",
                                                                     transforms=[("shift", dict(steps=1)),
                                                                                 #("one_hot", dict(range=(0, self.n_actions)))],
                                                                                 ],
                                                                     select_agent_ids=list(range(self.n_agents)),
                                                                     switch=self.args.mackrl_agent_use_past_actions),
                                                                ])

        # Set up schemes
        self.schemes = {}
        # level 1
        self.schemes_level1 = {}
        self.schemes_level1["agent_input_level1"] = self.agent_scheme_level1

        self.schemes_level1_noisy = {}
        for _obsid in range(self.n_agents):
            self.schemes_level1_noisy["agent_input_level1__agent{}".format(_obsid)] = deepcopy(self.agent_scheme_level1_noisy)

        # level 2
        self.schemes_level2 = {}
        for _agent_id1, _agent_id2 in sorted(combinations(list(range(self.n_agents)), 2)):
            self.schemes_level2["agent_input_level2__agent{}"
                .format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))] \
                = self.agent_scheme_level2_fn(_agent_id1, _agent_id2)

        self.schemes_level2_noisy = {}
        for _obsid in range(2):
            for _agent_id1, _agent_id2 in sorted(combinations(list(range(self.n_agents)), 2)):
                self.schemes_level2_noisy["agent_input_level2__agent{}"
                    .format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)*2+_obsid)] \
                    = self.agent_scheme_level2_noisy_fn(_agent_id1, _agent_id2)

        # level 3
        self.schemes_level3 = {}
        for _agent_id in range(self.n_agents):
            self.schemes_level3["agent_input_level3__agent{}".format(_agent_id)] = self.agent_scheme_level3_fn(_agent_id).agent_flatten()

        # create joint scheme from the agents schemes
        self.joint_scheme_dict_level1 = _join_dicts(self.schemes_level1)
        self.joint_scheme_dict_level2 = _join_dicts(self.schemes_level2)
        self.joint_scheme_dict_level3 = _join_dicts(self.schemes_level3)

        self.joint_scheme_dict = _join_dicts(self.schemes_level1,
                                             self.schemes_level2,
                                             self.schemes_level3)
        self.joint_scheme_dict_noisy = _join_dicts(self.schemes_level1_noisy,
                                                   self.schemes_level2_noisy,
                                                   self.schemes_level3)
        # construct model-specific input regions

        # level 1
        self.input_columns_level1 = {}
        self.input_columns_level1["agent_input_level1"] = {}
        self.input_columns_level1["agent_input_level1"]["main"] = \
            Scheme([dict(name="observations", select_agent_ids=list(range(self.n_agents)))
                        if not self.args.mackrl_use_obs_intersections else
                    dict(name="obs_intersection_all"),
                  ])
        self.input_columns_level1["agent_input_level1"]["epsilons_central_level1"] = \
             Scheme([dict(name="mackrl_epsilons_central_level1",
                          scope="episode")])

        self.input_columns_level1_noisy = {}
        for _aid in range(self.n_agents):
            self.input_columns_level1_noisy["agent_input_level1__agent{}".format(_aid)] = {}
            self.input_columns_level1_noisy["agent_input_level1__agent{}".format(_aid)]["main"] = \
                Scheme([
                        dict(name="obs_intersection_obsid{}_all".format(_aid)),
                      ])
            self.input_columns_level1_noisy["agent_input_level1__agent{}".format(_aid)]["epsilons_central_level1"] = \
                 Scheme([dict(name="mackrl_epsilons_central_level1",
                              scope="episode")])
        # level 2
        self.input_columns_level2 = {}
        for _agent_id1, _agent_id2 in sorted(combinations(list(range(self.n_agents)), 2)):
            self.input_columns_level2["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))] = {}
            self.input_columns_level2["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["main"] = \
                Scheme([dict(name="observations", select_agent_ids=[_agent_id1, _agent_id2])
                        if not self.args.mackrl_use_obs_intersections else
                        dict(name="obs_intersection__pair{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2),self.n_agents))),
                        dict(name="agent_ids", select_agent_ids=[_agent_id1, _agent_id2]),
                        ])
            self.input_columns_level2["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["epsilons_central_level2"] = \
                 Scheme([dict(name="mackrl_epsilons_central_level2",
                              scope="episode")])

            self.input_columns_level2["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["avail_actions_id1"] = Scheme([dict(name="avail_actions", select_agent_ids=[_agent_id1])])
            self.input_columns_level2[
                "agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["avail_actions_id2"] = Scheme([dict(name="avail_actions", select_agent_ids=[_agent_id2])])
            if self.args.mackrl_use_obs_intersections:
                self.input_columns_level2["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))] \
                                            ["avail_actions_pair"] = Scheme([dict(name="avail_actions__pair{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)),
                                                        switch=self.args.mackrl_use_obs_intersections)])
            if hasattr(self.args, "mackrl_delegate_if_zero_ck") and self.args.mackrl_delegate_if_zero_ck:
                self.input_columns_level2["agent_input_level2__agent{}".format(
                    _agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["pair_ck"] = Scheme([dict(name="obs_intersection__pair{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)))])
                # TODO: WEIRD
                self.input_columns_level2["agent_input_level2__agent{}".format(
                    _agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents))]["actions_level1__sample0"] = Scheme([dict(name="obs_intersection__pair{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)))])

        self.input_columns_level2_noisy = {}
        for _obsid in range(2):
            for _agent_id1, _agent_id2 in sorted(combinations(list(range(self.n_agents)), 2)):
                self.input_columns_level2_noisy["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)*2+_obsid)] = {}
                self.input_columns_level2_noisy["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)*2+_obsid)]["main"] = \
                    Scheme([dict(name="obs_intersection_obsid{}__pair{}".format(_obsid, _agent_ids_2_pairing_id((_agent_id1, _agent_id2),self.n_agents))),
                            dict(name="agent_ids", select_agent_ids=[_agent_id1, _agent_id2]),
                            ])
                self.input_columns_level2_noisy["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)*2+_obsid)]["epsilons_central_level2"] = \
                     Scheme([dict(name="mackrl_epsilons_central_level2",
                                  scope="episode")])

                self.input_columns_level2_noisy["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)*2+_obsid)]["avail_actions_id1"] = Scheme([dict(name="avail_actions", select_agent_ids=[_agent_id1])])
                self.input_columns_level2_noisy[
                    "agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)*2+_obsid)]["avail_actions_id2"] = Scheme([dict(name="avail_actions", select_agent_ids=[_agent_id2])])
                if self.args.mackrl_use_obs_intersections:
                    self.input_columns_level2_noisy["agent_input_level2__agent{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)*2+_obsid)] \
                                                ["avail_actions_pair"] = Scheme([dict(name="avail_actions__pair{}".format(_agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)),
                                                            switch=self.args.mackrl_use_obs_intersections)])
                if hasattr(self.args, "mackrl_delegate_if_zero_ck") and self.args.mackrl_delegate_if_zero_ck:
                    self.input_columns_level2_noisy["agent_input_level2__agent{}".format(
                        _agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)*2+_obsid)]["pair_ck"] = Scheme([dict(name="obs_intersection_obsid{}__pair{}".format(_obsid, _agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)))])
                    # TODO: WEIRD
                    self.input_columns_level2_noisy["agent_input_level2__agent{}".format(
                        _agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)*2+_obsid)]["actions_level1__sample0"] = Scheme([dict(name="obs_intersection_obsid{}__pair{}".format(_obsid, _agent_ids_2_pairing_id((_agent_id1, _agent_id2), self.n_agents)))])


        # level 3
        self.input_columns_level3 = {}
        for _agent_id in range(self.n_agents):
            self.input_columns_level3["agent_input_level3__agent{}".format(_agent_id)] = {}
            self.input_columns_level3["agent_input_level3__agent{}".format(_agent_id)]["main"] = \
                Scheme([dict(name="observations", select_agent_ids=[_agent_id]),
                        dict(name="agent_id", select_agent_ids=[_agent_id]),
                        dict(name="past_actions",
                             select_agent_ids=[_agent_id],
                             switch=self.args.mackrl_agent_use_past_actions),
                        ])
            self.input_columns_level3["agent_input_level3__agent{}".format(_agent_id)]["epsilons_central_level3"] = \
                Scheme([dict(name="mackrl_epsilons_central_level3",
                             scope="episode")])
            self.input_columns_level3["agent_input_level3__agent{}".format(_agent_id)]["avail_actions"] = \
                Scheme([dict(name="avail_actions",
                             select_agent_ids=[_agent_id])])

        self.magic_map = th.tensor(_ordered_2_agent_pairings(self.n_agents, self.args),
                                   device="cuda" if self.args.use_cuda else "cpu")
        pass

    def get_joint_scheme_dict(self, test_mode):
        if self.is_obs_noise(test_mode=test_mode): # and test_mode: DEBUG
            return self.joint_scheme_dict_noisy
        else:
            return self.joint_scheme_dict

    def get_parameters(self):
        return list(self.model.parameters())

    def is_obs_noise(self, test_mode):
        if self.args.obs_noise and test_mode: # DEBUG
            return True
        else:
            return False

    def select_actions(self, inputs, avail_actions, tformat, info, hidden_states=None, test_mode=False, **kwargs):
        """
        sample from the MACKRL tree
        """
        noise_params = kwargs.get("noise_params", None)

        T_env = info["T_env"]
        test_suffix = "" if not test_mode else "_test"

        if self.args.agent_level1_share_params:

            # --------------------- LEVEL 1

            if self.is_obs_noise(test_mode):
                inputs_level1, inputs_level1_tformat = _build_model_inputs(self.input_columns_level1_noisy,
                                                                           inputs,
                                                                           to_variable=True,
                                                                           inputs_tformat=tformat)
                inputs_level1_tformat = "a*bs*t*v"
            else:
                inputs_level1, inputs_level1_tformat = _build_model_inputs(self.input_columns_level1,
                                                                           inputs,
                                                                           to_variable=True,
                                                                           inputs_tformat=tformat)
            if self.args.debug_mode:
                _check_nan(inputs_level1)

            out_level1, hidden_states_level1, losses_level1, tformat_level1 = self.model.model_level1(inputs_level1["agent_input_level1"],
                                                                                                      hidden_states=hidden_states["level1"],
                                                                                                      loss_fn=None,
                                                                                                      tformat=inputs_level1_tformat,
                                                                                                      n_agents=self.n_agents,
                                                                                                      test_mode=test_mode,
                                                                                                      **kwargs)


            if self.args.debug_mode:
                _check_nan(inputs_level1)

            if self.is_obs_noise(test_mode):
                # have to do correlated sampling of what pair id everyone agrees on
                bs = out_level1.shape[_bsdim(inputs_level1_tformat)]
                ftype = th.FloatTensor if not out_level1.is_cuda else th.cuda.FloatTensor
                sampled_pair_ids = ftype(*out_level1.shape[:-1], 1)
                for _b in range(bs):
                    ps = out_level1[:, _b]
                    rn = np.random.random()
                    for _a in range(ps.shape[0]):
                        act = 0
                        s = ps[_a, 0, act]
                        while s <= rn:
                            act += 1
                            s += ps[_a, 0, act]
                        sampled_pair_ids[_a, _b, 0, :] = act

                modified_inputs_level1 = inputs_level1
                selected_actions_format_level1 = "a*bs*t*v"
            else:
                # TODO: This is the pair-product encoded ID of both selected pairs.
                sampled_pair_ids, modified_inputs_level1, selected_actions_format_level1 = self.action_selector.select_action({"policies":out_level1},
                                                                                                                               avail_actions=None,
                                                                                                                               tformat=tformat_level1,
                                                                                                                               test_mode=test_mode)
            _check_nan(sampled_pair_ids)

            if self.args.debug_mode in ["level2_actions_fixed_pair"]:
                """
                DEBUG MODE: LEVEL2 ACTIONS FIXED PAIR
                Here we pick level2 actions from a fixed agent pair (0,1) and the third action from IQL
                """
                assert self.n_agents == 3, "only makes sense in n_agents=3 scenario"
                sampled_pair_ids.fill_(0.0)

                # sample which pairs should be selected
            # TODO: HAVE TO ADAPT THIS FOR NOISY OBS!
            if self.is_obs_noise(test_mode):
                self.selected_actions_format = selected_actions_format_level1
            else:
                self.actions_level1 = sampled_pair_ids.clone()
                self.selected_actions_format = selected_actions_format_level1
                self.policies_level1 = modified_inputs_level1.squeeze(0).clone()

            if self.is_obs_noise(test_mode):
                inputs_level2, inputs_level2_tformat = _build_model_inputs(self.input_columns_level2_noisy,
                                                                           inputs,
                                                                           to_variable=True,
                                                                           inputs_tformat=tformat,
                                                                           )
            else:
                inputs_level2, inputs_level2_tformat = _build_model_inputs(self.input_columns_level2,
                                                                           inputs,
                                                                           to_variable=True,
                                                                           inputs_tformat=tformat,
                                                                           )

            assert self.args.agent_level2_share_params, "not implemented!"


            if "avail_actions_pair" in inputs_level2["agent_input_level2"]:
                pairwise_avail_actions = inputs_level2["agent_input_level2"]["avail_actions_pair"]
            else:
                assert False, "NOT SUPPORTED CURRENTLY."
                avail_actions1, params_aa1, tformat_aa1 = _to_batch(inputs_level2["agent_input_level2"]["avail_actions_id1"], inputs_level2_tformat)
                avail_actions2, params_aa2, _ = _to_batch(inputs_level2["agent_input_level2"]["avail_actions_id2"], inputs_level2_tformat)
                pairwise_avail_actions = th.bmm(avail_actions1.unsqueeze(2), avail_actions2.unsqueeze(1))
                pairwise_avail_actions = _from_batch(pairwise_avail_actions, params_aa2, tformat_aa1)

            ttype = th.cuda.FloatTensor if pairwise_avail_actions.is_cuda else th.FloatTensor
            delegation_avails = Variable(ttype(pairwise_avail_actions.shape[0],
                                               pairwise_avail_actions.shape[1],
                                               pairwise_avail_actions.shape[2], 1).fill_(1.0), requires_grad=False)
            pairwise_avail_actions = th.cat([delegation_avails, pairwise_avail_actions], dim=_vdim(tformat))


            out_level2, hidden_states_level2, losses_level2, tformat_level2 \
                = self.model.models["level2_{}".format(0)](inputs_level2["agent_input_level2"],
                                                            hidden_states=hidden_states["level2"],
                                                            loss_fn=None,
                                                            tformat=inputs_level2_tformat,
                                                            # sampled_pair_ids=sampled_pair_ids, # UNUSED?
                                                            pairwise_avail_actions=pairwise_avail_actions,
                                                            test_mode=test_mode,
                                                            seq_lens=inputs["agent_input_level2__agent0"].seq_lens,
                                                            **kwargs)

            if self.is_obs_noise(test_mode):

                # have to do correlated sampling of what pair id everyone agrees on
                bs = out_level2.shape[_bsdim(inputs_level2_tformat)]
                ftype = th.FloatTensor if not out_level2.is_cuda else th.cuda.FloatTensor
                pair_sampled_actions = ftype(*out_level2.shape[:-1], 1).view(int(out_level2.shape[0]/2),
                                                                             2,
                                                                             *out_level2.shape[1:-1],
                                                                             1)
                for _b in range(bs):
                    ps = out_level2.view(int(out_level2.shape[0]/2),
                                         2,
                                         *out_level2.shape[1:])[:, :, _b]
                    avail_actions = pairwise_avail_actions.view(int(out_level2.shape[0]/2),
                                         2,
                                         *out_level2.shape[1:])[:, :, _b]

                    _sum0 = th.sum(ps[:, 0] * avail_actions[:, 0], dim=-1, keepdim=True)
                    _sum0_mask = (_sum0 == 0.0)
                    _sum0.masked_fill_(_sum0_mask, 1.0)
                    ps[:, 0] = ps[:, 0] * avail_actions[:, 0] / _sum0

                    _sum1 = th.sum(ps[:, 1] * avail_actions[:, 1], dim=-1, keepdim=True)
                    _sum1_mask = (_sum1 == 0.0)
                    _sum1.masked_fill_(_sum1_mask, 1.0)
                    ps[:, 1] = ps[:, 1] * avail_actions[:, 1] / _sum1

                    rns = np.random.random(ps.shape[0]) #one seed for each pair / batch
                    for _a in range(ps.shape[0]):
                        for _j in range(2):
                            act = 0
                            s = ps[_a, _j, 0, act]
                            while s <= rns[_a]:
                                act += 1
                                s += ps[_a, _j, 0, act]
                            if act == 122: # DEBUG
                                a = 5
                                pass
                            pair_sampled_actions[_a, _j, _b, 0, :] = act

                # TODO: Fix the return values so I can debug in episode buffer!!!
                modified_inputs_level2 = inputs_level2
                selected_actions_format_level2 = "a*bs*t*v"
            else:

                # TODO: Implement for noisy obs!! # Need again correlated sampling
                pair_sampled_actions, \
                modified_inputs_level2, \
                selected_actions_format_level2 = self.action_selector.select_action({"policies":out_level2},
                                                                                    avail_actions=pairwise_avail_actions.data,
                                                                                    tformat=tformat_level2,
                                                                                    test_mode=test_mode)
            # if th.sum(pair_sampled_actions == 26.0) > 0.0:
            #     a = 5

            if sampled_pair_ids.shape[_tdim(tformat_level1)] > 1: # only used for mackrl sampling
                sampled_pairs = th.cat([ self.magic_map[sampled_pair_ids[:,:,_t:_t+1,:].long()].squeeze(2) for _t in range(sampled_pair_ids.shape[_tdim(tformat_level1)]) ],
                                       dim=_tdim(tformat_level1))
            else:
                sampled_pairs = self.magic_map[sampled_pair_ids.long()].squeeze(2)

            self.actions_level2 = pair_sampled_actions.clone()

            if self.is_obs_noise(test_mode):
                self.actions_level2_sampled = []
                for _aid in range(self.n_agents):
                    self.actions_level2_sampled.append([])
                    for i in range(sampled_pairs.shape[-1]):
                        self.actions_level2_sampled[_aid].append(
                            pair_sampled_actions[:, i].gather(0, sampled_pairs[_aid:_aid+1, :, :, i:i + 1].long()))
                    self.actions_level2_sampled[_aid] = th.cat(self.actions_level2_sampled[_aid], 0)
            else:
                # ToDO: Gather across all selected pairs!!
                self.actions_level2_sampled = []
                for i in range(sampled_pairs.shape[-1]):
                    self.actions_level2_sampled.append(pair_sampled_actions.gather(0, sampled_pairs[:,:,:,i:i+1].long()))

                self.actions_level2_sampled = th.cat(self.actions_level2_sampled, 0)
                self.selected_actions_format_level2 = selected_actions_format_level2
                self.policies_level2 = modified_inputs_level2.clone()


            inputs_level3, inputs_level3_tformat = _build_model_inputs(self.input_columns_level3,
                                                                       inputs,
                                                                       to_variable=True,
                                                                       inputs_tformat=tformat,
                                                                       )

            action_tensor = None
            if self.is_obs_noise(test_mode):
                action_tensor = ttype(self.n_agents,
                                      sampled_pairs.shape[_bsdim(tformat)],
                                      sampled_pairs.shape[_tdim(tformat)],
                                      1).fill_(float("nan"))
                for _bid in range(sampled_pairs.shape[_bsdim(tformat)]):
                    # each agent has it's own assumptions about what pair-wise actions were sampled!
                    for _aid in range(self.n_agents):
                        # work out which pair id agent _aid is in (if any) and whether at first or second position
                        partid = None
                        posid = None
                        #for _partid, _part in enumerate(_ordered_2_agent_pairings(self.n_agents)):
                        combid = int(sampled_pair_ids[_aid, _bid, 0, 0].item())
                        part = list(_ordered_2_agent_pairings(self.n_agents))[combid]
                        for pid, p in enumerate(part):
                            agentids = _pairing_id_2_agent_ids(p, self.n_agents)
                            if agentids[0] == _aid:
                                partid = pid
                                posid = 0
                                break
                            if agentids[1] == _aid:
                                partid = pid
                                posid = 1
                                break
                            pass
                        if partid is not None:
                            # ok so what actions did agent _aid finally select?
                            joint_act = self.actions_level2_sampled[_aid][partid,_bid,0,0].item()
                            joint_act_dec = _joint_actions_2_action_pair(int(joint_act), self.n_actions)
                            if joint_act_dec == 11: # DEBUG
                                a = 5
                            if joint_act_dec != 0: # else delegate
                                action_tensor[_aid,_bid,0,:] = joint_act_dec[posid]
                        else:
                            # decentralized anyway!
                            pass
            else:
                action_tensor = ttype(self.n_agents,
                                      pair_sampled_actions.shape[_bsdim(tformat)],
                                      pair_sampled_actions.shape[_tdim(tformat)],
                                      1).fill_(float("nan"))
                for i in range(sampled_pairs.shape[-1]):
                    sampled_pair = sampled_pairs[:,:,:,i:i+1]
                    pair_id1, pair_id2 = _pairing_id_2_agent_ids__tensor(sampled_pair, self.n_agents,
                                                                         "a*bs*t*v")  # sampled_pair_ids.squeeze(0).squeeze(2).view(-1), self.n_agents)

                    avail_actions1 = inputs_level3["agent_input_level3"]["avail_actions"].gather(
                        _adim(inputs_level3_tformat), Variable(pair_id1.repeat(1, 1, 1, inputs_level3["agent_input_level3"][
                            "avail_actions"].shape[_vdim(inputs_level3_tformat)])))
                    avail_actions2 = inputs_level3["agent_input_level3"]["avail_actions"].gather(
                        _adim(inputs_level3_tformat), Variable(pair_id2.repeat(1, 1, 1, inputs_level3["agent_input_level3"][
                            "avail_actions"].shape[_vdim(inputs_level3_tformat)])))

                    # selected_level_2_actions = pair_sampled_actions.gather(0, sampled_pair_ids.long())
                    this_pair_sampled_actions = pair_sampled_actions.gather(0, sampled_pair.long())

                    actions1, actions2 = _joint_actions_2_action_pair_aa(this_pair_sampled_actions.clone(),
                                                                         self.n_actions,
                                                                         avail_actions1,
                                                                         avail_actions2)
                    # count how often level2 actions are un-available at level 3
                    # TODO: Verify that 'this_pair_sampled_actions != 0' is the right thing to do!!
                    pair_action_unavail_rate = (th.mean(((actions1 != actions1) & (this_pair_sampled_actions != 0)).float()).item() +
                                                th.mean(((actions2 != actions2) & (this_pair_sampled_actions != 0)).float()).item()) / 2.0
                    if pair_action_unavail_rate != 0.0 and hasattr(self.args, "mackrl_delegate_if_zero_ck") and self.args.mackrl_delegate_if_zero_ck:
                        #assert False, "pair action unavail HAS to be zero in mackrl_delegate_if_zero_ck setting!"
                        self.logging_struct.py_logger.warning("ERROR: pair action unavail HAS to be zero in mackrl_delegate_if_zero_ck setting!")

                self._add_stat("pair_action_unavail_rate__runner",
                               pair_action_unavail_rate,
                               T_env=T_env,
                               suffix=test_suffix,
                               to_sacred=False)

                # Now check whether any of the pair_sampled_actions violate individual agent constraints on avail_actions
                ttype = th.cuda.FloatTensor if self.args.use_cuda else th.FloatTensor


                action_tensor.scatter_(0, pair_id1, actions1)
                action_tensor.scatter_(0, pair_id2, actions2)

            avail_actions_level3 = inputs_level3["agent_input_level3"]["avail_actions"].clone().data
            self.avail_actions = avail_actions_level3.clone()

            inputs_level3["agent_input_level3"]["avail_actions"] = Variable(avail_actions_level3,
                                                                            requires_grad=False)

            out_level3, hidden_states_level3, losses_level3, tformat_level3 = self.model.models["level3_{}".format(0)](inputs_level3["agent_input_level3"],
                                                                                                                       hidden_states=hidden_states["level3"],
                                                                                                                       loss_fn=None,
                                                                                                                       tformat=inputs_level3_tformat,
                                                                                                                       test_mode=test_mode,
                                                                                                                       seq_lens=inputs["agent_input_level3__agent0"].seq_lens,
                                                                                                                       **kwargs)
            # extract available actions
            avail_actions_level3 = inputs_level3["agent_input_level3"]["avail_actions"]

            individual_actions, \
            modified_inputs_level3, \
            selected_actions_format_level3 = self.action_selector.select_action({"policies":out_level3},
                                                                                 avail_actions=avail_actions_level3.data,
                                                                                 tformat=tformat_level3,
                                                                                 test_mode=test_mode)

            self.actions_level3 = individual_actions
            action_tensor[action_tensor != action_tensor] = individual_actions[action_tensor != action_tensor]

            # set states beyond episode termination to NaN
            if self.is_obs_noise(test_mode):
                action_tensor = _pad_nan(action_tensor, tformat=tformat_level3,
                                         seq_lens=inputs["agent_input_level1__agent0"].seq_lens)  # DEBUG
            else:
                action_tensor = _pad_nan(action_tensor, tformat=tformat_level3, seq_lens=inputs["agent_input_level1"].seq_lens) # DEBUG
            # l2 = action_tensor.squeeze()  # DEBUG
            if self.args.debug_mode in ["level3_actions_only"]:
                """
                DEBUG MODE: LEVEL3 ACTIONS ONLY
                Here we just pick actions from level3 - should therefore just correspond to vanilla COMA!
                """
                action_tensor  = individual_actions

            self.final_actions = action_tensor.clone()
            if th.sum(self.final_actions == 11).item() > 0: # DEBUG
                a = 5
                pass

            if self.is_obs_noise(test_mode):
                selected_actions_list = []
                selected_actions_list += [dict(name="actions",
                                               select_agent_ids=list(range(self.n_agents)),
                                               data=self.final_actions)]
                modified_inputs_list = []
            else:
                #self.actions_level3 = individual_actions.clone()
                self.selected_actions_format_level3 = selected_actions_format_level3
                self.policies_level3 = modified_inputs_level3.clone()
                self.avail_actions_active = avail_actions_level3.data

                selected_actions_list = []
                for _i in range(_n_agent_pair_samples(self.n_agents) if self.args.n_pair_samples is None else self.args.n_pair_samples): #_n_agent_pair_samples(self.n_agents)):
                    selected_actions_list += [dict(name="actions_level1__sample{}".format(_i),
                                                   data=self.actions_level1[_i])]
                for _i in range(_n_agent_pair_samples(self.n_agents)):
                    selected_actions_list += [dict(name="actions_level2__sample{}".format(_i),
                                                   data=self.actions_level2_sampled[_i])] # TODO: BUG!?
                selected_actions_list += [dict(name="actions_level2",
                                               select_agent_ids=list(range(_n_agent_pairings(self.n_agents))),
                                               data=self.actions_level2)]
                selected_actions_list += [dict(name="actions_level3",
                                               select_agent_ids=list(range(self.n_agents)),
                                               data=self.actions_level3)]
                selected_actions_list += [dict(name="actions",
                                               select_agent_ids=list(range(self.n_agents)),
                                               data=self.final_actions)]

                modified_inputs_list = []
                modified_inputs_list += [dict(name="policies_level1",
                                              data=self.policies_level1)]
                for _i in range(_n_agent_pair_samples(self.n_agents)):
                    modified_inputs_list += [dict(name="policies_level2__sample{}".format(_i),
                                                  data=self.policies_level2[_i])]
                modified_inputs_list += [dict(name="policies_level3",
                                              select_agent_ids=list(range(self.n_agents)),
                                              data=self.policies_level3)]
                modified_inputs_list += [dict(name="avail_actions_active",
                                              select_agent_ids=list(range(self.n_agents)),
                                              data=self.avail_actions_active)]
                modified_inputs_list += [dict(name="avail_actions",
                                              select_agent_ids=list(range(self.n_agents)),
                                              data=self.avail_actions)]

                #modified_inputs_list += [dict(name="avail_actions",
                #                              select_agent_ids=list(range(self.n_agents)),
                #                              data=self.avail_actions)]

                selected_actions_list += [dict(name="actions_onehot",
                                               select_agent_ids=list(range(self.n_agents)),
                                               data=_onehot(self.final_actions, rng=(0, self.n_actions)))]

            hidden_states = dict(level1=hidden_states_level1,
                                 level2=hidden_states_level2,
                                 level3=hidden_states_level3)

            return hidden_states, selected_actions_list, modified_inputs_list, self.selected_actions_format

            pass

        else:
            assert False, "Not implemented"

    def create_model(self, transition_scheme):

        self.scheme_shapes_level1 = _generate_scheme_shapes(transition_scheme=transition_scheme,
                                                        dict_of_schemes=self.schemes_level1)

        self.input_shapes_level1 = _generate_input_shapes(input_columns=self.input_columns_level1,
                                                          scheme_shapes=self.scheme_shapes_level1)

        self.scheme_shapes_level2 = _generate_scheme_shapes(transition_scheme=transition_scheme,
                                                            dict_of_schemes=self.schemes_level2)

        self.input_shapes_level2 = _generate_input_shapes(input_columns=self.input_columns_level2,
                                                          scheme_shapes=self.scheme_shapes_level2)

        self.scheme_shapes_level3 = _generate_scheme_shapes(transition_scheme=transition_scheme,
                                                            dict_of_schemes=self.schemes_level3)

        self.input_shapes_level3 = _generate_input_shapes(input_columns=self.input_columns_level3,
                                                          scheme_shapes=self.scheme_shapes_level3)


        # TODO: Set up agent models

        self.model = self.model_class(input_shapes=dict(level1=self.input_shapes_level1["agent_input_level1"],
                                                        level2=self.input_shapes_level2["agent_input_level2__agent0"],
                                                        level3=self.input_shapes_level3["agent_input_level3__agent0"]),
                                      n_agents=self.n_agents,
                                      n_actions=self.n_actions,
                                      model_classes=dict(level1=mo_REGISTRY[self.args.mackrl_agent_model_level1],
                                                         level2=mo_REGISTRY[self.args.mackrl_agent_model_level2],
                                                         level3=mo_REGISTRY[self.args.mackrl_agent_model_level3]),
                                      args=self.args)

        return

    def generate_initial_hidden_states(self, batch_size, test_mode=False, caller=None):
        """
        generates initial hidden states for each agent
        """

        # Set up hidden states for all levels - and propagate through the runner!
        hidden_dict = {}
        hidden_dict["level1"] = th.stack([Variable(th.zeros(batch_size, 1, self.args.agents_hidden_state_size)) for _
                                        in range(self.n_agents if self.is_obs_noise(test_mode) and caller != "learner" else 1)])
        hidden_dict["level2"] = th.stack([Variable(th.zeros(batch_size, 1, self.args.agents_hidden_state_size)) for _
                                        in range(len(sorted(combinations(list(range(self.n_agents)), 2)))*2
                                                 if self.is_obs_noise(test_mode) and caller != "learner"  else
                                                 len(sorted(combinations(list(range(self.n_agents)), 2))))])
        hidden_dict["level3"] = th.stack([Variable(th.zeros(batch_size, 1, self.args.agents_hidden_state_size)) for _
                                        in range(self.n_agents)])
        if self.args.use_cuda:
            hidden_dict = {_k:_v.cuda() for _k, _v in hidden_dict.items()}

        return hidden_dict, "?*bs*v*t"

    def share_memory(self):
        assert False, "TODO"
        pass

    def get_outputs(self, inputs, hidden_states, tformat, loss_fn=None, actions=None, **kwargs):

        if loss_fn is None or actions is None:
            assert False, "not implemented - always need loss function and selected actions!"

        if self.args.share_agent_params:
            inputs_level1, inputs_level1_tformat = _build_model_inputs(self.input_columns_level1,
                                                                       inputs,
                                                                       inputs_tformat=tformat,
                                                                       to_variable=True)

            inputs_level2, inputs_level2_tformat = _build_model_inputs(self.input_columns_level2,
                                                                       inputs,
                                                                       to_variable=True,
                                                                       inputs_tformat=tformat,
                                                                       )

            inputs_level3, inputs_level3_tformat = _build_model_inputs(self.input_columns_level3,
                                                                       inputs,
                                                                       to_variable=True,
                                                                       inputs_tformat=tformat,
                                                                       )

            out, hidden_states, losses, tformat = self.model(inputs=dict(level1=inputs_level1,
                                                                         level2=inputs_level2,
                                                                         level3=inputs_level3),
                                                             hidden_states=hidden_states,
                                                             loss_fn=loss_fn,
                                                             tformat=dict(level1=inputs_level1_tformat,
                                                                          level2=inputs_level2_tformat,
                                                                          level3=inputs_level3_tformat),
                                                             n_agents=self.n_agents,
                                                             actions=actions,
                                                             seq_lens=inputs["agent_input_level3__agent0"].seq_lens,
                                                             **kwargs)

            ret_dict = dict(hidden_states = hidden_states,
                            losses = losses[0] if losses is not None else None )
            ret_dict[self.agent_output_type] = out
            return ret_dict, tformat #losses[loss_level] if loss_level is not None else None), tformat_level3

        else:
            assert False, "Not yet implemented."

        pass

    def save_models(self, path, token, T):
        if not os.path.isdir(os.path.join(path, token)):
            os.makedirs(os.path.join(path, token))
        th.save(self.model.state_dict(),
                os.path.join(path, token, "{}_agentsp__{}_T.weights".format(self.args.learner, T)))
        pass

    def load_model(self, path, T=0):
        self.T_env = T
        self.model.load_state_dict(th.load(path))
        #th.save(self.model.state_dict(),
        #        os.path.join(path, token, "{}_agentsp__{}_T.weights".format(self.args.learner, T)))
        pass

    def _add_stat(self, name, value, T_env, suffix="", to_sacred=True, to_tb=True):
        name += suffix

        if isinstance(value, np.ndarray) and value.size == 1:
            value = float(value)

        if not hasattr(self, "_stats"):
            self._stats = {}

        if name not in self._stats:
            self._stats[name] = []
            self._stats[name+"_T_env"] = []
        self._stats[name].append(value)
        self._stats[name+"_T_env"].append(T_env)

        if hasattr(self, "max_stats_len") and len(self._stats) > self.max_stats_len:
            self._stats[name].pop(0)
            self._stats[name+"_T_env"].pop(0)

        # log to sacred if enabled
        if hasattr(self.logging_struct, "sacred_log_scalar_fn") and to_sacred:
            self.logging_struct.sacred_log_scalar_fn(key=_underscore_to_cap(name), val=value)

        # log to tensorboard if enabled
        if hasattr(self.logging_struct, "tensorboard_log_scalar_fn") and to_tb:
            self.logging_struct.tensorboard_log_scalar_fn(_underscore_to_cap(name), value, T_env)

        return

    def log(self, test_mode=None, T_env=None, log_directly = True):
        """
        Each learner has it's own logging routine, which logs directly to the python-wide logger if log_directly==True,
        and returns a logging string otherwise

        Logging is triggered in run.py
        """
        test_suffix = "" if not test_mode else "_test"

        stats = self.get_stats()
        try:
            stats["pair_action_unavail_rate"+test_suffix] = _seq_mean(stats["pair_action_unavail_rate__runner"+test_suffix])

            self._add_stat("pair_action_unavail_rate",
                           stats["pair_action_unavail_rate"+test_suffix],
                           T_env=T_env,
                           suffix=test_suffix,
                           to_sacred=True)
        except:
            pass

        if stats == {}:
            self.logging_struct.py_logger.warning("Stats is empty... are you logging too frequently?")
            return "", {}

        logging_dict =  dict(T_env=T_env)

        try:
            logging_dict["pair_action_unavail_rate"+test_suffix] =stats["pair_action_unavail_rate"+test_suffix]
        except:
            pass

        logging_str = ""
        logging_str += _make_logging_str(_copy_remove_keys(logging_dict, ["T_env"+test_suffix]))


        if log_directly:
            self.logging_struct.py_logger.info("{} MC INFO: {}".format("TEST" if self.test_mode else "TRAIN",
                                                                           logging_str))
        return logging_str, logging_dict


    def get_stats(self):
        if hasattr(self, "_stats"):
            tmp = deepcopy(self._stats)
            self._stats={}
            return tmp
        else:
            return []


    pass




