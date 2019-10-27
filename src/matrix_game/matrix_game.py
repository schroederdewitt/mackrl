# This notebook implements a proof-of-principle for
# Multi-Agent Common Knowledge Reinforcement Learning (MACKRL)
# The entire notebook can be executed online, no need to download anything

# http://pytorch.org/
from itertools import chain

import torch
import torch.nn.functional as F
from torch.multiprocessing import Pool, set_start_method, freeze_support
try:
     set_start_method('spawn')
except RuntimeError:
    pass

from torch.nn import init
from torch.optim import Adam, SGD
import numpy as np

import matplotlib.pyplot as plt

use_cuda =  False

payoff_values = []
payoff_values.append(torch.tensor([  # payoff values
    [5, 0, 0, 2, 0],
    [0, 1, 2, 4, 2],
    [0, 0, 0, 2, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=torch.float32) * 0.2)
payoff_values.append(
    torch.tensor([  # payoff values
        [0, 0, 1, 0, 5],
        [0, 0, 2, 0, 0],
        [1, 2, 4, 2, 1],
        [0, 0, 2, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=torch.float32) * 0.2)

n_agents = 2
n_actions = len(payoff_values[0])
n_states_dec = 5
n_states_joint = 3
n_mix_hidden = 3

p_observation = 0.5
p_ck_noise = [0.0]

# Number of gradient steps
t_max = 202

# We'll be using a high learning rate, since we have exact gradients
lr = 0.05 # DEBUG: 0.05 if exact gradients!
optim = 'adam'

# You can reduce this number if you are short on time. (Eg. n_trials = 20)
#n_trials = 100 # 30
n_trials = 20 #15 #100
std_val = 1.0

# These are the 3 settings we run: MACRKL, Joint-action-learner (always uses CK),
# Independent Actor-Critic (always uses decentralised actions selection)
labels = ["IAC", "JAL"]
p_vec = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

final_res = []
# # Pair-Controller with 3 input state (no CK, CK & Matrix ID = 0, CK & Matrix ID = 1), n_actions^2 actions for
# # joint action + 1 action for delegation to the independent agents.
# theta_joint = init.normal_(torch.zeros(n_states_joint, n_actions ** 2 + 1, requires_grad=True), std=0.1)

# Produce marginalised policy: pi_pc[0] * pi^a * pi^b + p(u^ab)
def p_joint_all(pi_pc, pi_dec):
    p_joint = pi_pc[1:].view(n_actions, n_actions).clone()
    pi_a_pi_b = torch.ger(pi_dec[0], pi_dec[1])
    p_joint = pi_pc[0] * pi_a_pi_b + p_joint
    return p_joint

def p_joint_all_noise_alt(pi_pcs, pi_dec, p_ck_noise, ck_state):

    p_none = (1-p_ck_noise) ** 2 # both unnoised
    p_both = (p_ck_noise) ** 2 # both noised
    p_one = (1-p_ck_noise) * p_ck_noise # exactly one noised

    p_marg_ag0_ck1 = pi_pcs[1][1:].view(n_actions, n_actions).clone().sum(dim=0)
    p_marg_ag0_ck2 = pi_pcs[2][1:].view(n_actions, n_actions).clone().sum(dim=0)
    p_marg_ag1_ck1 = pi_pcs[1][1:].view(n_actions, n_actions).clone().sum(dim=1)
    p_marg_ag1_ck2 = pi_pcs[2][1:].view(n_actions, n_actions).clone().sum(dim=1)

    p_joint_ck0 = pi_pcs[0][1:].view(n_actions, n_actions).clone()
    p_joint_ck1 = pi_pcs[1][1:].view(n_actions, n_actions).clone()
    p_joint_ck2 = pi_pcs[2][1:].view(n_actions, n_actions).clone()

    p_d_ck0 = pi_pcs[0][0]
    p_d_ck1 = pi_pcs[1][0]
    p_d_ck2 = pi_pcs[2][0]

    def make_joint(p1, p2, mode="interval"):
        """
        1. Pick uniform random variable between [0,1]
        2. Do multinomial sampling through contiguous, ordered bucketing for both p1, p2
        """
        p1 = p1.clone().view(-1)
        p2 = p2.clone().view(-1)
        p_final = p1.clone().zero_()
        if mode == "interval":
            for i in range(p1.shape[0]):
                # calculate overlap between the probability distributions
                low1 = torch.sum(p1[:i])
                high1 = low1 + p1[i]
                low2 = torch.sum(p2[:i])
                high2 = low2 + p2[i]
                if low1 >= low2 and high2 > low1:
                    p_final[i] = torch.min(high1, high2) - low1
                    pass
                elif low2 >= low1 and high1 > low2:
                    p_final[i] = torch.min(high1, high2) - low2
                else:
                    p_final[i] = 0

            return p_final.clone().view(n_actions, n_actions)

    if ck_state == 0:
        p_joint = p_joint_ck0 + p_d_ck0 * torch.ger(pi_dec[0], pi_dec[1])
        return p_joint # always delegate
    elif ck_state == 1:
        p_joint = p_none * p_joint_ck1 + \
                  p_both * p_joint_ck2 + \
                  p_one * make_joint(p_joint_ck1, p_joint_ck2) + \
                  p_one * make_joint(p_joint_ck2, p_joint_ck1) + \
                  (p_one * p_d_ck1 * p_d_ck2
                   + p_one * p_d_ck2 * p_d_ck1
                   + p_both * p_d_ck2
                   + p_none * p_d_ck1) * torch.ger(pi_dec[0], pi_dec[1]) \
                  + p_one * p_d_ck1 * (1 - p_d_ck2) * torch.ger(pi_dec[0], p_marg_ag1_ck2) \
                  + p_one * (1 - p_d_ck2) * p_d_ck1 * torch.ger(p_marg_ag0_ck2, pi_dec[1]) \
                  + p_one * p_d_ck2 * (1 - p_d_ck1) * torch.ger(pi_dec[0], p_marg_ag1_ck1) \
                  + p_one * (1 - p_d_ck1) * p_d_ck2 * torch.ger(p_marg_ag0_ck1, pi_dec[1])
        return p_joint
    elif ck_state == 2:
        p_joint = p_none * p_joint_ck2 + \
                  p_both * p_joint_ck1 + \
                  p_one * make_joint(p_joint_ck2, p_joint_ck1) + \
                  p_one * make_joint(p_joint_ck1, p_joint_ck2) + \
                  (p_one * p_d_ck2 * p_d_ck1
                   + p_one * p_d_ck1 * p_d_ck2
                   + p_both * p_d_ck1
                   + p_none * p_d_ck2) * torch.ger(pi_dec[0], pi_dec[1]) \
                  + p_one * p_d_ck2 * (1 - p_d_ck1) * torch.ger(pi_dec[0], p_marg_ag1_ck1) \
                  + p_one * (1 - p_d_ck1) * p_d_ck2 * torch.ger(p_marg_ag0_ck1, pi_dec[1]) \
                  + p_one * p_d_ck1 * (1 - p_d_ck2) * torch.ger(pi_dec[0], p_marg_ag1_ck2) \
                  + p_one * (1 - p_d_ck2) * p_d_ck1 * torch.ger(p_marg_ag0_ck2, pi_dec[1])
        return p_joint
    pass


def get_policies(common_knowledge, observations, run, test, thetas_dec, theta_joint, p_ck_noise=0):
    if test:
        beta = 100
    else:
        beta = 1
    actions = []
    pi_dec = []

    # common_knowledge decides whether ck_state is informative
    if common_knowledge == 0:
        ck_state = 0
    else:
        ck_state = int(observations[0] + 1)


    if p_ck_noise == 0:

        pol_vals = theta_joint[ck_state, :].clone()

        # logits get masked out for independent learner and joint-action-learner
        # independent learner has a pair controller that always delegates
        if run == 'JAL':
            pol_vals[0] = -10 ** 10
        elif run == 'IAC':
            pol_vals[1:] = -10 ** 10

        # apply temperature to set testing
        pi_pc = F.softmax(pol_vals * beta, -1)

        # calcuate decentralised policies
        for i in range(n_agents):
            dec_state = int(observations[i])
            pi = F.softmax(thetas_dec[i][dec_state] * beta, -1)
            pi_dec.append(pi)

        return pi_pc, pi_dec

    else:

        pol_vals = theta_joint.clone()
        pi_pcs = []
        for i in range(n_states_joint):
            if run == 'JAL':
                pol_vals[i][0] = -10 ** 10
            elif run == 'IAC':
                pol_vals[i][1:] = -10 ** 10
            # apply temperature to set testing
            pi_pcs.append(F.softmax(pol_vals[i] * beta, -1))
            # calcuate decentralised policies

        for i in range(n_agents):
            dec_state = int(observations[i])
            pi = F.softmax(thetas_dec[i][dec_state] * beta, -1)
            pi_dec.append(pi)

        return pi_pcs, pi_dec, ck_state


def get_state(common_knowledge, obs_0, obs_1, matrix_id):
    receives_obs = [obs_0, obs_1]
    if common_knowledge == 1:
        observations = np.repeat(matrix_id, 2)
    else:
        observations = np.ones((n_agents)) * 2  #
        for ag in range(n_agents):
            if receives_obs[ag]:
                observations[ag] += matrix_id + 1
    return common_knowledge, observations, matrix_id


# Calculate the expected return: sum_{\tau} P(\tau | pi) R(\tau)
def expected_return(p_common, p_observation, thetas, run, test, p_ck_noise=0):
    thetas_dec = thetas["dec"]
    theta_joint = thetas["joint"]

    # Probability of CK
    p_common_val = [1 - p_common, p_common]
    # Probability of observation given no CK)
    p_obs_val = [1 - p_observation, p_observation]

    # Matrices are chosen 50 / 50
    p_matrix = [0.5, 0.5]
    # p_matrix =  [1.0, 0.0] # DEBUG!

    # Initialise expected return
    ret_val = 0
    for ck in [0, 1]:
        for matrix_id in [0, 1]:
            for obs_0 in [0, 1]:
                for obs_1 in [0, 1]:
                    p_state = p_common_val[ck] * p_obs_val[obs_0] * p_obs_val[obs_1] * p_matrix[matrix_id]
                    common_knowledge, observations, matrix_id = get_state(ck, obs_0, obs_1, matrix_id)

                    # Get final probabilities for joint actions
                    if p_ck_noise==0:
                        pi_pc, pi_dec = get_policies(common_knowledge, observations, run, test, thetas_dec, theta_joint)
                        p_joint_val = p_joint_all(pi_pc, pi_dec)
                    else:
                        pol_vals, pi_dec, ck_state = get_policies(common_knowledge, observations, run, test, thetas_dec, theta_joint, p_ck_noise)
                        p_joint_val = p_joint_all_noise_alt(pol_vals, pi_dec, p_ck_noise, ck_state)

                    # Expected return is just the elementwise product of rewards and action probabilities
                    expected_ret = (p_joint_val * payoff_values[matrix_id]).sum()

                    # Add return from given state
                    ret_val = ret_val + p_state * expected_ret
    return ret_val


def _proc(args):
    p_common, p_observation, run, p_ck_noise, t_max, n_trials = args

    results = []
    for nt in range(n_trials):
        print("Run: {} P_CK_NOISE: {} P_common: {} #Trial: {}".format(run, p_ck_noise, p_common, nt))
        results_log = np.zeros((t_max // (t_max // 100),))
        results_log_test = np.zeros((t_max // (t_max // 100),))

        thetas = {}
        thetas["dec"] = [init.normal_(torch.zeros(n_states_dec, n_actions, requires_grad=True), std=std_val) for i in
                         range(n_agents)]
        thetas["joint"] = init.normal_(torch.zeros(n_states_joint, n_actions ** 2 + 1, requires_grad=True),
                                       std=std_val)

        params = chain(*[_v if isinstance(_v, (list, tuple)) else [_v] for _v in thetas.values()])
        params = list(params)

        if use_cuda:
            for param in params:
                param = param.to("cuda")

        if optim == 'sgd':
            optimizer = SGD(params, lr=lr)
        else:
            optimizer = Adam(params, lr=lr)

        for i in range(t_max):
            if run in ['MACKRL',
                       'JAL',
                       'IAC']:
                loss = - expected_return(p_common, p_observation, thetas, run, False, p_ck_noise)
                r_s = -loss.data.numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % (t_max // 100) == 0:
                if run in ['MACKRL',
                           'JAL',
                           'IAC']:
                    r_test = expected_return(p_common, p_observation, thetas, run, True, p_ck_noise)
                results_log_test[i // (t_max // 100)] = r_test
                results_log[i // (t_max // 100)] = r_s
        results.append((results_log_test, results_log))

    return results

def main():

    use_mp = True
    if use_mp:
        pool = Pool(processes=2)

        # Well be appending results to these lists
        run_results = []
        for run in labels:
            noise_results = []
            for pnoise in p_ck_noise:
                print("Run: {} P_CK_NOISE: {}".format(run, pnoise))
                results = pool.map(_proc, [ (pc, p_observation, run, pnoise, t_max, n_trials) for pc in p_vec ])
                noise_results.append(results)
            run_results.append(noise_results)

        for p_common_id, p_common in enumerate(p_vec):
            all_res = []
            all_res_test = []
            for run_id, run in enumerate(labels):
                for pnoise_id, pnoise in enumerate(p_ck_noise):
                    try:
                        results = run_results[run_id][pnoise_id][p_common_id]
                    except Exception as e:
                        pass
                    all_res_test.append(np.stack([r[0] for r in results], axis=1))
                    all_res.append(np.stack([r[1] for r in results], axis=1))
            final_res.append([all_res_test, all_res])

        pool.close()
        pool.join()
    else:

        # Well be appending results to these lists
        run_results = []
        for run in labels:
            noise_results = []
            for pnoise in p_ck_noise:
                print("Run: {} P_CK_NOISE: {}".format(run, pnoise))
                results = [_proc((pc, p_observation, run, pnoise, t_max, n_trials)) for pc in p_vec ]
                noise_results.append(results)
            run_results.append(noise_results)

        for p_common_id, p_common in enumerate(p_vec):
            all_res = []
            all_res_test = []
            for run_id, run in enumerate(labels):
                for pnoise_id, pnoise in enumerate(p_ck_noise):
                    try:
                        results = run_results[run_id][pnoise_id][p_common_id]
                    except Exception as e:
                        pass
                    all_res_test.append(np.stack([r[0] for r in results], axis=1))
                    all_res.append(np.stack([r[1] for r in results], axis=1))
            final_res.append([all_res_test, all_res])

    import pickle
    import uuid
    import os
    res_dict = {}
    res_dict["final_res"] = final_res
    res_dict["labels"] = labels
    res_dict["p_ck_noise"] = p_ck_noise
    res_dict["p_vec"] = p_vec
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "pickles")):
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "pickles"))
    pickle.dump(res_dict, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "pickles",
                                             "final_res_{}.p".format(uuid.uuid4().hex[:4])), "wb"))
    plt.figure(figsize=(5, 5))

    color = ['b', 'r','g', 'c', 'm', 'y', 'k','b', 'r','g', 'c', 'm', 'y', 'k']
    titles = ['Test', 'Train Performance']
    for pl in [0,1]:
        ax = plt.subplot(1, 1, 1)
        for i in range(len(labels)):
            for pck, pcknoise in enumerate(p_ck_noise):
                mean_vals = []
                min_vals = []
                max_vals = []
                for j, p in enumerate( p_vec ):
                    vals = final_res[j][pl]
                    this_mean = np.mean( vals[i*len(p_ck_noise) + pck], 1)[-1]
                    std = np.std(vals[i], 1)[-1]/0.5
                    low = this_mean-std / (n_trials)**0.5
                    high = this_mean + std / (n_trials)**0.5
                    mean_vals.append( this_mean )
                    min_vals.append( low )
                    max_vals.append( high )
                plt.plot(p_vec,
                         mean_vals,
                         color[(i*len(p_ck_noise) + pck) % len(color)],
                         label = "{} p_ck_noise: {}".format(labels[i], pcknoise))
                plt.fill_between(p_vec,
                                 min_vals,
                                 max_vals,
                                 facecolor=color[i],
                                 alpha=0.3)

        plt.xlabel('P(common knowledge)')
        plt.ylabel('Expected Return')
        plt.ylim([0.0, 1.01])
        plt.xlim([-0.01, 1.01])
        ax.set_facecolor((1.0, 1.0, 1.0))
        ax.grid(color='k', linestyle='-', linewidth=1)
        ax.set_title(titles[pl])
        plt.legend()
        plt.xticks([0, 0.5, 1])
        plt.yticks([0.5, 0.75, 1])
        plt.savefig("MACKRL {}.pdf".format(titles[pl]))
        plt.show(block=False)

if __name__ == "__main__":
    freeze_support()
    main()