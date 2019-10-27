import datetime
from functools import partial
from math import ceil
import numpy as np
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.dict2namedtuple import convert
from utils.logging import get_logger, append_scalar, log_stats, HDFLogger
from utils.timehelper import time_left, time_str

from components.replay_buffer import ReplayBuffer
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY

def run(_run, _config, _log, pymongo_client):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    # convert _config dict to GenericDict objects (which cannot be overwritten later)
    args = convert(_config)
    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    import os
    _log.info("OS ENVIRON KEYS: {}".format(os.environ))

    if _config.get("debug_mode", None) is not None:
        _log.warning("ATTENTION DEBUG MODE: {}".format(_config["debug_mode"]))

    # ----- configure logging
    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if args.use_tensorboard:
        import tensorboard
        if tensorboard:
            from tensorboard_logger import configure, log_value
        import os
        from os.path import dirname, abspath
        file_tb_path = os.path.join(dirname(dirname(abspath(__file__))), "tb_logs")
        configure(os.path.join(file_tb_path, "{}").format(unique_token))

    # configure trajectory logger


    # set up logging object to be passed on from now on
    logging_struct = SN(py_logger=_log,
                        sacred_log_scalar_fn=partial(append_scalar, run=_run))
    if args.use_tensorboard:
        logging_struct.tensorboard_log_scalar_fn=log_value

    if hasattr(args, "use_hdf_logger") and args.use_hdf_logger:
        logging_struct.hdf_logger = HDFLogger(path=args.local_results_path, name=args.name, logging_struct=logging_struct)

    # ----- execute runners
    # run framework in run_mode selected
    if args.run_mode in ["parallel_subproc"]:
        run_parallel(args=args, _run=_run, _logging_struct=logging_struct, unique_token=unique_token)
    else:
        run_sequential(args=args, _run=_run, _logging_struct=logging_struct, unique_token=unique_token)

    #Clean up after finishing
    print("Exiting Main")

    if pymongo_client is not None: #args.use_mongodb:
        print("Attempting to close mongodb client")
        pymongo_client.close()
        print("Mongodb client closed")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)

def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    assert (config["run_mode"] in ["parallel_subproc"] and config["use_replay_buffer"]) or (not config["run_mode"] in ["parallel_subproc"]),  \
        "need to use replay buffer if running in parallel mode!"

    assert not (not config["use_replay_buffer"] and (config["batch_size_run"]!=config["batch_size"]) ) , "if not using replay buffer, require batch_size and batch_size_run to be the same."

    if config["learner"] == "coma":
       assert (config["run_mode"] in ["parallel_subproc"]  and config["batch_size_run"]==config["batch_size"]) or \
       (not config["run_mode"] in ["parallel_subproc"]  and not config["use_replay_buffer"]), \
           "cannot use replay buffer for coma, unless in parallel mode, when it needs to have exactly have size batch_size."

    return config

def run_parallel(args, _logging_struct, _run, unique_token):
    """
     this run mode runs runner and learner in parallel subprocesses and uses a specially protected shared replay buffer in between
     TODO: under construction!
     """
    from torch import multiprocessing as mp
    def runner_process(self, args):
        """
        TODO: add inter-process communication
        :return:
        """
        runner_train_obj = r_REGISTRY[args.runner](args=args)
        while True:
            runner_train_obj.run()
        pass

    def learner_process():
        """
        TODO: add inter-process communication
        :return:
        """
        runner_test_obj = r_REGISTRY[args.runner](multiagent_controller=runner_train_obj.multiagent_controller,
                                                  args=args,
                                                  test_mode=True)
        while True:
            runner_test_obj.run()
        pass

    runner_obj = NStepRunner(args=args)
    runner_obj.share_memory()

    test_runner_obj = NStepRunner(multiagent_controller=runner_obj.multiagent_controller,
                                  args=args)
    test_runner_obj.share_memory()

    # TODO: Create a data scheme registry (possibly with yaml files or sth)
    replay_buffer = ReplayBuffer(scheme, args.buffer_size, is_cuda=True, is_shared_mem=True)

    mp.set_start_method('forkserver')
    runner_proc = mp.Process(target=runner_process, args=(runner_obj, replay_buffer,))
    learner_proc = mp.Process(target=learner_process, args=(learner_obj, replay_buffer,))
    runner_proc.start()
    learner_proc.start()

    ###
    #  Main process keeps train / runner ratios roughly in sync, caters for logging and termination
    ###
    while True:
        break

    runner_proc.join()
    learner_proc.join()

def run_sequential(args, _logging_struct, _run, unique_token):

    # set up train runner
    runner_obj = r_REGISTRY[args.runner](args=args,
                                         logging_struct=_logging_struct)

    # create the learner
    learner_obj = le_REGISTRY[args.learner](multiagent_controller=runner_obj.multiagent_controller,
                                            logging_struct=_logging_struct,
                                            args=args)

    # create non-agent models required by learner
    learner_obj.create_models(runner_obj.data_scheme)

    # set up replay buffer (if enabled)
    buffer = None
    if args.use_replay_buffer:
        buffer = ReplayBuffer(data_scheme=runner_obj.data_scheme,
                              n_bs=args.buffer_size,
                              n_t=runner_obj.env_episode_limit + 1,
                              n_agents=runner_obj.n_agents,
                              batch_size=args.batch_size,
                              is_cuda=args.use_cuda,
                              is_shared_mem=not args.use_cuda,
                              logging_struct=_logging_struct)

    #---- start training

    T = 0
    episode = 0
    last_test_T = 0
    model_save_time = 0
    # start_time = time.time()

    _logging_struct.py_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner_obj.T_env <= args.t_max:

        # Run for a whole episode at a time
        episode_rollout = runner_obj.run(test_mode=False)

        episode_sample = None
        if args.use_replay_buffer:
            buffer.put(episode_rollout)

            if buffer.can_sample(args.batch_size):
                episode_sample = buffer.sample(args.batch_size, seq_len=0)
        else:
            episode_sample = episode_rollout

        if episode_sample is not None:
            if hasattr(args, "save_episode_samples") and args.save_episode_samples:
                assert hasattr(args, "use_hdf_logger") and args.use_hdf_logger, "use_hdf_logger needs to be enabled if episode samples are to be stored!"
                _logging_struct.hdf_logger.log("", episode_sample, runner_obj.T_env)
            learner_obj.train(episode_sample, T_env=runner_obj.T_env)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner_obj.batch_size)
        if ( runner_obj.T_env - last_test_T) / args.test_interval > 1.0:

            _logging_struct.py_logger.info("T_env: {}".format(runner_obj.T_env))
            runner_obj.log() # log runner statistics derived from training runs

            last_test_T =  runner_obj.T_env
            if args.obs_noise: # bad hack
                for ons in args.obs_noise_std:
                    _logging_struct.py_logger.info("Testing for noise std {}".format(ons))
                    for _ in range(n_test_runs):
                        runner_obj.run(test_mode=True,
                                       obs_noise_std=ons)
                    runner_obj.log(obs_noise_std=ons)  # log runner statistics derived from test runs
            else:
                for _ in range(n_test_runs):
                    runner_obj.run(test_mode=True)
                runner_obj.log()  # log runner statistics derived from test runs
            learner_obj.log()

        # save model once in a while
        if args.save_model and (runner_obj.T_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner_obj.T_env
            _logging_struct.py_logger.info("Saving models")

            save_path = os.path.join(args.local_results_path, "models") #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)

            # learner obj will save all agent and further models used
            learner_obj.save_models(path=save_path, token=unique_token, T=runner_obj.T_env)

        episode += 1

    _logging_struct.py_logger.info("Finished Training")
    # end of parallel / anti-parallel branch
