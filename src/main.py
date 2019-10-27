import numpy as np
import os
from os.path import dirname, abspath
import pymongo
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th

from components.transforms import _merge_dicts
from run import run
from utils.logging import get_logger

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("fastmarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")

def setup_file_observer():
    file_obs_path = os.path.join(results_path, "sacred")
    logger.info("FileStorageObserver path: {}".format(file_obs_path))
    logger.info("Using the FileStorageObserver in results/sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    pass

@ex.main
def my_main(_run, _config, _log, env_args):
    global mongo_client

    # Setting the random seed throughout the modules
    np.random.seed(_config["seed"])
    th.manual_seed(_config["seed"])
    env_args['seed'] = _config["seed"]

    # run the framework
    run(_run, _config, _log, mongo_client)

    # force exit
    os._exit()

if __name__ == '__main__':

    ### Execute functions that modify the directory tree
    from copy import deepcopy
    from distutils.dir_util import copy_tree
    import os
    if os.path.exists("/fastmarl/3rdparty") and os.path.exists("/fastmarl/src"): # de facto only happens if called in docker file


        fromDirectory = "/fastmarl/src/envs/starcraft2/maps"

        toDirectory = "/fastmarl/3rdparty/StarCraftII__3.16.1/Maps/Melee"
        if os.path.exists(toDirectory):
            print("COPYING... {} to {}".format(fromDirectory, toDirectory))
            copy_tree(fromDirectory, toDirectory)

        toDirectory = "/fastmarl/3rdparty/StarCraftII__4.1.2/Maps/Melee"
        if os.path.exists(toDirectory):
            print("COPYING... {} to {}".format(fromDirectory, toDirectory))
            copy_tree(fromDirectory, toDirectory)
    ### End execute functions that modify the directory tree

    params = deepcopy(sys.argv)

    defaults = []
    config_dic = {}

    # manually parse for experiment tags
    del_indices = []
    exp_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == "--exp_name":
            del_indices.append(_i)
            exp_name = _v.split("=")[1]
            break

    # load experiment config (if there is such as thing)
    exp_dic = None
    if exp_name is not None:
        from config.experiments import REGISTRY as exp_REGISTRY
        assert exp_name in exp_REGISTRY, "Unknown experiment name: {}".format(exp_name)
        exp_dic = exp_REGISTRY[exp_name](None, logger)
        if "defaults" in exp_dic:
            defaults.extend(exp_dic["defaults"].split(" "))
            del exp_dic["defaults"]
        config_dic = deepcopy(exp_dic)

    # check for defaults in command line parameters
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == "--default_cfgs":
            del_indices.append(_i)
            defaults.extend(_v.split("=")[1].split(" "))
            break

    # load default configs in order
    for _d in defaults:
        from config.defaults import REGISTRY as def_REGISTRY
        def_dic = def_REGISTRY[_d](config_dic, logger)
        config_dic = _merge_dicts(config_dic, def_dic)

    #  finally merge with experiment config
    if exp_name is not None:
        config_dic = _merge_dicts(config_dic, exp_dic)

    # add results path to config
    config_dic["local_results_path"] = results_path

    # now add all the config to sacred
    ex.add_config(config_dic)

    # delete indices that contain custom experiment tags
    for _i in sorted(del_indices, reverse=True):
        del params[_i]

    if config_dic.get("observe_file", True):
        setup_file_observer()
    ex.run_commandline(params)

