def get_cfg(existing_cfg, _log):
    """

    """
    _sanity_check(existing_cfg, _log)
    import ntpath, os, ruamel.yaml as yaml
    with open(os.path.join(os.path.dirname(__file__), "{}.yml".format(ntpath.basename(__file__).split(".")[0])), 'r') as stream:
        try:
            ret = yaml.load(stream, Loader=yaml.Loader)
        except yaml.YAMLError as exc:
            assert False, "Default config yaml for '{}' not found!".format(os.path.splitext(__file__)[0])

    if not "batch_size_run" in ret:
        ret["batch_size_run"] = ret["batch_size"]

    if not "runner_test_batch_size" in ret:
        ret["runner_test_batch_size"] = ret["batch_size_run"]

    if not "name" in ret:
        ret["name"] = ntpath.basename(__file__).split(".")[0]

    return ret

def _sanity_check(existing_cfg, _log):
    """
    """
    return