def get_cfg(existing_cfg, _log):
    """
    generates
    """
    _sanity_check(existing_cfg, _log)
    import ntpath, os, ruamel.yaml as yaml
    with open(os.path.join(os.path.dirname(__file__), "{}.yml".format(ntpath.basename(__file__).split(".")[0])),
              'r') as stream:
        try:
            ret = yaml.load(stream, Loader=yaml.Loader)
        except yaml.YAMLError as exc:
            assert "Default config yaml for '{}' not found!".format(os.path.splitext(__file__)[0])

    if ret["coma_critic_use_sampling"] and "coma_critic_sample_size" not in ret and hasattr(ret, "batch_size_run"):
        ret["coma_critic_sample_size"] = ret["batch_size_run"] * 50
    return ret

def _sanity_check(existing_cfg, _log):
    """
    """
    return