from functools import partial


def env_fn(env, **kwargs):
    return env(**kwargs)

REGISTRY = {}

from .starcraft2 import StarCraft2Env
REGISTRY["sc2"] = partial(env_fn,
                          env=StarCraft2Env)

