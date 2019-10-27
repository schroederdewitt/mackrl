REGISTRY = {}

from .basic import BasicLearner
REGISTRY["basic"] = BasicLearner

from .mackrel import MACKRELLearner
REGISTRY["mackrel"] = MACKRELLearner

from .centralV import CentralVLearner
REGISTRY["centralV"] = CentralVLearner