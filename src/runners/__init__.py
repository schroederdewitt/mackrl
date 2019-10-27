REGISTRY = {}

from .nstep_runner import NStepRunner
REGISTRY["nstep"] = NStepRunner

from .mackrl_runner import MACKRLRunner
REGISTRY["mackrl"] = MACKRLRunner