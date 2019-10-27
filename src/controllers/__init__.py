REGISTRY = {}

from .basic_agent import BasicAgentController

REGISTRY["basic_ac"] = BasicAgentController

from .mackrl_agents import MACKRLMultiagentController
REGISTRY["mackrl_mac"] = MACKRLMultiagentController
