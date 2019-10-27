REGISTRY = {}

from .basic import DQN, RNN, FCEncoder
REGISTRY["DQN"] = DQN
REGISTRY["RNN"] = RNN
REGISTRY["fc_encoder"] = FCEncoder



from .mackrl import MACKRLCritic
REGISTRY["mackrl_critic"] = MACKRLCritic
from .mackrl import MACKRLAgent
REGISTRY["mackrl_agent"] = MACKRLAgent
from .mackrl import MACKRLRecurrentAgentLevel1, MACKRLRecurrentAgentLevel2, MACKRLRecurrentAgentLevel3
REGISTRY["mackrl_recurrent_agent_level1"] = MACKRLRecurrentAgentLevel1
REGISTRY["mackrl_recurrent_agent_level2"] = MACKRLRecurrentAgentLevel2
REGISTRY["mackrl_recurrent_agent_level3"] = MACKRLRecurrentAgentLevel3
from .mackrl import MACKRLNonRecurrentAgentLevel1, MACKRLNonRecurrentAgentLevel2, MACKRLNonRecurrentAgentLevel3
REGISTRY["mackrl_nonrecurrent_agent_level1"] = MACKRLNonRecurrentAgentLevel1
REGISTRY["mackrl_nonrecurrent_agent_level2"] = MACKRLNonRecurrentAgentLevel2
REGISTRY["mackrl_nonrecurrent_agent_level3"] = MACKRLNonRecurrentAgentLevel3

from .centralV import CentralVCritic
REGISTRY["centralV_critic"] = CentralVCritic




