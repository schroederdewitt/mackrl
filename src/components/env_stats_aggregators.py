REGISTRY = {}

from envs.starcraft2 import StatsAggregator as SC2StatsAggregator
REGISTRY["sc2"] = SC2StatsAggregator
