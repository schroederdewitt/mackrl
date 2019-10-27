import torch as th
from torch import nn

from components.transforms import _to_batch, _from_batch

class EntropyRegularisationLoss(nn.Module):

    def __init__(self):
        super(EntropyRegularisationLoss, self).__init__()
        pass

    def forward(self, policies, tformat):

        _policies, policies_params, policies_tformat = _to_batch(policies, tformat)

        # need batch scalar product as in COMA!!!
        entropy = th.bmm(th.log(_policies).unsqueeze(1),
                          _policies.unsqueeze(2)).squeeze(2)

        ret = _from_batch(entropy, policies_params, policies_tformat)
        return ret