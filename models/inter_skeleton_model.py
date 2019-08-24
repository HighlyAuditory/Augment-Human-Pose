import torch
from torch.autograd import Variable
from .base_model import BaseModel
from . import networks

import torch.nn as nn

class InterSkeleton_Model(BaseModel):
    def name(self):
        return 'InterSkeleton_Model'

        # input shape (b, 7, 3)

    def __init__(self, opt):
        super(InterSkeleton_Model, self).__init__()
        BaseModel.initialize(self, opt)
        self.alpha = Variable(torch.randn((1,7,1), device="cuda"), requires_grad=True)

    def forward(self, input1, input2):
        self.alpha_clipped = torch.clamp(self.alpha, 0, 1)
        out = self.alpha_clipped * input1 + (1 - self.alpha_clipped) * input2
        return out