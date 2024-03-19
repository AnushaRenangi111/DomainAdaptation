import torch
import torch.nn as nn
from torch.autograd import Function
import uuid

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_val, None

class GradientReversal(nn.Module):
    def __init__(self, hp_lambda):
        super(GradientReversal, self).__init__()
        self.hp_lambda = nn.Parameter(torch.tensor(hp_lambda, dtype=torch.float32))

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.hp_lambda)

    def set_hp_lambda(self, hp_lambda):
        self.hp_lambda.data = torch.tensor(hp_lambda)

    def increment_hp_lambda_by(self, increment):
        self.hp_lambda.data += increment

    def get_hp_lambda(self):
        return self.hp_lambda.item()
