import torch
import torch.nn as nn
import torch.nn.functional as F

class LLSA(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, window_size=5, hidden_size=10, change_point_threshold=0.1):
        super(LLSA, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.change_point_threshold = nn.Parameter(torch.tensor(change_point_threshold))
        self.linear = nn.Linear(window_size, hidden_size)
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
