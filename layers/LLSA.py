import torch
import torch.nn as nn
import torch.nn.functional as F

class LLSA(nn.Module):
    def __init__(self, window_size=5, hidden_size=10):
        super(LLSA, self).__init__()
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(window_size, hidden_size)
        self.change_point_threshold = nn.Parameter(torch.tensor(0.1))

    def forward(self, series):
        # (batch_size, time_steps)
        batch_size, time_steps = series.shape
        change_points = []
        losses = []

        for i in range(time_steps - self.window_size):
            local_series = series[:, i:i+self.window_size]
            target = series[:, i+1:i+self.window_size+1]
            transformed = self.linear(local_series)
            loss = F.mse_loss(transformed, target)
            losses.append(loss)

            if i > 0:
                diff = torch.abs(losses[i] - losses[i-1])
                if diff > self.change_point_threshold:
                    change_points.append((i, diff.item()))

        return change_points, torch.stack(losses)