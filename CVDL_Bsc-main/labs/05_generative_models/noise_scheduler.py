from abc import ABC, abstractmethod

import torch


# ---------------------------------------------------------------------------------
# helper Class
# ---------------------------------------------------------------------------------
class NoiseScheduler(ABC):
    def __init__(self, num_timesteps: int, device):
        self.num_timesteps = num_timesteps
        self.device = device

        # vars for forward & sample
        self.b = self.calc_betas().to(device)
        self.a = (1.0 - self.b).to(device)
        self.a_cp = torch.cumprod(self.a, dim=0).to(device)
        self.sqrt_a = torch.sqrt(self.a).to(device)
        self.sqrt_a_cp = torch.sqrt(self.a_cp).to(device)
        self.sqrt_one_min_a_cp = torch.sqrt(1 - self.a_cp).to(device)

    @abstractmethod
    def calc_betas(self) -> torch.Tensor:
        pass

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor | int):
        sqrt_alpha_cum_prod = self.sqrt_a_cp[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_min_a_cp[t].view(-1, 1, 1, 1)

        return sqrt_alpha_cum_prod * x0 + sqrt_one_minus_alpha_cum_prod * noise

    def sample_timesteps(self, n: int) -> torch.Tensor:
        return torch.randint(0, self.num_timesteps, (n,), device=self.device)

    def sample_prev(self, xt: torch.Tensor, pred: torch.Tensor, t: torch.Tensor | int):
        x0 = xt - (self.sqrt_one_min_a_cp[t] * pred) / self.sqrt_a_cp[t]
        x0 = torch.clamp(x0, -1.0, 1.0)

        mean = (xt - self.b[t] * pred / self.sqrt_one_min_a_cp[t]) / self.sqrt_a[t]

        if t == 0:
            return mean, x0
        else:
            variance = (1.0 - self.a_cp[t - 1]) / (1.0 - self.a_cp[t])
            variance = variance * self.b[t]
            sigma = variance**0.5
            z = torch.randn(xt.shape, device=self.device)

            return mean + sigma * z, x0


# ---------------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------------


class LinearNoisescheduler(NoiseScheduler):
    def __init__(self, beta_start: float, beta_end: float, num_timesteps: int, device):
        self.beta_start = beta_start
        self.beta_end = beta_end
        # initialize super
        super().__init__(num_timesteps, device)

    def calc_betas(self) -> torch.Tensor:
        return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)


# TODO add more schedulers here
