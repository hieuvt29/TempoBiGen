import torch
from torch.distributions import MixtureSameFamily as TorchMixtureSameFamily


class MixtureSameFamily(TorchMixtureSameFamily):
    def log_cdf(self, x):
        x = self._pad(x)
        log_cdf_x = self.component_distribution.log_cdf(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_cdf_x + mix_logits, dim=-1)

    def log_survival_function(self, x):
        x = self._pad(x)
        log_sf_x = self.component_distribution.log_survival_function(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_sf_x + mix_logits, dim=-1)


import torch
import torch.distributions as dist

class MixtureWeibull(dist.Distribution):
    def __init__(self, weights, shapes, scales):
        """
        Initializes a Mixture of Weibull distributions.

        Args:
            weights (torch.Tensor): Mixing probabilities for each component.
            shapes (torch.Tensor): Shape parameters for each Weibull distribution.
            scales (torch.Tensor): Scale parameters for each Weibull distribution.
        """
        self.components = dist.Weibull(shapes, scales)
        self.mixture_dist = dist.Categorical(weights)
        super().__init__(self.components.batch_shape, self.components.event_shape)

    def sample(self, sample_shape=torch.Size()):
         # Sample component indices
        component_indices = self.mixture_dist.sample(sample_shape)
        # Sample from selected components
        samples = self.components[component_indices].sample()
        return samples

    def log_prob(self, x):
        """
        Computes the log probability of the mixture model.

        Args:
            x (torch.Tensor): Values at which to evaluate the log probability.

        Returns:
            torch.Tensor: Log probability of the mixture model.
        """
        # Log probability of each component
        component_log_probs = self.components.log_prob(x.unsqueeze(-1))
        # Weighted log probabilities
        weighted_log_probs = component_log_probs + torch.log(self.mixture_dist.probs)
        # Log sum exp trick for numerical stability
        log_prob = torch.logsumexp(weighted_log_probs, dim=-1)
        return log_prob