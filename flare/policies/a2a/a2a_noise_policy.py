"""
A2A-Noise: Action-to-Action Flow Matching with History Noise

A variant of A2A that adds Gaussian noise to proprioceptive states before encoding.
This introduces stochasticity into the flow source, enabling multimodal behavior —
the same observation can lead to different action trajectories across rollouts.

Architecture (same as A2A, except noise injection):
    States [s_{t-n+1}, ..., s_t] + N(0, noise_std) --encode--> state_latents (x_0)
    Visual Obs --encode--> obs_latents (condition)
    Flow Matching: x_0 --flow(condition)--> x_1 (action_latents)
    x_1 --decode--> Future Actions
"""

import torch
import logging

from flare.factory import registry
from flare.policies.a2a.a2a_policy import A2APolicy

logger = logging.getLogger(__name__)


@registry.register_policy("a2a_noise")
class A2ANoisePolicy(A2APolicy):
    def __init__(self, config, stats):
        super().__init__(config, stats)
        self.history_noise_std = config.policy.a2a.history_noise_std
        logger.info(f"A2A-Noise: history_noise_std = {self.history_noise_std}")

    def _encode_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        states = batch[self.config.task.state_key][:, : self.obs_horizon]
        # Add Gaussian noise to raw states before encoding (both train & inference)
        # Noise at inference is intentional — it enables multimodal behavior
        if self.history_noise_std > 0:
            noise = torch.randn_like(states) * self.history_noise_std
            states = states + noise
        return self.state_encoder(states.flatten(start_dim=1))