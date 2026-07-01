import torch
import torch.nn as nn


class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        return next(iter(self.parameters())).device


class LowdimMaskGenerator(ModuleAttrMixin):
    def __init__(
        self,
        action_dim,
        obs_dim,
        max_n_obs_steps=2,
        fix_obs_steps=True,
        action_visible=False,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.obs_dim = int(obs_dim)
        self.max_n_obs_steps = int(max_n_obs_steps)
        self.fix_obs_steps = bool(fix_obs_steps)
        self.action_visible = bool(action_visible)

    @torch.no_grad()
    def forward(self, shape, seed=None):
        device = self.device
        b, t, d = shape
        assert d == (self.action_dim + self.obs_dim)

        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)

        dim_mask = torch.zeros(size=shape, dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[..., : self.action_dim] = True
        is_obs_dim = ~is_action_dim

        if self.fix_obs_steps:
            obs_steps = torch.full((b,), fill_value=self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(
                low=1,
                high=self.max_n_obs_steps + 1,
                size=(b,),
                generator=rng,
                device=device,
            )

        steps = torch.arange(0, t, device=device).reshape(1, t).expand(b, t)
        obs_mask = (steps.T < obs_steps).T.reshape(b, t, 1).expand(b, t, d)
        obs_mask = obs_mask & is_obs_dim

        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1, torch.tensor(0, dtype=obs_steps.dtype, device=device)
            )
            action_mask = (steps.T < action_steps).T.reshape(b, t, 1).expand(b, t, d)
            action_mask = action_mask & is_action_dim
            return obs_mask | action_mask

        return obs_mask
