import torch
import numpy as np
from torch.nn.functional import mse_loss
from tqdm import tqdm


class Diffusion:
    def __init__(self, device, schedule_type="linear", T=1000):
        self.device = device
        self.T = T
        if schedule_type == "linear":
            self.betas = np.linspace(1e-4, 0.02, T)
        elif schedule_type == "cosine":
            self.betas = self.cosine_beta_schedule()
        else:
            raise ValueError("Invalid schedule type")
        self.alphas = 1. - self.betas
        self.alpha_bars = np.cumprod(self.alphas, axis=0)
        self.alpha_bars_prev = np.append(1.0, self.alpha_bars[:-1])
        self.alpha_bars_next = np.append(self.alpha_bars[1:], 0.0)

        self.sqrt_alpha_bars = np.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = np.sqrt(1. - self.alpha_bars)
        self.log_one_minus_alpha_bars = np.log(1. - self.alpha_bars)
        self.sqrt_recip_alphas_bars = np.sqrt(1. / self.alpha_bars)
        self.sqrt_recipm1_alphas_bars = np.sqrt(1. / self.alpha_bars - 1.)

        self.posterior_variance = self.betas * (1. - self.alpha_bars_prev) / (1. - self.alpha_bars)
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = self.betas * np.sqrt(self.alpha_bars_prev) / (1. - self.alpha_bars)
        self.posterior_mean_coef2 = (1. - self.alpha_bars_prev) * np.sqrt(self.alphas) / (1. - self.alpha_bars)

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.T, (batch_size,), device=self.device).long()

    def cosine_beta_schedule(self, s=0.008):
        betas = []
        for i in range(self.T):
            t1 = i / self.T
            t2 = (i + 1) / self.T
            a1 = np.cos((t1 + s) / (1 + s) * np.pi * 0.5) ** 2
            a2 = np.cos((t2 + s) / (1 + s) * np.pi * 0.5) ** 2
            betas.append(min(1 - a2 / a1, 0.999))
        return np.array(betas)

    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        res = torch.from_numpy(a).to(self.device)[t].float()
        while len(res.shape) < len(x_shape):
            res = res.unsqueeze(-1)
        return res

    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bars_t = self._extract(self.sqrt_alpha_bars, t, x_start.shape)
        sqrt_one_minus_alpha_bars_t = self._extract(self.sqrt_one_minus_alpha_bars, t, x_start.shape)

        return sqrt_alpha_bars_t * x_start + sqrt_one_minus_alpha_bars_t * noise

    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alpha_bars, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alpha_bars, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alpha_bars, t, x_start.shape)
        return mean, variance, log_variance

    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_bars, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_bars, t, x_t.shape) * noise
        )

    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, x_t, t, clip_denoised=True, label=None):
        # predict noise using model
        pred_noise = model(x_t, t, label)
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = \
                    self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    # denoise_step: sample x_{t-1} from x_t and pred_noise
    @torch.no_grad()
    def p_sample(self, model, x_t, t: torch.Tensor, clip_denoised=True, label=None):
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised, label=label)
        noise = torch.randn_like(x_t).to(self.device)
        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # compute x_{t-1}
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred

    def p_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True, progress=False, label=None):
        if noise is None:
            input_ = torch.randn(shape, device=self.device).to(self.device)
        else:
            input_ = noise
        indices = list(range(self.T))[::-1]
        if progress:
            indices = tqdm(indices, desc='sampling loop time step', total=self.T)

        with torch.no_grad():
            for i in indices:
                t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
                out = self.p_sample(model, input_,t, clip_denoised, label)
                yield out
                input_ = out

    def p_sample_loop(self, model, shape, noise=None, progress=False, clip_denoised=True, label=None):
        final = None
        for sample in self.p_sample_loop_progressive(model, shape, noise, clip_denoised, progress, label):
            final = sample
        assert final is not None
        return final

    def sample(self, model, size=32, batch_size=1, channels=3, label=None):
        return self.p_sample_loop(model, (batch_size, channels, size, size, size), progress=True, label=label)

    def train_losses(self, model, x_start, t, label=None):
        noise = torch.randn_like(x_start).to(self.device)
        x_t = self.q_sample(x_start, t, noise)
        pred_noise = model(x_t, t, label)
        loss = mse_loss(pred_noise, noise)
        return loss
