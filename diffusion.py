import torch
import numpy as np
from torch.nn.functional import mse_loss
from tqdm import tqdm


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))


def discretized_gaussian_log_likelihood(x, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image, using the eq(13) of the paper DDPM.
    """
    assert x.shape == means.shape == log_scales.shape

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999,
            log_one_minus_cdf_min,
            torch.log(cdf_delta.clamp(min=1e-12))
        )
    )
    return log_probs


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
        # a_ = torch.from_numpy(a).to(self.device)
        # out = a_.gather(0, t).float()
        # out = out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))
        # return out
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

    def predict_noise_from_start(self, x_t, t, x_start):
        return (
            self._extract(self.sqrt_recip_alphas_bars, t, x_t.shape) * x_t - x_start
        ) / self._extract(self.sqrt_recipm1_alphas_bars, t, x_t.shape)

    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, x_t, t, clip_denoised=True, label=None):
        B, C = x_t.shape[:2]
        # predict noise using model
        model_output = model(x_t, t, label)

        if model_output.shape[:2] == (B, C * 2):
            pred_noise, pred_variance_v = torch.split(model_output, C, dim=1)
            # compute predicted variance by eq(15) in the paper
            mix_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
            max_log_variance = self._extract(np.log(self.betas), t, x_t.shape)
            # The predicted value is in [-1, 1], convert it to [0, 1]
            frac = (pred_variance_v + 1.) / 2.
            model_log_variance = frac * max_log_variance + (1. - frac) * mix_log_variance
            model_variance = model_log_variance.exp()
        else:
            pred_noise = model_output
            model_variance = self._extract(self.posterior_variance, t, x_t.shape)
            model_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, _, _ = self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, model_variance, model_log_variance, x_recon

    # denoise_step: sample x_{t-1} from x_t and pred_noise
    @torch.no_grad()
    def p_sample(self, model, x_t, t: torch.Tensor, clip_denoised=True, label=None):
        # predict mean and variance
        model_mean, _, model_log_variance, _ = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised, label=label)
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

    def sample(self, model, size=32, batch_size=1, channels=4, label=None):
        return self.p_sample_loop(model, (batch_size, channels, size, size, size), progress=True, label=label)

    @torch.no_grad()
    def ddim_p_sample(self, model, x_t, t: torch.Tensor, clip_denoised=True, label=None, eta=0.0):
        # predict mean and variance
        _, _, _, x_start = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, label=label
        )
        noise = self.predict_noise_from_start(x_t, t, x_start)
        alpha_bar = self._extract(self.alpha_bars, t, x_t.shape)
        alpha_bar_prev = self._extract(self.alpha_bars_prev, t, x_t.shape)
        sigma = eta * torch.sqrt((1. - alpha_bar_prev) / (1. - alpha_bar)) * torch.sqrt(1. - alpha_bar / alpha_bar_prev)
        noise1 = torch.randn_like(x_t)
        mean_pred = x_start * torch.sqrt(alpha_bar_prev) + noise * torch.sqrt(1. - alpha_bar_prev - sigma ** 2)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        pred = mean_pred + nonzero_mask * sigma * noise1
        return pred

    def ddim_p_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True, progress=False, label=None, eta=0.0):
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
                out = self.ddim_p_sample(model, input_, t, clip_denoised, label, eta)
                yield out
                input_ = out

    def ddim_p_sample_loop(self, model, shape, noise=None, progress=False, clip_denoised=True, label=None, eta=0.0):
        final = None
        for sample in self.ddim_p_sample_loop_progressive(model, shape, noise, clip_denoised, progress, label, eta):
            final = sample
        assert final is not None
        return final

    def ddim_sample(self, model, size=32, batch_size=1, channels=4, label=None, eta=0.0):
        return self.ddim_p_sample_loop(model, (batch_size, channels, size, size, size), progress=True, label=label, eta=eta)

    def _vb_terms_bpd(
        self, model, x_start, x_t, t: torch.Tensor, clip_denoised=True, label=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a shape [N] tensor of NLLs or KLs.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        model_mean, _, model_log_variance, _ = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, label=label
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, model_mean, model_log_variance
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=model_mean, log_scales=0.5 * model_log_variance
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return output

    def train_losses(self, model, x_start, t: torch.Tensor, label=None):
        # generate random noise
        noise = torch.randn_like(x_start)
        # get x_t
        x_t = self.q_sample(x_start, t, noise=noise)

        # predict
        model_output = model(x_t, t, label)
        B, C = x_t.shape[:2]
        if model_output.shape[:2] == (B, C * 2):
            pred_noise, pred_variance_v = torch.split(model_output, C, dim=1)
            frozen_out = torch.cat([pred_noise.detach(), pred_variance_v], dim=1)
            vb = self._vb_terms_bpd(lambda *args, r=frozen_out: r, x_start, x_t, t, label=label)
            vb *= self.T / 1000.
        else:
            pred_noise = model_output
            vb = None

        loss = mean_flat((noise - pred_noise) ** 2)
        if vb is not None:
            loss += vb
        return loss.mean()
