import math
import numpy as np
import torch

class NoiseScheduler():
    ''''
    This class implements a noise scheduler for diffusion models.
    '''
    def __init__(self, p_mean=-1.2, p_std=1.2, num_timesteps=50, sigma_min=0.01, sigma_max=10.):
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigmas = self.get_sigmas_karras(num_timesteps, sigma_min=sigma_min, sigma_max=sigma_max)
        self.betas = self.get_named_beta_schedule("cosine", num_timesteps)
        self.alpha_bars = torch.cumprod(1 - self.betas, dim=0)

    def euler_step(self, model_output, sigma, sample, i):
        """
        ODE step for Euler method.
        Args:
            model_output: E[x | x_t] size [batch_size, n_dim]
            sigma: timestep value. Scalar
            sample: The current sample. Size [batch_size, n_dim]
            i: The current timestep index.
        """
        device = sample.device
        sigma = self.ensure_device(sigma, device)
        if hasattr(self.sigmas, 'device') and self.sigmas.device != device:
            self.sigmas = self.sigmas.to(device)
        
        d = (sample - model_output) / self.append_dims(sigma, sample.ndim)
        dt = self.sigmas[i + 1] - sigma
        sample = sample + d * dt
        return sample, d

    def euler_maruyama_step(self, model_output, sigma, sample, i):
        '''
        SDE step for Euler-Maruyama method.
        Args:
            model_output: E[x | x_t] size [batch_size, n_dim]
            sigma: timestep value. Scalar
            sample: The current sample. Size [batch_size, n_dim]
            i: The current timestep index.
        '''
        device = sample.device
        sigma = self.ensure_device(sigma, device)
        if hasattr(self.sigmas, 'device') and self.sigmas.device != device:
            self.sigmas = self.sigmas.to(device)
        d = (sample - model_output) / self.append_dims(sigma, sample.ndim)
        dt = self.sigmas[i + 1] - sigma
        sample = sample + 2 * d * dt + torch.sqrt(torch.abs(2. * sigma * dt)) * torch.randn_like(sample)
        return sample, d
    
    def edm_stochastic_step(self, 
                            model_output,
                            sigma, sigma_next, sample, i, denoiser, labels,
                            s_churn = 80, 
                            s_min = 0.05,
                            s_max = 1,
                            s_noise = 1.007,):
        '''
        EDM stochastic sampling step for ImageNet.
        Args:
            model_output: Denoiser output D(x_t, sigma) 
            sigma: Current noise level
            sigma_next: Next noise level
            sample: Current sample x_t
            i: Current timestep index
            denoiser: The denoiser model
            labels: Class labels for conditional denoising
        '''
        device = sample.device
        
        num_steps = len(self.sigmas) - 1
        
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, device=device, dtype=sample.dtype)
        else:
            sigma = sigma.to(device)
        
        if not isinstance(sigma_next, torch.Tensor):
            sigma_next = torch.tensor(sigma_next, device=device, dtype=sample.dtype)
        else:
            sigma_next = sigma_next.to(device)
        
        gamma_val = min(s_churn / num_steps, np.sqrt(2) - 1)
        in_range = (sigma >= s_min) & (sigma <= s_max)
        gamma = torch.full_like(sigma, gamma_val if in_range.item() else 0.0)
        
        sigma_hat = sigma * (1 + gamma)
        
        if gamma.item() > 0:
            noise_scale = torch.sqrt(sigma_hat**2 - sigma**2) * s_noise
            eps = torch.randn_like(sample)
            sample = sample + noise_scale * eps
        
        # Denoising step
        sigma_hat_expanded = self.append_dims(sigma_hat, sample.ndim)
        d1 = (sample - model_output) / sigma_hat_expanded
        dt = sigma_next - sigma_hat
        dt_expanded = self.append_dims(dt, sample.ndim)
        x_euler = sample + d1 * dt_expanded
        
        # Compute second derivative
        if (sigma_next.item() != 0):
            sigma_next_expanded = self.append_dims(sigma_next, sample.ndim)
            model_output_2, _ = denoiser.denoise_conditional(x_euler, sigma_next, labels)
            d2 = (x_euler - model_output_2) / sigma_next_expanded
            sample = sample + 0.5 * (d1 + d2) * dt_expanded
        else:
            sample = x_euler

        return sample, d1
    
    def clipping_reverse_step(self, numerator, beta, C, sigma, sample, i):
        '''
        SDE steps for the clipping reverse process with clipping
        Args:
            numerator: v, size [batch_size, n_dim]
            beta: 1/denominator [batch_size, 1]
            C: Clipping constant
            sigma: timestep value. Scalar
            sample: The current sample. Size [batch_size, n_dim]
            i: The current timestep index.
        '''
        device = sample.device
        sigma = self.ensure_device(sigma, device)
        if hasattr(self.sigmas, 'device') and self.sigmas.device != device:
            self.sigmas = self.sigmas.to(device)
        deltaT = sigma - self.sigmas[i + 1] 
        # UPDATED THIS FOR GDP sensitivity
        d = numerator + 2* C * ((sigma * torch.sqrt(sigma/deltaT))/(2**0.5 * beta)) * torch.randn_like(sample)
        sample = (1 - 2 * deltaT/sigma) * sample + 2 * deltaT / sigma * beta * d
        return sample, d
    
    def ddpm_reverse_step(self, model_output, samples, i):
        """
        DDPM reverse step.
        model_output: predicted noise eps(x_t,t)
        samples: x_t
        i: timestep index
        """
        device = samples.device

        beta_t = self.betas[i].to(device)
        alpha_t = 1 - beta_t

        alpha_bar_t = self.alpha_bars[i].to(device)
        alpha_bar_prev = self.alpha_bars[i-1].to(device) if i > 0 else torch.tensor(1.0, device=device)

        # x0 prediction
        x0 = (samples - torch.sqrt(1 - alpha_t) * model_output) / torch.sqrt(alpha_t)

        # posterior mean
        coef1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t)
        coef2 = (torch.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
        mean = coef1 * x0 + coef2 * samples

        # variance
        if i > 0:
            var = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
            noise = torch.randn_like(samples)
            samples = mean + torch.sqrt(var) * noise
        else:
            samples = mean

        return samples, mean
        

    def add_noise(self, x_start, x_noise, timesteps):
        return x_start + timesteps[:,None] * x_noise

    def get_sigma(self, bs, device='cuda:0'):
        log_sigmas = self.p_mean + self.p_std * torch.randn(bs, device=device)
        sigmas = torch.exp(log_sigmas)
        return sigmas

    def get_sigmas_karras(self, n, sigma_min=0.002, sigma_max=80., rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return self.append_zero(sigmas).to(device)
    
    def get_sigma_diffs(self, sigmas):
        diffs = [sigmas[i-1] - sigmas[i] for i in range(1, len(sigmas))]
        diffs = diffs + [sigmas[-1].clone().detach()]  # Append the last sigma as the last difference
        return diffs

    def append_zero(self, x):
        return torch.cat([x, x.new_zeros([1])])
    
    def append_dims(self, x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(
                f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
            )
        return x[(...,) + (None,) * dims_to_append]

    def ensure_device(self, tensor, target_device):
        """
        Ensures a tensor is on the specified device.
        Works with both tensors and scalar values.
        
        Args:
            tensor: Tensor or scalar value to ensure device for
            target_device: Target device (from another tensor's .device attribute)
        
        Returns:
            Tensor on the target device
        """
        if isinstance(tensor, torch.Tensor):
            return tensor.to(target_device) if tensor.device != target_device else tensor
        else:
            return torch.tensor(tensor, device=target_device)
        
    def get_named_beta_schedule(self, schedule_name, num_diffusion_timesteps):
        """
        Get a pre-defined beta schedule for the given name.

        The beta schedule library consists of beta schedules which remain similar
        in the limit of num_diffusion_timesteps.
        Beta schedules may be added, but should not be removed or changed once
        they are committed to maintain backwards compatibility.
        """
        if schedule_name == "linear":
            # Linear schedule from Ho et al, extended to work for any number of
            # diffusion steps.
            scale = 1000 / num_diffusion_timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(
                beta_start, beta_end, num_diffusion_timesteps, dtype=torch.float64
            )
        elif schedule_name == "cosine":
            return self.betas_for_alpha_bar(
                num_diffusion_timesteps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        else:
            raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].

        :param num_diffusion_timesteps: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                        produces the cumulative product of (1-beta) up to that
                        part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                        prevent singularities.
        """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float64)

