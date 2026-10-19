import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import autocast


class EmpiricalDenoiser:
    """
    A class for implementing an empirical denoiser.
    """

    def __init__(self, dataset, data_loader = None, clipping=False, n_dim=2, C=1.0, beta_hyper=None, beta_schedule=None, non_dp=False, device = 'cpu', is_full_dataset = False):
        self.clipping = clipping
        self.n_dim = n_dim
        self.C = C
        self.beta_hyper = beta_hyper
        self.beta_schedule = beta_schedule
        if (dataset is not None):
            self.ones = torch.ones(dataset.shape[0], 1, device=dataset.device)
            self.dataset = dataset.reshape(1, -1, n_dim)
        self._denominator = None
        self._numerator = None
        self._non_dp = non_dp
        self.device = device
        self.is_full_dataset = is_full_dataset
        self.data_loader = data_loader

    def denoise(self, query, sigma, i=None):
        '''
        Denoise the query using the empirical denoiser. Primarily to calculate equation 9: E[x | x_t]
        Args:
            query: The query tensor to denoise. Size [batch_size, n_dim]
            sigma: timestep value. Scalar
            i: the index at that timestep. Scalar
        
        Returns:
            The denoised tensor. numerator/denominator [batch_size, n_dim]
            beta: 1/denominator [batch_size, 1] 
        '''
        if not self.is_full_dataset:
            self.denoise_batch(query, sigma, i)
            return self._numerator * self._denominator, self._denominator
        else:
            return self._denoise_full_dataset_online(query, sigma, i)

    def _denoise_full_dataset_online(self, query, sigma, i):
        """Single-pass online log-sum-exp over the full dataset.

        Accumulates weighted sums in one pass by rescaling whenever a new batch
        raises the per-query running maximum — mathematically equivalent to the
        two-pass (find-global-max, then accumulate) approach but twice as fast.

        The rescaling identity:
            acc  ← acc * exp(old_max − new_max)
            acc += exp(log_kernel − new_max) @ data
        ensures that, after all batches, acc equals the result you would get
        from a single pass with the true global max subtracted throughout.
        """
        epsilon = 1e-12
        running_max = None   # [B, 1]  per-query max seen so far
        acc_num = None       # [B, n_dim]
        acc_den = None       # [B, 1]

        for inputs, _ in self.data_loader:
            batch = inputs.reshape(-1, self.n_dim).to(self.device)
            log_kernel = self.get_log_kernel(query, sigma, dataset=batch)  # [B, N_batch]
            batch_max = log_kernel.max(dim=1, keepdim=True)[0]             # [B, 1]

            if running_max is None:
                running_max = batch_max
                weights = (log_kernel - running_max).exp()
            else:
                new_max = torch.maximum(running_max, batch_max)
                # Rescale accumulated sums to the new (higher) max.
                scale = (running_max - new_max).exp()  # in (0, 1]
                acc_num = acc_num * scale
                acc_den = acc_den * scale
                running_max = new_max
                weights = (log_kernel - running_max).exp()

            if self.clipping:
                norms = batch.norm(p=2, dim=-1).reshape(1, -1)
                scales = torch.clamp(self.C / (weights * norms + epsilon), max=1.0)
                batch_num = (weights * scales) @ batch
            else:
                batch_num = weights @ batch
            batch_den = weights.sum(dim=1, keepdim=True)

            if acc_num is None:
                acc_num = batch_num
                acc_den = batch_den
            else:
                acc_num = acc_num + batch_num
                acc_den = acc_den + batch_den

        if self.beta_schedule is not None:
            if i is not None:
                self._denominator = torch.ones_like(acc_den) * self.beta_schedule[i]
            else:
                raise ValueError("i must be provided when using beta_schedule")
        elif self.beta_hyper is not None:
            self._denominator = torch.ones_like(acc_den) * self.beta_hyper
        else:
            self._denominator = 1.0 / torch.clamp(acc_den + epsilon, min=1e-6, max=1e6)

        self._numerator = acc_num
        denoised = acc_num / torch.clamp(acc_den + epsilon, min=1e-6)
        return denoised, self._denominator

    def get_log_kernel(self, query, sigma, dataset=None):
        epsilon = 1e-12
        if dataset is None:
            dataset = self.dataset[0]
        query = query.reshape(-1, self.n_dim)
        dataset = dataset.reshape(-1, self.n_dim)

        euclidian_distance = (
            query.square().sum(dim=1, keepdim=True)
            + dataset.square().sum(dim=1).reshape(1, -1)
            - 2.0 * (query @ dataset.T)
        ).clamp_min_(0)
        return -euclidian_distance / (2. * sigma ** 2 + epsilon)

    def get_kernel(self, query, sigma):
        log_kernel = self.get_log_kernel(query, sigma)
        return log_kernel.reshape(query.shape[0], self.dataset.shape[1], 1).expand(-1, -1, self.n_dim + 1)

    def _compute_weighted_sums(self, query, sigma, global_max=None):
        epsilon = 1e-12
        dataset = self.dataset[0]
        log_kernel = self.get_log_kernel(query, sigma, dataset)

        # Log sum exp (max) trick to avoid overflow and restrict the range of the kernel
        if global_max is not None:
            if global_max.dim() == 3:
                global_max = global_max[..., 0]
            logits_centered = log_kernel - global_max
        else:
            max_kernel = log_kernel.max(dim=1, keepdim=True)[0]
            logits_centered = log_kernel - max_kernel

        weights = logits_centered.exp()
        if self.clipping:
            norms = dataset.norm(p=2, dim=-1).reshape(1, -1)
            scales = torch.clamp(self.C / (weights * norms + epsilon), max=1.0)
            numerator = (weights * scales) @ dataset
        else:
            numerator = weights @ dataset
        denominator = weights.sum(dim=1, keepdim=True)
        return numerator, denominator

    def denoise_batch(self, query, sigma, i, global_max=None):
        epsilon = 1e-12  # Small constant to prevent division by zero
        numerator, denominator = self._compute_weighted_sums(query, sigma, global_max)
        
        # return the beta value for testing
        if self.beta_schedule is not None:
            if i is not None:
                timestep_beta = self.beta_schedule[i]
                self._denominator = torch.ones_like(denominator) * timestep_beta
            else:
                raise ValueError("i must be provided when using beta_schedule")
        elif self.beta_hyper is not None:
            self._denominator = torch.ones_like(denominator) * self.beta_hyper
        else:
            self._denominator = 1/torch.clamp((denominator + epsilon), min = 1e-6, max = 1e6)
       
        self._numerator = numerator

    def clipping_dataset_row(self, row, C):
        return row/torch.max(torch.tensor(1), torch.norm(row, p = 2)/C)
    
    def get_denominator(self):
        return self._denominator
    def get_numerator(self):
        return self._numerator
    
class CIFAR10NeuralDenoiser(EmpiricalDenoiser):
    """
    A class for implementing a CIFAR10 denoiser
    """
    
    def __init__(self, model, device = 'cpu'):
        super().__init__(dataset=None, data_loader = None, device=device)
        self.model = model.to(device)

    def denoise(self, query, sigma, i=None):
        '''
        Denoise the query using the neural network denoiser.
        Args:
            query: The query tensor to denoise. Size [batch_size, n_dim]
            sigma: timestep value. Scalar
            i: the index at that timestep. Scalar
        
        Returns:
            The denoised tensor. [batch_size, n_dim]
            beta: None
        '''
        with torch.no_grad():
            query = query.to(self.device)
            query = query.reshape(query.shape[0], 3, 32, 32)
            denoised = self.model(query.to(self.device), torch.full((query.shape[0],), sigma.item(), device=self.device)).to(torch.float64)
                
        denoised = denoised.view(denoised.size(0), -1)
        
        return denoised, 1
    
class ImageNet32NeuralDenoiser(EmpiricalDenoiser):
    """
    A class for implementing a neural network denoiser.
    """

    def __init__(self, model, device = 'cpu'):
        super().__init__(dataset=None, data_loader = None, device=device)
        self.model = model.to(device)

    def denoise(self, query, sigma, i=None):
        '''
        Denoise the query using the neural network denoiser.
        Args:
            query: The query tensor to denoise. Size [batch_size, n_dim]
            sigma: timestep value. Scalar
            i: the index at that timestep. Scalar
        
        Returns:
            The denoised tensor. [batch_size, n_dim]
            beta: None
        '''
        null_class_index = self.model.num_classes
        batch = query.shape[0]
        sigma_batch = torch.full((batch,), sigma.item(), device=self.device)
        class_labels = torch.full((batch,), fill_value = 20, dtype=torch.long, device=self.device)

        with torch.no_grad():
            query = query.to(self.device)
            query = query.reshape(query.shape[0], 3, 32, 32)
            denoised = self.model(query.to(self.device),
                timesteps = sigma_batch,   # make it shape [1]
                y = class_labels).to(torch.float32)
                
        # denoised = query - sigma_batch * denoised
        denoised = denoised.view(denoised.size(0), -1)
        
        print("Max, min, std:", (denoised.max(), denoised.min(), denoised.std()))
        
        return denoised, 1
    
    def generate_ddpm_sigmas(self, T: int = 1000, B_MIN: float = 1e-4, B_MAX: float = 0.02, device: str = 'cpu') -> torch.Tensor:
        """
        Generates the DDPM noise standard deviation (sigma_t) schedule tensor.

        This tensor is required to map continuous sigma values (like those from 
        Karras scheduling) to the discrete integer timesteps (0 to T-1) of a DDPM-trained U-Net.

        Args:
            T: Total number of discrete timesteps (default: 1000).
            B_MIN: Minimum beta value (default: 1e-4).
            B_MAX: Maximum beta value (default: 0.02).
            device: The PyTorch device to place the tensor on (e.g., 'cuda:0').

        Returns:
            A torch.Tensor of shape (T,) containing the sigma_t values, ready to be 
            used as self.model_sigmas.
        """
        # 1. Calculate the beta values (linear schedule)
        betas = torch.linspace(B_MIN, B_MAX, T, dtype=torch.float64, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # 3. Calculate sigma_t using the DDPM formula
        model_sigmas = torch.sqrt((1.0 - alphas_cumprod) / alphas_cumprod).float()
        
        print("Model Sigmas:", model_sigmas)
        
        return model_sigmas
    
    def sigma_to_t_index(self, sigma, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        """
        Convert Karras sigma to a UNet timestep in [0..999].
        """
        sigma = torch.clamp(sigma, sigma_min, sigma_max)

        t_cont = ((sigma ** (1/rho)) - (sigma_max ** (1/rho))) / \
                ((sigma_min ** (1/rho)) - (sigma_max ** (1/rho)))

        t_index = torch.round(t_cont * 999).long()
        return t_index
    
class ImageNet64NeuralDenoiser(ImageNet32NeuralDenoiser):
    """
    A class for implementing a neural network denoiser for ImageNet64.
    """

    def __init__(self, model, device = 'cpu'):
        super().__init__(model, device=device)

    def denoise(self, query, sigma, i=None):
        '''
        Denoise the query using the neural network denoiser.
        Args:
            query: The query tensor to denoise. Size [batch_size, n_dim]
            sigma: timestep value. Scalar
            i: the index at that timestep. Scalar
        
        Returns:
            The denoised tensor. [batch_size, n_dim]
            beta: None
        '''
        with torch.no_grad():
            query = query.reshape(query.shape[0], 3, 64, 64)
            
            batch = query.shape[0]
            sigma_batch = torch.full((batch,), sigma.item(), device=self.device)
            random_classes = torch.randint(0, 1000, (batch,), device=self.device)
            labels = torch.nn.functional.one_hot(random_classes, num_classes=1000).float()

            denoised = self.model(query, sigma_batch, labels).to(torch.float32)
            denoised = denoised.view(denoised.size(0), -1)
        
        return denoised, 1
    
    def denoise_conditional(self, query, sigma, class_labels, i=None):
        '''
        Denoise the query using the neural network denoiser with class conditioning.
        Args:
            query: The query tensor to denoise. Size [batch_size, n_dim]
            sigma: timestep value. Scalar
            class_labels: One hot Tensor of class labels for conditioning. Size [batch_size, num_classes]
            i: the index at that timestep. Scalar
        
        Returns:
            The denoised tensor. [batch_size, n_dim]
            beta: None
        '''
        n_dim = query.shape[1]
        spatial = int((n_dim / 3) ** 0.5)
        query = query.reshape(query.shape[0], 3, spatial, spatial)

        batch = query.shape[0]
        sigma_batch = torch.full((batch,), sigma, device=self.device)

        with torch.inference_mode():
            denoised = self.model(query, sigma_batch, class_labels)
        denoised = denoised.view(denoised.size(0), -1)
        
        return denoised, 1
    
class FFHQ64NeuralDenoiser(ImageNet64NeuralDenoiser):
    """
    A class for implementing an unconditional neural network denoiser (label_dim=0), e.g. FFHQ64.
    """

    def denoise(self, query, sigma, i=None):
        # The FFHQ checkpoint runs in float32, so autocast must stay disabled
        spatial = int((query.shape[1] / 3) ** 0.5)
        with torch.no_grad(), autocast(enabled=False):
            query = query.reshape(query.shape[0], 3, spatial, spatial).float()
            batch = query.shape[0]
            sigma_batch = torch.full((batch,), sigma.item(), device=self.device)
            denoised = self.model(query, sigma_batch, None).to(torch.float32)
            denoised = denoised.view(denoised.size(0), -1)
        return denoised, 1

    def denoise_conditional(self, query, sigma, class_labels=None, i=None):
        # class_labels is ignored; kept for compatibility with the shared neural sampling loop
        spatial = int((query.shape[1] / 3) ** 0.5)
        batch = query.shape[0]
        with torch.inference_mode(), autocast(enabled=False):
            query = query.reshape(query.shape[0], 3, spatial, spatial).float()
            sigma_batch = torch.full((batch,), sigma, device=self.device)
            denoised = self.model(query, sigma_batch, None).to(torch.float32)
        denoised = denoised.view(denoised.size(0), -1)
        return denoised, 1

class SubSampleEmpiricalDenoiser(EmpiricalDenoiser):
    """
    A class for implementing an empirical denoiser with subsampling.
    """

    def __init__(self, dataset, data_loader, clipping=False, n_dim=2, C=1.0, beta_hyper=None, beta_schedule=None, non_dp=False, device = 'cpu'):
        super().__init__(dataset, data_loader, clipping, n_dim, C, beta_hyper, beta_schedule, non_dp, device)

    def denoise(self, query, sigma, i=None):
        self.get_subsample()
        return super().denoise(query, sigma, i=i)

    def get_subsample(self):
        '''
        Get a subsample of the dataset for the empirical denoiser.
        Uses a shuffle = True dataloader to get a random batch_size sample of the dataset.
        After the first call, the datasets are replaced, aka take m random samples, then replace in the dataset.
        Sample with overlap!
        '''
        batch_features, _ = next(iter(self.data_loader))
        batch_features = batch_features.reshape(1, -1, self.n_dim)
        self.dataset = batch_features.to(self.device)
        return self.dataset 
    
class RowSampleEmpiricalDenoiser(EmpiricalDenoiser):
    """
    A class for implementing an empirical denoiser with row subsampling.
    """

    def __init__(self, dataset, data_loader = None, clipping=False, n_dim=2, C=1.0, beta_hyper=None, beta_schedule=None, non_dp=False, device = 'cpu'):
        super().__init__(dataset, data_loader, clipping, n_dim, C, beta_hyper, beta_schedule, non_dp, device)

    def denoise(self, query, sigma, i=None):
        # Do not resample for every query, only once
        return super().denoise(query, sigma, i=i)
    
    def get_subsample(self):
        '''
        Get a subsample of the dataset for the empirical denoiser.
        Uses a shuffle = True dataloader to get a random batch_size sample of the dataset.
        After the first call, the datasets are replaced, aka take m random samples, then replace in the dataset.
        '''
        batch_features, _ = next(iter(self.data_loader))
        batch_features = batch_features.reshape(1, -1, self.n_dim)
        self.dataset = batch_features.to(self.device)
        return self.dataset 
    
class BatchCyclingEmpiricalDenoiser(EmpiricalDenoiser):
    def __init__(self, dataset, data_loader = None, clipping=False, n_dim=2, C=1.0, beta_hyper=None, beta_schedule=None, non_dp=False, device = 'cpu'):
        super().__init__(dataset, data_loader, clipping, n_dim, C, beta_hyper, beta_schedule, non_dp, device)
        self.data_iter = iter(self.data_loader)
        
    def denoise(self, query, sigma, i=None):
        self.get_subsample()
        return super().denoise(query, sigma, i=i)
    
    def get_subsample(self):
        '''
        Get a subsample of the dataset for the empirical denoiser.
        Uses a shuffle = True dataloader to get a random batch_size sample of the dataset.
        After the first call, the datasets are replaced, aka take m random samples, then replace in the dataset.
        '''
        try:
            batch_features, _ = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            batch_features, _ = next(self.data_iter)
        
        batch_features = batch_features.reshape(1, -1, self.n_dim)
        self.dataset = batch_features.to(self.device)
        return self.dataset

class RandomFourierFeatureDenoiser(EmpiricalDenoiser):
    """
    A class for implementing an empirical denoiser with random fourier features.
    """

    def __init__(self, dataset, data_loader = None, clipping=False, n_dim=2, C=1.0, beta_hyper=None, beta_schedule=None, non_dp=False, device = 'cpu', D=1000):
        super().__init__(dataset, data_loader, clipping, n_dim, C, beta_hyper, beta_schedule, non_dp, device)
        self.D = D
        self.W = torch.randn(self.n_dim, self.D).to(device) # Random projection matrix
        self.b = 2 * np.pi * torch.rand(self.D).to(device)  # Random phase shifts
        self.dataset = dataset.reshape(-1, n_dim).to(device)
        self.dataset_kernel = self.get_precomputed_dataset_rff(self.dataset)

    def get_rff_features(self, x):
        '''
        Compute the random Fourier features for the input x.
        Args:
            x: Input tensor of shape [batch_size, n_dim]
        
        Returns:
            RFF features of shape [batch_size, D]
        '''
        projection = x @ self.W + self.b
        rff_features = torch.sqrt(torch.tensor(2.0 / self.D)) * torch.cos(projection)
        return rff_features

    def get_precomputed_dataset_rff(self, dataset):
        M = torch.zeros(self.n_dim, self.D).to(self.device)
        for i in range(dataset.shape[0]):
            x = dataset[i]
            rff_features = self.get_rff_features(x)
            M += x.unsqueeze(0).T @ rff_features.unsqueeze(0)
        return M

    def denoise(self, query, sigma, i=None):
        '''
        Denoise the query using the empirical denoiser with RFF.
        Args:
            query: The query tensor to denoise. Size [batch_size, n_dim]
            sigma: timestep value. Scalar
            i: the index at that timestep. Scalar
        
        Returns:
            The denoised tensor. numerator/denominator [batch_size, n_dim]
            beta: 1/denominator [batch_size, 1] 
        '''
        query_rff = self.get_rff_features(query)
        kernel_ = query_rff @ self.dataset_kernel.T

        # For testing
        self.dataset = self.dataset.reshape(1, -1, self.n_dim).to(self.device)
        numerator, denominator = super().denoise(query, sigma, i)
        self.dataset = self.dataset.reshape(-1, self.n_dim).to(self.device)
        cos_sim = F.cosine_similarity(numerator, kernel_, dim=1)
        print("Cosine similarity between numerator and kernel:", cos_sim.mean().item())

        return kernel_, self.beta_hyper
