import argparse
import os
import pickle

import numpy as np
from sklearn.datasets import make_moons
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from cm.unet import UNetModel
import dnnlib
from empirical_denoiser import EmpiricalDenoiser, ImageNet64NeuralDenoiser, ImageNet32NeuralDenoiser, RandomFourierFeatureDenoiser, SubSampleEmpiricalDenoiser
from noise_scheduler import NoiseScheduler
from utils.visualization import (
    plot_mnist_grid, 
    plot_rgb_grid, 
    plot_diffusion_grid, 
)
from utils.logger import DPLogger, LogLevel

import sys

# Returns a dataset of moons with the specified number of samples of size [n, 2].
def get_moons_dataset(n=8000):
    X, _ = make_moons(n_samples=n, random_state=np.random.randint(1000000), noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    X = torch.from_numpy(X.astype(np.float32)) / 3.
    X = X - X.mean(0)
    dummy_labels = torch.zeros_like(X[:, 0])
    data_loader = DataLoader(TensorDataset(X, dummy_labels), batch_size=n, shuffle=True)
    return data_loader, X.shape[0]

# Returns Flattened MNIST with n samples and size [n, 28*28]. Pixels are normalized to [-1, 1].
def get_mnist_dataset(n):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),            # (x−0.5)/0.5 = 2x−1
                                    transforms.Lambda(lambda x: x.view(-1))
                                    ])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_loader = DataLoader(mnist_train, batch_size=n, shuffle=True)
    return data_loader, len(mnist_train)

# Returns Flattened CIFAR10 with n samples and size [n, 32*32*3]. Pixels are normalized to [-1, 1].
def get_cifar10_dataset(n, shuffle = True, generator = None):
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),            # (x−0.5)/0.5 = 2x−1
                                transforms.Lambda(lambda x: x.view(-1))
                                ])
    cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    data_loader = DataLoader(cifar10_train, batch_size=n, shuffle=shuffle, generator=generator)
    return data_loader, len(cifar10_train)

def get_cifar10_dataset_64(n, shuffle=True, generator=None):
    """Like get_cifar10_dataset but upsamples images to 64x64 and flattens to 12288-dim."""
    transform = transforms.Compose([
        transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    
    cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    data_loader = DataLoader(cifar10_train, batch_size=n, shuffle=shuffle, generator=generator)
    return data_loader, len(cifar10_train)

def get_celeba_dataset_64(n, shuffle=True, generator=None):
    """Returns flattened CelebA at 64x64 with size [n, 64*64*3]. Pixels normalized to [-1, 1].

    CelebA images are 178x218; center-crop to 140x140 around the face then resize to 64x64.
    """
    transform = transforms.Compose([
        transforms.CenterCrop(140),
        transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])

    celeba_train = datasets.CelebA(root='./data', split='train', download=True, transform=transform)
    data_loader = DataLoader(celeba_train, batch_size=n, shuffle=shuffle, generator=generator)
    return data_loader, len(celeba_train)

def filter_dataset(dataset):
    filter_ = (dataset[:,1] < -0.66)
    return dataset[filter_], len(dataset) / len(dataset[filter_])

def perform_PCA_2d(centered_data):
    '''
    Perform PCA on the dataset to reduce it to 2D.
    Args:
        centered_data: The dataset to perform PCA on. Shape [n_samples, n_dim]
    Returns:
        data_2d: The dataset reduced to 2D. Shape [n_samples, 2]
        V: The PCA components. Shape [2, n_dim]
    '''
    U, S, V = torch.linalg.svd(centered_data, full_matrices=False)
    # Project data onto the first two principal components.
    
    data_2d = torch.matmul(centered_data, V[:2, :].T)
    return data_2d, V[:2, :].T

def denoise_reverse_step(denoiser, sampler_type, C, noise_scheduler, sigmas_for_inference, samples, i, t, device = "cuda"):
    class_labels = torch.randint(0, 1000, (samples.size(0),), device=device)
    class_labels_one_hot = F.one_hot(class_labels, num_classes=1000).float()
    with torch.no_grad():
        residual3, beta_t = denoiser.denoise(samples, t, i=i)
    if sampler_type == 'ode':
        samples, _ = noise_scheduler.euler_step(residual3, t, samples, i)
    elif sampler_type == 'sde':
        samples, _ = noise_scheduler.euler_maruyama_step(residual3, t, samples, i)
    elif sampler_type == 'clipsde':
        samples, _ = noise_scheduler.clipping_reverse_step(
                        denoiser.get_numerator(), beta_t, C, t, samples, i)
    elif sampler_type == 'edm':
        samples, _ = noise_scheduler.edm_stochastic_step(
                        residual3, t, sigmas_for_inference[i+1], samples, i, denoiser, class_labels_one_hot)
    else:
        raise NotImplementedError
    return samples, residual3

def get_imagenet32_neural_denoiser(model_path, device='cuda'):
    model_precond = UNetModel(
        # image sizes
        input_size=32,
        output_size=32,

        # core channels
        in_channels=3,
        out_channels=3,            # 3 * 2 (learn_sigma=True)
        model_channels=256,

        # block structure
        num_res_blocks=3,          # 2 residual blocks per resolution
        encoder_channel_mult=[1, 2, 4],  # encoder progression
        decoder_channel_mult=[1, 2, 4],  # decoder progression

        # attention
        attention_resolutions=(2, 4, 8, 16),
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        attention_type="standard",    # if you want flash attention; otherwise "standard"

        # normalization & residuals
        use_scale_shift_norm=True,
        resblock_updown=True,
        dropout=0.1,

        # training options
        num_classes=1000,          # ImageNet conditional
        # num_classes = None,
        use_checkpoint=False,
        use_fp16=False,
        use_new_attention_order=False,

        # extra args your class supports
        model_channels_high=256,   # just mirror base; not used in EDM
        use_scale_shift_norm_high=False,
        num_res_blocks_high=0,
        training_mode="",
        new=False,
        no_grad=False,
    )
    
    model_pretrained_weights = torch.load(model_path, map_location='cpu')
    model_precond.load_state_dict(model_pretrained_weights, strict = True)
    model_precond.eval()
    
    return ImageNet32NeuralDenoiser(model_precond, device=device)

def generate_samples_standard(
    denoiser,
    dimen,
    num_samples=1000,
    batch_size=100,
    dataset_name="cifar10",
    sigma_min=0.02, 
    sigma_max=80.0, 
    num_timesteps=50, 
    sampler_type="sde", 
    C=1.0, 
    rho=7.0,
    save=False,
    experiment_name="standard",
    samplesdir="./samples",
    imgdir="./images_app",
    device=None,
    logger=None,
    config=None,
):
    """
    Generate samples using the standard (non-row-subsampled) approach.
    
    Args:
        denoiser: The empirical denoiser object
        dimen: Dimensionality of the data
        num_samples: Number of samples to generate
        batch_size: Batch size for sample generation
        dataset_name: Name of the dataset (mnist, cifar10)
        sigma_min: Minimum sigma value for noise schedule
        sigma_max: Maximum sigma value for noise schedule
        num_timesteps: Number of timesteps for diffusion
        sampler_type: Type of sampler (sde, ode, clipsde)
        C: Clipping constant
        rho: Rho value for Karras schedule
        save: Whether to save the samples
        experiment_name: Name for the experiment
        samplesdir: Directory to save samples
        imgdir: Directory to save images
        logger: Logger instance
        
    Returns:
        Generated samples tensor and visualization data
    """
    if logger:
        logger.info(f"Generating {num_samples} samples using standard approach")
    else:
        print(f"Generating {num_samples} samples using standard approach")
        
    noise_scheduler = NoiseScheduler(
        sigma_min=sigma_min, 
        sigma_max=sigma_max, 
        num_timesteps=num_timesteps
    )
    
    sigmas_for_inference = noise_scheduler.get_sigmas_karras(
        num_timesteps, 
        sigma_max=sigma_max, 
        sigma_min=sigma_min,
        rho=rho,
    )
    
    num_batches = int(np.ceil(num_samples / batch_size))
    
    # Store all samples
    all_samples = []
    
    # Visualization data
    sample_transformation = []
    timesteps = []
    num_denoise = min(5, num_samples)
    num_sigma_beta = min(10, num_samples)
    sigma_beta_t = [[] for _ in range(num_sigma_beta)]
    first_run = True
    
    # Sample in batches
    for batch_idx in range(num_batches):
        if logger:
            logger.info(f"Sampling batch {batch_idx + 1}/{num_batches}...")
        else:
            print(f"Sampling batch {batch_idx + 1}/{num_batches}...")
            
        sample_size = min(batch_size, num_samples - batch_idx * batch_size)
        samples = torch.randn([sample_size, dimen], device=device) * sigma_max
        
        for i, t in enumerate(tqdm(sigmas_for_inference)):
            if i < sigmas_for_inference.shape[0] - 1:
                samples, residual3 = denoise_reverse_step(denoiser, sampler_type, C, noise_scheduler, sigmas_for_inference, samples, i, t)
                if first_run:
                    timesteps.append(t)
                    if dataset_name in ["mnist", "cifar10", "imagenet32"]:
                        sample_transformation.append(samples.clone().detach().cpu()[:num_denoise])
        
        all_samples.append(samples.clone().detach().cpu())
        first_run = False
    
    sample_tensor = torch.cat(all_samples, dim=0)
    if logger:
        logger.info(f"Final sample tensor shape: {sample_tensor.shape}")
    else:
        print(f"Final sample tensor shape: {sample_tensor.shape}")
    
    # Save the samples
    if save:
        sample_name = f"samples_{dataset_name}_{experiment_name}_{sampler_type}.pt"
        torch.save({"samples": sample_tensor}, os.path.join(samplesdir, sample_name))
        if logger:
            logger.info(f"Samples saved to {os.path.join(samplesdir, sample_name)}")
        else:
            print(f"Samples saved to {os.path.join(samplesdir, sample_name)}")
    
    # Return samples and visualization data
    return {
        "samples": sample_tensor,
        "sample_transformation": sample_transformation,
        "timesteps": timesteps,
        "sigma_beta_t": sigma_beta_t
    }

def generate_samples_with_row_subsampling(
    full_dataset, 
    data_loader,
    num_samples=10000, 
    samples_per_seed=1000, 
    dataset_name="cifar10",
    sigma_min=0.02, 
    sigma_max=80.0, 
    num_timesteps=50, 
    sampler_type="sde", 
    C=1.0, 
    rho=7.0,
    save=False,
    experiment_name="row_subsampled",
    samplesdir="./samples",
    imgdir="./images_app",
    logger=None,
    beta_schedule=None,
    comp_batch_size=100, # note, this is the computational batch size, not m in q=m/n
    config=None,
    device=None
):
    """
    generates samples using row-wise subsampling with different seeds.
    
    Args:
        full_dataset: The complete dataset tensor
        data_loader: DataLoader for getting different subsets of data
        num_samples: Total number of samples to generate
        samples_per_seed: Number of samples to generate per seed
        dataset_name: Name of the dataset (mnist, cifar10)
        sigma_min: Minimum sigma value for noise schedule
        sigma_max: Maximum sigma value for noise schedule
        num_timesteps: Number of timesteps for diffusion
        sampler_type: Type of sampler (sde, ode, clipsde)
        C: Clipping constant
        rho: Rho value for Karras schedule
        save: Whether to save the samples
        experiment_name: Name for the experiment
        samplesdir: Directory to save samples
        imgdir: Directory to save images
        logger: Logger instance
        batch_size: Computational batch size for generation (memory management)
        
    Returns:
        Generated samples tensor
    """
    if logger:
        logger.info(f"Generating {num_samples} samples using row-wise subsampling")
    else:
        print(f"Generating {num_samples} samples using row-wise subsampling")
    
    # calculate how many seeds we need
    num_seeds = (num_samples + samples_per_seed - 1) // samples_per_seed
    dimen = full_dataset.shape[1]
    
    noise_scheduler = NoiseScheduler(
        sigma_min=sigma_min, 
        sigma_max=sigma_max, 
        num_timesteps=num_timesteps
    )
    
    sigmas_for_inference = noise_scheduler.get_sigmas_karras(
        num_timesteps, 
        sigma_max=sigma_max, 
        sigma_min=sigma_min,
        rho=rho,
    )
    
    all_generated_samples = []
    num_denoise_samples = min(5, samples_per_seed)
    sample_transformation = []
    timesteps = []
    first_seed = True
    
    for seed_idx in range(num_seeds):
        if config.batch_seed == 1:
            seed_value = config.batch_seed if num_seeds == 1 else config.batch_seed + seed_idx
            if logger:
                logger.info(f"Using provided batch seed: {seed_value} ({seed_idx + 1}/{num_seeds})")
            else:
                print(f"Using provided batch seed: {seed_value} ({seed_idx + 1}/{num_seeds})")
        else:
            seed_value = seed_idx + 1
            if logger:
                logger.info(f"Generating samples with seed {seed_value} ({seed_idx + 1}/{num_seeds})")
            else:
                print(f"Generating samples with seed {seed_value} ({seed_idx + 1}/{num_seeds})")
        
        torch.manual_seed(seed_value)
        np.random.seed(seed_value)

        generator = torch.Generator()
        generator.manual_seed(seed_value)
        
        seed_dataset = TensorDataset(full_dataset, torch.zeros(full_dataset.shape[0]))
        seed_data_loader = DataLoader(
            seed_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            generator=generator
        )
        subsampled_dataset, _ = next(iter(seed_data_loader))
        subsampled_dataset = subsampled_dataset.to(device)
        
        # create a denoiser with the subsampled dataset
        curr_denoiser = EmpiricalDenoiser(
            subsampled_dataset,
            data_loader=data_loader,
            clipping=(sampler_type == "clipsde"), 
            n_dim=dimen, 
            C=C,
            beta_schedule=beta_schedule,
            non_dp=(beta_schedule is None)  # use non-DP version
        )
        samples_for_this_seed = min(samples_per_seed, num_samples - sum(len(batch) for batch in all_generated_samples))
        
        seed_samples = []
        num_batches = (samples_for_this_seed + comp_batch_size - 1) // comp_batch_size
        
        for batch_idx in range(num_batches):
            if logger:
                logger.info(f"Seed {seed_value}, processing batch {batch_idx + 1}/{num_batches}")
            else:
                print(f"Seed {seed_value}, processing batch {batch_idx + 1}/{num_batches}")
            
            # Calculate batch size for this iteration
            current_batch_size = min(comp_batch_size, samples_for_this_seed - len(seed_samples))
            
            samples = torch.randn([current_batch_size, dimen], device=device) * sigma_max
            
            # Denoising loop
            for i, t in enumerate(tqdm(sigmas_for_inference)):
                if i < sigmas_for_inference.shape[0] - 1:
                    samples, residual3 = denoise_reverse_step(curr_denoiser, sampler_type, C, noise_scheduler, sigmas_for_inference, samples, i, t)
                    # track denoising process for visualization (first seed, first batch only)
                    if first_seed and batch_idx == 0:
                        timesteps.append(t)
                        if dataset_name in ["mnist", "cifar10"]:
                            sample_transformation.append(samples.clone().detach().cpu()[:num_denoise_samples])
                if i == len(sigmas_for_inference) - 2:
                    print("Final sample stats:", samples.min().item(), samples.max().item(), samples.std().item())
                if i == len(sigmas_for_inference) - 2:
                    final_x0 = residual3
                    print("x0 stats :", final_x0.min().item(), final_x0.max().item(), final_x0.std().item())

            # seed_samples.append(samples.clone().detach().cpu())
            # seed_samples.append(residual3.clone().detach().cpu())
            seed_samples.append(final_x0.clone().detach().cpu())
            
            if logger:
                logger.info(f"Generated {sum(batch.shape[0] for batch in seed_samples)}/{samples_for_this_seed} samples for seed {seed_value}")
            else:
                print(f"Generated {sum(batch.shape[0] for batch in seed_samples)}/{samples_for_this_seed} samples for seed {seed_value}")
        
        all_generated_samples.append(torch.cat(seed_samples, dim=0))
        first_seed = False
         
    sample_tensor = torch.cat(all_generated_samples, dim=0)[:num_samples]
    if logger:
        logger.info(f"Final sample tensor shape: {sample_tensor.shape}")
    else:
        print(f"Final sample tensor shape: {sample_tensor.shape}")
    
    if save:
        sample_name = f"samples_{dataset_name}_{experiment_name}_{sampler_type}.pt"
        torch.save({"samples": sample_tensor}, os.path.join(samplesdir, sample_name))
        if logger:
            logger.info(f"Samples saved to {os.path.join(samplesdir, sample_name)}")
        else:
            print(f"Samples saved to {os.path.join(samplesdir, sample_name)}")
    
    # Visualize denoising process if we have tracking data
    if sample_transformation and dataset_name in ["mnist", "cifar10"]:
        if dataset_name == "mnist":
            plot_diffusion_grid(
                sample_transformation, 
                timesteps, 
                dimen=(28, 28), 
                imgname=f"denoiser_{sampler_type}_{experiment_name}",
                imgdir=imgdir, 
                sigma_max=sigma_max, 
                sigma_min=sigma_min
            )
        elif dataset_name == "cifar10":
            plot_diffusion_grid(
                sample_transformation, 
                timesteps, 
                rgb=True, 
                dimen=(32, 32), 
                imgname=f"denoiser_{sampler_type}_{experiment_name}",
                imgdir=imgdir, 
                sigma_max=sigma_max, 
                sigma_min=sigma_min
            )

    if dataset_name == "cifar10":
        max_vis_samples = min(100, num_samples)
        plot_rgb_grid(sample_tensor[:max_vis_samples], dimen=(32, 32), grid_size=(10, 10),
                          imgname=f"cifar_samples_{sampler_type}_{experiment_name}", imgdir=imgdir,
                          sigma_min=sigma_min, sigma_max=sigma_max)
    elif dataset_name == "mnist":
        max_vis_samples = min(100, num_samples)
        plot_mnist_grid(sample_tensor[:max_vis_samples], dimen=(28, 28), grid_size=(10, 10),
                       imgname=f"mnist_samples_{sampler_type}_{experiment_name}", imgdir=imgdir,
                       sigma_min=sigma_min, sigma_max=sigma_max)
    
    
    return {
        "samples": sample_tensor,
        "sample_transformation": sample_transformation,
        "timesteps": timesteps,
    }