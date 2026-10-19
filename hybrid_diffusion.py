import argparse
import os
import pickle

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import dnnlib
from empirical_denoiser import EmpiricalDenoiser, ImageNet64NeuralDenoiser, ImageNet32NeuralDenoiser, FFHQ64NeuralDenoiser, RowSampleEmpiricalDenoiser, BatchCyclingEmpiricalDenoiser, SubSampleEmpiricalDenoiser
from noise_scheduler import NoiseScheduler
from utils.visualization import (
    plot_mnist_grid, 
    plot_rgb_grid, 
    plot_diffusion_grid,
)
from utils.logger import DPLogger, LogLevel

import gc
import sys
from torch_utils import distributed as dist
from diffusion_2d import *
from torch.cuda.amp import autocast

from cm.unet import UNetModel

def denoising_step_hybrid(t_switch, 
                          model1, 
                          model2,
                          sampler_type,
                          sigmas_for_inference,
                          noise_scheduler,
                          sigma_max = 80.0,
                          dimen = 2,
                          num_samples = 1000, 
                          device = 'cuda',
                          empirical_toggle = False,
                          C = 1.0,
                          ):
    # Store all samples
    all_samples = []
    # Store one_step_samples
    one_step_samples = []

    # Visualization data
    sample_transformation = []
    timesteps = []
    num_denoise = min(5, num_samples)
    first_run = True
    samples = torch.randn([num_samples, dimen], device=device) * sigma_max
    denoiser = model1
    # Ensure the same noise
    # torch.manual_seed(42)
    # torch.cuda.manual_seed_all(42)
    for i, t in enumerate(tqdm(sigmas_for_inference)):
        if i < sigmas_for_inference.shape[0] - 1:
            if (i == t_switch + 1):
                denoiser = model2
                empirical_toggle = not empirical_toggle
            with torch.no_grad():
                residual3, beta_t = denoiser.denoise(samples, t, i=i)
            if sampler_type == 'ode':
                samples, _ = noise_scheduler.euler_step(residual3, t, samples, i)
            elif sampler_type == 'sde':
                samples, _ = noise_scheduler.euler_maruyama_step(residual3, t, samples, i)
            elif sampler_type == 'clipsde':
                if (empirical_toggle):
                    samples, _ = noise_scheduler.clipping_reverse_step(
                    denoiser.get_numerator(), beta_t, C, t, samples, i)
                else:
                    samples, _ = noise_scheduler.euler_maruyama_step(residual3, t, samples, i)
            elif sampler_type == 'ddpm':
                if (empirical_toggle):
                    samples, _ = noise_scheduler.clipping_reverse_step(
                    denoiser.get_numerator(), beta_t, C, t, samples, i)
                else:
                    samples, _ = noise_scheduler.ddpm_reverse_step(residual3, samples, len(sigmas_for_inference) - i - 2)
            elif sampler_type == 'heun':
                if (empirical_toggle):
                    samples, _ = noise_scheduler.clipping_reverse_step(
                    denoiser.get_numerator(), beta_t, C, t, samples, i)
                else:
                    samples = noise_scheduler.heun_step(denoiser, t, samples, i)
            elif sampler_type == 'edm':
                if (empirical_toggle):
                    samples, _ = noise_scheduler.clipping_reverse_step(
                    denoiser.get_numerator(), beta_t, C, t, samples, i)
                samples, _ = noise_scheduler.edm_stochastic_step(
                    residual3, t, sigmas_for_inference[i+1], samples, i)
            else:
                raise NotImplementedError
            
            # Save the one-step-empirical denoiser
            if (i == t_switch + 1):
                one_step_samples.append(samples.clone().detach().cpu())
        
            if first_run:
                timesteps.append(t)
                sample_transformation.append(samples.clone().detach().cpu()[:num_denoise])

    all_samples.append(samples.clone().detach().cpu())
    first_run = False

    sample_tensor = torch.cat(all_samples, dim=0)
    one_step_samples_tensor = torch.cat(one_step_samples, dim = 0)

    return {
        "samples": sample_tensor,
        "sample_transformation": sample_transformation,
        "timesteps": timesteps,
        "t_switch": t_switch,
        "one_step_samples": one_step_samples_tensor,
    }

def denoising_step_double_switch(t1, 
                                t2, 
                                model1, 
                                model2,
                                sampler_type,
                                sigmas_for_inference,
                                noise_scheduler,
                                sigma_max = 80.0,
                                dimen = 2,
                                num_samples = 1000, 
                                device = 'cuda',
                                empirical_toggle = False,
                                C = 1.0):
    """Run denoising while switching from model1 -> model2 at index t1, and back to model1 at index t2."""
    # Basic validation: require two distinct switch points in range
    if t2 <= t1:
        raise ValueError("t2 must be greater than t1 for a meaningful double switch")

    # Store all samples
    all_samples = []

    # Visualization data
    sample_transformation = []
    timesteps = []
    num_denoise = min(5, num_samples)
    first_run = True
    samples = torch.randn([num_samples, dimen], device=device) * sigma_max
    denoiser = model2 if empirical_toggle else model1
    # Ensure the same noise
    # torch.manual_seed(42)
    # torch.cuda.manual_seed_all(42)

    # Track whether we've performed each switch so it only happens once
    switched_first = False
    switched_second = False

    for i, t in enumerate(tqdm(sigmas_for_inference)):
        if i < sigmas_for_inference.shape[0] - 1:
            # Switch to model2 at t1 (when hitting index t1+1 to match previous logic)
            if (not switched_first) and (i == t1 + 1):
                denoiser = model2 if denoiser is model1 else model1
                switched_first = True
                empirical_toggle = not empirical_toggle

            # Switch back to model1 at t2
            if (not switched_second) and (i == t2 + 1):
                denoiser = model1 if denoiser is model2 else model2
                switched_second = True
                empirical_toggle = not empirical_toggle

            with torch.no_grad():
                    residual3, beta_t = denoiser.denoise(samples, t, i=i)
            if sampler_type == 'ode':
                samples, _ = noise_scheduler.euler_step(residual3, t, samples, i)
            elif sampler_type == 'sde':
                samples, _ = noise_scheduler.euler_maruyama_step(residual3, t, samples, i)
            elif sampler_type == 'clipsde':
                if (empirical_toggle):
                    samples, _ = noise_scheduler.clipping_reverse_step(
                    denoiser.get_numerator(), beta_t, C, t, samples, i)
                else:
                    samples, _ = noise_scheduler.euler_maruyama_step(residual3, t, samples, i)
            else:
                raise NotImplementedError

            if first_run:
                timesteps.append(t)
                sample_transformation.append(samples.clone().detach().cpu()[:num_denoise])

    all_samples.append(samples.clone().detach().cpu())
    first_run = False

    sample_tensor = torch.cat(all_samples, dim=0)

    return {
        "samples": sample_tensor,
        "sample_transformation": sample_transformation,
        "timesteps": timesteps,
        "t1": t1,
        "t2": t2
    }

# Standard Denoising for Testing
def denoising_step_standard(model,
                            sampler_type,
                            sigmas_for_inference,
                            noise_scheduler,
                            sigma_max = 80.0,
                            dimen = 2,
                            num_samples = 1000, 
                            device = 'cuda',
                            empirical_toggle = False,
                            C = 1.0):
    """Run a standard denoising pass using a single model (no switching).

    Returns a dict with keys: samples, sample_transformation, timesteps.
    """
    # Store all samples
    all_samples = []

    # Visualization data
    sample_transformation = []
    timesteps = []
    num_denoise = min(5, num_samples)
    first_run = True

    # Initialize samples and denoiser
    samples = torch.randn([num_samples, dimen], device=device) * sigma_max
    denoiser = model

    # Ensure deterministic noise
    # torch.manual_seed(42)
    # torch.cuda.manual_seed_all(42)

    for i, t in enumerate(tqdm(sigmas_for_inference)):
        if i < sigmas_for_inference.shape[0] - 1:
            with torch.no_grad():
                residual3, beta_t = denoiser.denoise(samples, t, i=i)

            if sampler_type == 'ode':
                samples, _ = noise_scheduler.euler_step(residual3, t, samples, i)
            elif sampler_type == 'sde':
                samples, _ = noise_scheduler.euler_maruyama_step(residual3, t, samples, i)
            elif sampler_type == 'clipsde':
                if (empirical_toggle):
                    samples, _ = noise_scheduler.clipping_reverse_step(
                        denoiser.get_numerator(), beta_t, C, t, samples, i)
                else:
                    samples, _ = noise_scheduler.euler_maruyama_step(residual3, t, samples, i)
            else:
                raise NotImplementedError

            if first_run:
                timesteps.append(t)
                sample_transformation.append(samples.clone().detach().cpu()[:num_denoise])

    all_samples.append(samples.clone().detach().cpu())
    first_run = False

    sample_tensor = torch.cat(all_samples, dim=0)

    return {
        "samples": sample_tensor,
        "sample_transformation": sample_transformation,
        "timesteps": timesteps
    }
    
    
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
    model_precond.load_state_dict(model_pretrained_weights, strict = False)
    model_precond.eval()
    
    return ImageNet32NeuralDenoiser(model_precond, device=device)

def load_model_from_pickle(file_path, device):
    with dnnlib.util.open_url(file_path, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)
    return net

def get_imagenet64_neural_denoiser(model_path, device='cuda'):
    return load_model_from_pickle(model_path, device)

def generate_samples_hybrid_model_row_subsampling(
    full_dataset, 
    data_loader,
    t_switch,
    pmodel_imagenet,
    num_samples=10000, 
    samples_per_seed=1000, 
    dataset_name="cifar10",
    sigma_min=0.002, 
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
        t_switch: Timestep index to switch denoisers
        pmodel_imagenet: Pretrained neural denoiser model
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
    all_generated_one_step_samples = []
    num_denoise_samples = min(5, samples_per_seed)
    sample_transformation = []
    timesteps = []
    first_batch = True
    
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

        generator = torch.Generator()
        generator.manual_seed(seed_value)
        
        # Get the full dataset with shuffle = True
        seed_data_loader, n_samples = get_cifar10_dataset(n = batch_size, shuffle = True, generator = generator)
        
        subsampled_dataset, _ = next(iter(seed_data_loader))
        subsampled_dataset = subsampled_dataset.to(device)
        
        # create a denoiser with the subsampled dataset
        curr_denoiser = EmpiricalDenoiser(
            subsampled_dataset,
            data_loader=seed_data_loader,
            clipping=(sampler_type == "clipsde"), 
            n_dim=dimen, 
            C=C,
            non_dp=True,  # use non-DP version
            device=device
        )
        samples_for_this_seed = min(samples_per_seed, num_samples - sum(len(batch) for batch in all_generated_samples))
        
        seed_samples = []
        seed_samples_one_time = []
        num_batches = (samples_for_this_seed + comp_batch_size - 1) // comp_batch_size
        
        for batch_idx in range(num_batches):
            if logger:
                logger.info(f"Seed {seed_value}, processing batch {batch_idx + 1}/{num_batches}")
            else:
                print(f"Seed {seed_value}, processing batch {batch_idx + 1}/{num_batches}")
            
            # Calculate batch size for this iteration
            current_batch_size = min(comp_batch_size, samples_for_this_seed - len(seed_samples))
            
            denoise_cfg = dict(C = config.C, 
                sigma_max = config.sigma_max, 
                noise_scheduler=noise_scheduler, 
                sigmas_for_inference=sigmas_for_inference, 
                sampler_type=config.sampler_type, 
                dimen = dimen, 
                num_samples = current_batch_size,
                device = device,)
            
            batch_results = denoising_step_hybrid(t_switch, pmodel_imagenet, curr_denoiser, empirical_toggle=False, **denoise_cfg)

            seed_samples.append(batch_results["samples"].clone().detach().cpu())
            seed_samples_one_time.append(batch_results["one_step_samples"].clone().detach().cpu())
            
            if (first_batch):
                timesteps = batch_results["timesteps"]
                sample_transformation = batch_results["sample_transformation"]
            
            if logger:
                logger.info(f"Generated {sum(batch.shape[0] for batch in seed_samples)}/{samples_for_this_seed} samples for seed {seed_value}")
            else:
                print(f"Generated {sum(batch.shape[0] for batch in seed_samples)}/{samples_for_this_seed} samples for seed {seed_value}")
        
        all_generated_samples.append(torch.cat(seed_samples, dim=0))
        all_generated_one_step_samples.append(torch.cat(seed_samples_one_time, dim = 0))
        first_batch = False
         
    sample_tensor = torch.cat(all_generated_samples, dim=0)[:num_samples]
    one_step_sample_tensor = torch.cat(all_generated_one_step_samples, dim=0)[:num_samples]
    
    if logger:
        logger.info(f"Final sample tensor shape: {sample_tensor.shape}")
    else:
        print(f"Final sample tensor shape: {sample_tensor.shape}")
    
    if save:
        sample_name = f"samples_{dataset_name}_{experiment_name}_{sampler_type}.pt"
        torch.save({"samples": sample_tensor}, os.path.join(samplesdir, sample_name))
        
        sample_one_step_name = f"samples_one_step_{dataset_name}_{experiment_name}_{sampler_type}.pt"
        torch.save({"samples": one_step_sample_tensor}, os.path.join(samplesdir, sample_one_step_name))
        
        if logger:
            logger.info(f"Samples saved to {os.path.join(samplesdir, sample_name)}")
            logger.info(f"One Step Samples saved to {os.path.join(samplesdir, sample_one_step_name)}")
        else:
            print(f"Samples saved to {os.path.join(samplesdir, sample_name)}")
    
    # Visualize denoising process if we have tracking data
    if sample_transformation and dataset_name in ["mnist", "cifar10"]:
        if dataset_name == "mnist":
            # Get indices for 10 evenly spaced elements
            proportion_indices = np.linspace(0, num_timesteps - 1, 10 + 1, dtype=int)

            sample_transformation_short = [sample_transformation[idx] for i, idx in enumerate(proportion_indices)]
            timesteps_short = [timesteps[idx] for i, idx in enumerate(proportion_indices)]
            
            plot_diffusion_grid(
                sample_transformation_short, 
                timesteps_short, 
                dimen=(28, 28), 
                imgname=f"denoiser_{sampler_type}_{experiment_name}",
                imgdir=imgdir, 
                sigma_max=sigma_max, 
                sigma_min=sigma_min
            )
        elif dataset_name == "cifar10":
             # Get indices for 10 evenly spaced elements
            proportion_indices = np.linspace(0, num_timesteps - 1, 10 + 1, dtype=int)

            sample_transformation_short = [sample_transformation[idx] for i, idx in enumerate(proportion_indices)]
            timesteps_short = [timesteps[idx] for i, idx in enumerate(proportion_indices)]

            plot_diffusion_grid(sample_transformation_short, 
                                timesteps_short, 
                                rgb = True, 
                                dimen=(32, 32), 
                                imgname=f"denoiser_{config.sampler_type}_{config.experiment_name}", 
                                imgdir=imgdir, 
                                sigma_max=config.sigma_max, 
                                sigma_min=config.sigma_min)

    if dataset_name == "cifar10":
        max_vis_samples = min(100, num_samples)
        plot_rgb_grid(sample_tensor[:max_vis_samples], dimen=(32, 32), grid_size=(10, 10),
                          imgname=f"cifar_samples_{sampler_type}_{experiment_name}", imgdir=imgdir,
                          sigma_min=sigma_min, sigma_max=sigma_max)
        plot_rgb_grid(one_step_sample_tensor[:max_vis_samples], dimen=(32, 32), grid_size=(10, 10),
                          imgname=f"cifar_samples_one_step_{sampler_type}_{experiment_name}", imgdir=imgdir,
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
        "one_step_sample": one_step_sample_tensor,
    }

def generate_samples_empirical_then_neural_row_subsampling(
    full_dataset,
    data_loader,
    t_switch,
    pmodel_imagenet,
    num_samples=10000,
    samples_per_seed=1000,
    dataset_name="cifar10",
    sigma_min=0.002,
    sigma_max=80.0,
    num_timesteps=50,
    sampler_type="sde",
    C=1.0,
    rho=7.0,
    save=False,
    experiment_name="empirical_first_row_subsampled",
    samplesdir="./samples",
    imgdir="./images_app",
    logger=None,
    beta_schedule=None,
    comp_batch_size=100,
    config=None,
    device=None,
    full_dataset_mode=False,
):
    """
    Generates samples using empirical denoiser for the first t_switch steps,
    then switches to the neural denoiser for the remaining steps.

    This is the reverse of generate_samples_hybrid_model_row_subsampling, which
    runs neural first and empirical second.

    Args:
        full_dataset: The complete dataset tensor
        data_loader: DataLoader for getting different subsets of data
        t_switch: Timestep index to switch from empirical to neural denoiser
        pmodel_imagenet: Pretrained neural denoiser model (used in second phase)
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
        comp_batch_size: Computational batch size for generation (memory management)

    Returns:
        Generated samples tensor
    """
    if logger:
        logger.info(f"Generating {num_samples} samples using empirical-first row-wise subsampling")
    else:
        print(f"Generating {num_samples} samples using empirical-first row-wise subsampling")

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
    all_generated_one_step_samples = []
    num_denoise_samples = min(5, samples_per_seed)
    sample_transformation = []
    timesteps = []
    first_batch = True

    # ── Full-dataset mode: single denoiser that sweeps all training data ──────
    if full_dataset_mode:
        full_data_loader, _ = get_cifar10_dataset(n=comp_batch_size, shuffle=False)
        init_batch, _ = next(iter(full_data_loader))
        curr_denoiser = EmpiricalDenoiser(
            init_batch.to(device),
            data_loader=full_data_loader,
            clipping=(sampler_type == "clipsde"),
            n_dim=dimen,
            C=C,
            beta_hyper=config.beta_hyper if config is not None else None,
            non_dp=True,
            is_full_dataset=True,
            device=device,
        )

        denoise_cfg = dict(
            C=C,
            sigma_max=sigma_max,
            noise_scheduler=noise_scheduler,
            sigmas_for_inference=sigmas_for_inference,
            sampler_type=sampler_type,
            dimen=dimen,
            num_samples=num_samples,
            device=device,
        )
        batch_results = denoising_step_hybrid(
            t_switch, curr_denoiser, pmodel_imagenet, empirical_toggle=True, **denoise_cfg
        )
        sample_tensor = batch_results["samples"].clone().detach().cpu()
        one_step_sample_tensor = batch_results["one_step_samples"][0].clone().detach().cpu() if batch_results["one_step_samples"] else sample_tensor
        sample_transformation = batch_results["sample_transformation"]
        timesteps = batch_results["timesteps"]

        if save:
            sample_name = f"samples_{dataset_name}_{experiment_name}_{sampler_type}.pt"
            torch.save({"samples": sample_tensor}, os.path.join(samplesdir, sample_name))
            if logger:
                logger.info(f"Samples saved to {os.path.join(samplesdir, sample_name)}")
            else:
                print(f"Samples saved to {os.path.join(samplesdir, sample_name)}")

        if dataset_name == "cifar10":
            max_vis_samples = min(100, num_samples)
            plot_rgb_grid(sample_tensor[:max_vis_samples], dimen=(32, 32), grid_size=(10, 10),
                          imgname=f"cifar_samples_{sampler_type}_{experiment_name}", imgdir=imgdir,
                          sigma_min=sigma_min, sigma_max=sigma_max)

        return {"samples": sample_tensor}

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

        generator = torch.Generator()
        generator.manual_seed(seed_value)

        seed_data_loader, n_samples = get_cifar10_dataset(n=comp_batch_size, shuffle=True, generator=generator)

        subsampled_dataset, _ = next(iter(seed_data_loader))
        subsampled_dataset = subsampled_dataset.to(device)

        curr_denoiser = EmpiricalDenoiser(
            subsampled_dataset,
            data_loader=seed_data_loader,
            clipping=(sampler_type == "clipsde"),
            n_dim=dimen,
            C=C,
            non_dp=True,
            device=device
        )
        samples_for_this_seed = min(samples_per_seed, num_samples - sum(len(batch) for batch in all_generated_samples))

        seed_samples = []
        seed_samples_one_time = []
        num_batches = (samples_for_this_seed + comp_batch_size - 1) // comp_batch_size

        for batch_idx in range(num_batches):
            if logger:
                logger.info(f"Seed {seed_value}, processing batch {batch_idx + 1}/{num_batches}")
            else:
                print(f"Seed {seed_value}, processing batch {batch_idx + 1}/{num_batches}")

            current_batch_size = min(comp_batch_size, samples_for_this_seed - len(seed_samples))

            denoise_cfg = dict(C=config.C,
                sigma_max=config.sigma_max,
                noise_scheduler=noise_scheduler,
                sigmas_for_inference=sigmas_for_inference,
                sampler_type=config.sampler_type,
                dimen=dimen,
                num_samples=current_batch_size,
                device=device,)

            # Empirical denoiser is model1 (runs first), neural is model2 (runs after t_switch).
            # empirical_toggle=True activates clipping sampler logic from the start.
            batch_results = denoising_step_hybrid(t_switch, curr_denoiser, pmodel_imagenet, empirical_toggle=True, **denoise_cfg)

            seed_samples.append(batch_results["samples"].clone().detach().cpu())
            seed_samples_one_time.append(batch_results["one_step_samples"].clone().detach().cpu())

            if first_batch:
                timesteps = batch_results["timesteps"]
                sample_transformation = batch_results["sample_transformation"]

            if logger:
                logger.info(f"Generated {sum(batch.shape[0] for batch in seed_samples)}/{samples_for_this_seed} samples for seed {seed_value}")
            else:
                print(f"Generated {sum(batch.shape[0] for batch in seed_samples)}/{samples_for_this_seed} samples for seed {seed_value}")

        all_generated_samples.append(torch.cat(seed_samples, dim=0))
        all_generated_one_step_samples.append(torch.cat(seed_samples_one_time, dim=0))
        first_batch = False

    sample_tensor = torch.cat(all_generated_samples, dim=0)[:num_samples]
    one_step_sample_tensor = torch.cat(all_generated_one_step_samples, dim=0)[:num_samples]

    if logger:
        logger.info(f"Final sample tensor shape: {sample_tensor.shape}")
    else:
        print(f"Final sample tensor shape: {sample_tensor.shape}")

    if save:
        sample_name = f"samples_{dataset_name}_{experiment_name}_{sampler_type}.pt"
        torch.save({"samples": sample_tensor}, os.path.join(samplesdir, sample_name))

        sample_one_step_name = f"samples_one_step_{dataset_name}_{experiment_name}_{sampler_type}.pt"
        torch.save({"samples": one_step_sample_tensor}, os.path.join(samplesdir, sample_one_step_name))

        if logger:
            logger.info(f"Samples saved to {os.path.join(samplesdir, sample_name)}")
            logger.info(f"One Step Samples saved to {os.path.join(samplesdir, sample_one_step_name)}")
        else:
            print(f"Samples saved to {os.path.join(samplesdir, sample_name)}")

    if sample_transformation and dataset_name in ["mnist", "cifar10"]:
        if dataset_name == "mnist":
            proportion_indices = np.linspace(0, num_timesteps - 1, 10 + 1, dtype=int)
            sample_transformation_short = [sample_transformation[idx] for i, idx in enumerate(proportion_indices)]
            timesteps_short = [timesteps[idx] for i, idx in enumerate(proportion_indices)]
            plot_diffusion_grid(
                sample_transformation_short,
                timesteps_short,
                dimen=(28, 28),
                imgname=f"denoiser_{sampler_type}_{experiment_name}",
                imgdir=imgdir,
                sigma_max=sigma_max,
                sigma_min=sigma_min
            )
        elif dataset_name == "cifar10":
            proportion_indices = np.linspace(0, num_timesteps - 1, 10 + 1, dtype=int)
            sample_transformation_short = [sample_transformation[idx] for i, idx in enumerate(proportion_indices)]
            timesteps_short = [timesteps[idx] for i, idx in enumerate(proportion_indices)]
            plot_diffusion_grid(sample_transformation_short,
                                timesteps_short,
                                rgb=True,
                                dimen=(32, 32),
                                imgname=f"denoiser_{config.sampler_type}_{config.experiment_name}",
                                imgdir=imgdir,
                                sigma_max=config.sigma_max,
                                sigma_min=config.sigma_min)

    if dataset_name == "cifar10":
        max_vis_samples = min(100, num_samples)
        plot_rgb_grid(sample_tensor[:max_vis_samples], dimen=(32, 32), grid_size=(10, 10),
                      imgname=f"cifar_samples_{sampler_type}_{experiment_name}", imgdir=imgdir,
                      sigma_min=sigma_min, sigma_max=sigma_max)
        plot_rgb_grid(one_step_sample_tensor[:max_vis_samples], dimen=(32, 32), grid_size=(10, 10),
                      imgname=f"cifar_samples_one_step_{sampler_type}_{experiment_name}", imgdir=imgdir,
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
        "one_step_sample": one_step_sample_tensor,
    }

def generate_samples_empirical_per_sample_subsampling(
    full_dataset,
    data_loader,
    num_samples=10000,
    samples_per_seed=1000,
    dataset_name="cifar10",
    sigma_min=0.002,
    sigma_max=80.0,
    num_timesteps=50,
    sampler_type="sde",
    C=1.0,
    rho=7.0,
    save=False,
    experiment_name="empirical_per_sample_subsampled",
    samplesdir="./samples",
    imgdir="./images_app",
    logger=None,
    beta_schedule=None,
    comp_batch_size=1,
    config=None,
    device=None,
):
    """Pure empirical denoising with one subsample drawn per generated sample.

    Implements the "B = 1" mechanism from the privacy-accounting note: for each
    generated sample, a minibatch of size ``config.batch_size`` is drawn with
    replacement (via the shuffled DataLoader) once at the start of the trajectory
    and reused for all ``num_timesteps`` denoising steps.  No neural denoiser is
    used — the empirical denoiser handles every step.

    Setting ``comp_batch_size=1`` is the privacy-relevant default (one independent
    subsample per generated sample).  Larger ``comp_batch_size`` shares a
    subsample across that many samples for GPU efficiency, which corresponds to
    a different (and weaker) privacy claim.

    Args:
        full_dataset:    Complete dataset tensor (used only for shape / dimen).
        data_loader:     DataLoader; per-seed it is rebuilt with ``shuffle=True``.
        comp_batch_size: How many samples share a subsample.  Default 1.
        ...

    Returns:
        dict with keys: samples, sample_transformation, timesteps.
    """
    if logger:
        logger.info(
            f"Generating {num_samples} samples with per-sample row subsampling "
            f"(empirical-only, comp_batch_size={comp_batch_size})"
        )
    else:
        print(
            f"Generating {num_samples} samples with per-sample row subsampling "
            f"(empirical-only, comp_batch_size={comp_batch_size})"
        )

    num_seeds = (num_samples + samples_per_seed - 1) // samples_per_seed
    dimen = full_dataset.shape[1]

    noise_scheduler = NoiseScheduler(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_timesteps=num_timesteps,
    )

    sigmas_for_inference = noise_scheduler.get_sigmas_karras(
        num_timesteps,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        rho=rho,
    )

    all_generated_samples = []
    sample_transformation = []
    timesteps = []
    first_batch = True

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

        generator = torch.Generator()
        generator.manual_seed(seed_value)

        # Shuffled loader; each call to next(iter(loader)) yields a fresh random
        # minibatch — this is sampling with replacement across calls.
        seed_data_loader, _ = get_cifar10_dataset(n=config.batch_size, shuffle=True, generator=generator)

        # Initialize RowSampleEmpiricalDenoiser with any first batch; we refresh
        # the subsample explicitly via get_subsample() once per computational batch.
        init_subsample, _ = next(iter(seed_data_loader))
        init_subsample = init_subsample.to(device)
        curr_denoiser = RowSampleEmpiricalDenoiser(
            init_subsample,
            data_loader=seed_data_loader,
            clipping=(sampler_type == "clipsde"),
            n_dim=dimen,
            C=C,
            beta_schedule=beta_schedule,
            non_dp=True,
            device=device,
        )

        samples_for_this_seed = min(
            samples_per_seed,
            num_samples - sum(len(batch) for batch in all_generated_samples),
        )

        seed_samples = []
        num_batches = (samples_for_this_seed + comp_batch_size - 1) // comp_batch_size

        for batch_idx in range(num_batches):
            if logger:
                logger.info(f"Seed {seed_value}, processing batch {batch_idx + 1}/{num_batches}")
            else:
                print(f"Seed {seed_value}, processing batch {batch_idx + 1}/{num_batches}")

            current_batch_size = min(comp_batch_size, samples_for_this_seed - len(seed_samples))

            # Refresh the subsample once for this batch; with comp_batch_size=1
            # this is exactly once per generated sample.
            curr_denoiser.get_subsample()

            batch_results = denoising_step_standard(
                model=curr_denoiser,
                sampler_type=sampler_type,
                sigmas_for_inference=sigmas_for_inference,
                noise_scheduler=noise_scheduler,
                sigma_max=sigma_max,
                dimen=dimen,
                num_samples=current_batch_size,
                device=device,
                empirical_toggle=True,
                C=C,
            )

            seed_samples.append(batch_results["samples"].clone().detach().cpu())

            if first_batch:
                timesteps = batch_results["timesteps"]
                sample_transformation = batch_results["sample_transformation"]

            if logger:
                logger.info(
                    f"Generated {sum(b.shape[0] for b in seed_samples)}/{samples_for_this_seed} "
                    f"samples for seed {seed_value}"
                )
            else:
                print(
                    f"Generated {sum(b.shape[0] for b in seed_samples)}/{samples_for_this_seed} "
                    f"samples for seed {seed_value}"
                )

        all_generated_samples.append(torch.cat(seed_samples, dim=0))
        first_batch = False

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

    if sample_transformation and dataset_name in ["mnist", "cifar10"]:
        proportion_indices = np.linspace(0, num_timesteps - 1, 10 + 1, dtype=int)
        sample_transformation_short = [sample_transformation[idx] for idx in proportion_indices]
        timesteps_short = [timesteps[idx] for idx in proportion_indices]
        if dataset_name == "mnist":
            plot_diffusion_grid(
                sample_transformation_short,
                timesteps_short,
                dimen=(28, 28),
                imgname=f"denoiser_{sampler_type}_{experiment_name}",
                imgdir=imgdir,
                sigma_max=sigma_max,
                sigma_min=sigma_min,
            )
        elif dataset_name == "cifar10":
            plot_diffusion_grid(
                sample_transformation_short,
                timesteps_short,
                rgb=True,
                dimen=(32, 32),
                imgname=f"denoiser_{sampler_type}_{experiment_name}",
                imgdir=imgdir,
                sigma_max=sigma_max,
                sigma_min=sigma_min,
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


def generate_samples_empirical_first_neural_later_64(
    full_dataset,
    data_loader,
    t_switch,
    pmodel_imagenet,
    num_samples=10000,
    samples_per_seed=1000,
    sigma_min=0.002,
    sigma_max=80.0,
    num_timesteps=50,
    empirical_sampler_type="sde",
    neural_sampler_type="edm",
    C=1.0,
    rho=7.0,
    save=False,
    experiment_name="empirical_first_64",
    samplesdir="./samples",
    imgdir="./images_app",
    logger=None,
    config=None,
    device=None,
    comp_batch_size=100,
    neural_batch_size=None,
    without_replacement=False,
    full_dataset_mode=False,
):
    """Two-phase hybrid pipeline: empirical denoiser on upscaled 64x64 CIFAR10 for the
    first t_switch steps, then the ImageNet-64 neural denoiser for the remaining steps.
    Both phases operate entirely in 64x64 space so there is no mid-trajectory dimension
    bridge. The final 64x64 samples are downsampled to 32x32 at the end.

    Args:
        full_dataset: CIFAR10 dataset tensor, shape (N, 3072).
        data_loader: DataLoader for row-wise subsampling of the 32x32 CIFAR10 data.
        t_switch: Timestep index at which the switch from empirical to neural occurs.
        pmodel_imagenet: ImageNet64NeuralDenoiser instance.
        num_samples: Total samples to generate.
        samples_per_seed: Samples sharing the same empirical denoiser.
        empirical_sampler_type: Sampler for phase 1 ('sde', 'ode', 'clipsde').
        neural_sampler_type: Sampler for phase 2 (must be 'edm').
        comp_batch_size: Computational batch size.
        without_replacement: If True, use BatchCyclingEmpiricalDenoiser (no-replacement
            subsampling). Ignored when full_dataset_mode=True.
        full_dataset_mode: If True, use the base EmpiricalDenoiser with
            is_full_dataset=True so each denoise call sweeps the entire 64x64 CIFAR10
            dataset via the data_loader (batched) instead of subsampling.
    """
    dimen_64 = 64 * 64 * 3  # 12288

    noise_scheduler = NoiseScheduler(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_timesteps=num_timesteps,
    )
    sigmas_for_inference = noise_scheduler.get_sigmas_karras(
        num_timesteps,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        rho=rho,
    )

    # Upsample the full CIFAR10 dataset from 32x32 to 64x64 once upfront so
    # the empirical denoiser operates in the same space as the neural model.
    full_dataset_64 = F.interpolate(
        full_dataset.view(-1, 3, 32, 32).float().to(device),
        size=(64, 64),
        mode='bilinear',
        align_corners=False,
    ).view(-1, dimen_64)

    all_empirical_samples = []
    num_track = min(5, num_samples)
    empirical_transformation = []
    empirical_timesteps = []

    # ── prop=0.0 fast path: skip empirical entirely, run all-neural ───────────
    if t_switch == 0:
        if logger:
            logger.info("prop=0.0: skipping empirical phase, starting neural from fresh noise")
        else:
            print("prop=0.0: skipping empirical phase, starting neural from fresh noise")
        empirical_samples_64 = (torch.randn(num_samples, dimen_64) * sigma_max).cpu()
        neural_t_switch = -1
    else:
        neural_t_switch = t_switch

    # ── Phase 1: empirical denoiser on 64×64 CIFAR10, steps 0 → t_switch ─────
    if t_switch > 0:
        # 64×64 data_loader so SubSample/BatchCycling denoisers get correctly shaped batches.
        data_loader_64, _ = get_cifar10_dataset_64(n=comp_batch_size, shuffle=True)
        init_batch, _ = next(iter(data_loader_64))
        init_batch = init_batch.to(device)

        if full_dataset_mode:
            curr_denoiser = EmpiricalDenoiser(
                init_batch,
                data_loader=data_loader_64,
                clipping=(empirical_sampler_type == "clipsde"),
                n_dim=dimen_64,
                C=C,
                beta_hyper=config.beta_hyper,
                non_dp=False,
                is_full_dataset=True,
                device=device,
            )
        elif without_replacement:
            curr_denoiser = BatchCyclingEmpiricalDenoiser(
                init_batch,
                data_loader=data_loader_64,
                clipping=(empirical_sampler_type == "clipsde"),
                n_dim=dimen_64,
                C=C,
                beta_hyper=config.beta_hyper,
                non_dp=False,
                device=device,
            )
        else:
            curr_denoiser = SubSampleEmpiricalDenoiser(
                init_batch,
                data_loader=data_loader_64,
                clipping=(empirical_sampler_type == "clipsde"),
                n_dim=dimen_64,
                C=C,
                beta_hyper=config.beta_hyper,
                non_dp=False,
                device=device,
            )

        samples_all = torch.randn([num_samples, dimen_64], device=device) * sigma_max
        track_buf = {}  # timestep index -> (sigma_t, list of [B, dimen_64] cpu tensors)

        if full_dataset_mode:
            # Full-dataset mode: every query sees the same dataset, so we can batch
            # queries together rather than looping per sample.
            query_batch = comp_batch_size
            for start in range(0, num_samples, query_batch):
                end = min(start + query_batch, num_samples)
                sample = samples_all[start:end].clone()

                with torch.no_grad():
                    for i, t in enumerate(sigmas_for_inference):
                        if i > t_switch:
                            break
                        if i < sigmas_for_inference.shape[0] - 1:
                            residual, beta_t = curr_denoiser.denoise(sample, t, i=i)
                            if empirical_sampler_type == 'ode':
                                sample, _ = noise_scheduler.euler_step(residual, t, sample, i)
                            elif empirical_sampler_type == 'sde':
                                sample, _ = noise_scheduler.euler_maruyama_step(residual, t, sample, i)
                            elif empirical_sampler_type == 'clipsde':
                                sample, _ = noise_scheduler.clipping_reverse_step(
                                    curr_denoiser.get_numerator(), beta_t, C, t, sample, i)
                            else:
                                raise NotImplementedError(
                                    f"Sampler '{empirical_sampler_type}' not supported for empirical phase")

                            # Track the first num_track samples across batches.
                            if start < num_track:
                                track_end = min(end, num_track) - start
                                if i not in track_buf:
                                    track_buf[i] = (t, [])
                                track_buf[i][1].append(sample[:track_end].clone().detach().cpu())

                samples_all[start:end] = sample

                if logger:
                    logger.info(f"[Phase 1 full-batch] {end}/{num_samples} samples done")
                else:
                    print(f"[Phase 1 full-batch] {end}/{num_samples} samples done")
        else:
            for sample_idx in range(num_samples):
                sample = samples_all[sample_idx:sample_idx + 1].clone()

                with torch.no_grad():
                    for i, t in enumerate(sigmas_for_inference):
                        if i > t_switch:
                            break
                        if i < sigmas_for_inference.shape[0] - 1:
                            residual, beta_t = curr_denoiser.denoise(sample, t, i=i)
                            if empirical_sampler_type == 'ode':
                                sample, _ = noise_scheduler.euler_step(residual, t, sample, i)
                            elif empirical_sampler_type == 'sde':
                                sample, _ = noise_scheduler.euler_maruyama_step(residual, t, sample, i)
                            elif empirical_sampler_type == 'clipsde':
                                sample, _ = noise_scheduler.clipping_reverse_step(
                                    curr_denoiser.get_numerator(), beta_t, C, t, sample, i)
                            else:
                                raise NotImplementedError(
                                    f"Sampler '{empirical_sampler_type}' not supported for empirical phase")

                            if sample_idx < num_track:
                                if i not in track_buf:
                                    track_buf[i] = (t, [])
                                track_buf[i][1].append(sample.clone().detach().cpu())

                samples_all[sample_idx] = sample[0]

                if logger:
                    logger.info(f"[Phase 1] {sample_idx + 1}/{num_samples} samples done")
                else:
                    print(f"[Phase 1] {sample_idx + 1}/{num_samples} samples done")

        empirical_samples_64 = samples_all.cpu()

        for i_key in sorted(track_buf.keys()):
            t_val, tensors = track_buf[i_key]
            empirical_transformation.append(torch.cat(tensors, dim=0))
            empirical_timesteps.append(t_val)

    # ── Phase 2: neural denoiser on 64×64, steps neural_t_switch+1 → end ─────
    if logger:
        logger.info(f"[Phase 2] Neural denoiser from step {neural_t_switch + 1} onward")
    else:
        print(f"[Phase 2] Neural denoiser from step {neural_t_switch + 1} onward")

    neural_result = sample_neural_denoiser_later_steps(
        denoiser=pmodel_imagenet,
        t_switch=neural_t_switch,
        early_stop_samples=empirical_samples_64,
        dimen=dimen_64,
        batch_size=neural_batch_size if neural_batch_size is not None else comp_batch_size,
        num_samples=num_samples,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_timesteps=num_timesteps,
        sampler_type=neural_sampler_type,
        rho=rho,
        device=device,
        logger=logger,
        num_track=num_track,
    )

    # ── Downsample 64×64 → 32×32 at the very end ─────────────────────────────
    sample_tensor_64 = neural_result["samples"].clamp(-1, 1)
    sample_tensor_32 = convert_64_to_32(sample_tensor_64.to(device)).cpu()

    if save:
        sample_name = f"samples_cifar10_{experiment_name}_{neural_sampler_type}.pt"
        torch.save({"samples": sample_tensor_32}, os.path.join(samplesdir, sample_name))
        if logger:
            logger.info(f"Samples saved to {os.path.join(samplesdir, sample_name)}")
        else:
            print(f"Samples saved to {os.path.join(samplesdir, sample_name)}")

    max_vis_samples = min(100, num_samples)
    plot_rgb_grid(
        sample_tensor_32[:max_vis_samples],
        dimen=(32, 32), grid_size=(10, 10),
        imgname=f"cifar_samples_{neural_sampler_type}_{experiment_name}",
        imgdir=imgdir,
        sigma_min=sigma_min, sigma_max=sigma_max,
    )

    # ── Diffusion grid: full trajectory (empirical phase + neural phase) ──────
    neural_transformation = neural_result["sample_transformation"]  # list of [num_track, dimen_64]
    neural_timesteps = neural_result["timesteps"]

    # Combine both phases into one ordered timeline.
    all_transformation_64 = empirical_transformation + neural_transformation
    all_timesteps = empirical_timesteps + neural_timesteps

    if all_transformation_64:
        # Downsample each tracked state from 64x64 to 32x32 for plot_diffusion_grid.
        all_transformation_32 = [
            convert_64_to_32(s.to(device)).cpu() for s in all_transformation_64
        ]

        # Select ~10 evenly spaced steps across the full trajectory.
        n_total = len(all_transformation_32)
        indices = np.linspace(0, n_total - 1, min(11, n_total), dtype=int)
        transformation_short = [all_transformation_32[idx] for idx in indices]
        timesteps_short = [all_timesteps[idx] for idx in indices]

        plot_diffusion_grid(
            transformation_short,
            timesteps_short,
            rgb=True,
            dimen=(32, 32),
            imgname=f"diffusion_grid_{experiment_name}",
            imgdir=imgdir,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
        )

    return {
        "samples": sample_tensor_32,
    }


def generate_samples_sdedit_empirical_neural_64(
    full_dataset,
    data_loader,
    t_switch,
    noise_steps,
    pmodel_imagenet,
    num_samples=10000,
    samples_per_seed=1000,
    sigma_min=0.002,
    sigma_max=80.0,
    num_timesteps=50,
    empirical_sampler_type="sde",
    neural_sampler_type="edm",
    C=1.0,
    rho=7.0,
    save=False,
    experiment_name="sdedit_empirical_neural_64",
    samplesdir="./samples",
    imgdir="./images_app",
    logger=None,
    config=None,
    device=None,
    comp_batch_size=100,
    neural_batch_size=None,
    without_replacement=False,
    full_dataset_mode=False,
    row_subsample=False,
):
    """SDEdit-style hybrid pipeline on 64x64 space.

    Phase 1: empirical denoiser from step 0 → t_switch (high σ to low σ).
    Bridge:  add Gaussian noise to the intermediate state to push it back by
             noise_steps positions in the Karras σ schedule (SDEdit re-noising).
    Phase 2: ImageNet-64 neural denoiser from (t_switch - noise_steps) → end.

    The re-noising preserves the rough structure produced by the empirical
    denoiser while handing off to the neural model at a higher noise level,
    giving it more room to improve sample quality.

    Args:
        t_switch: Step index where the empirical denoiser stops.  Empirical
            runs steps i = 0 … t_switch (inclusive).
        noise_steps: How many Karras-schedule steps to re-noise backward after
            the empirical phase.  0 → identical to generate_samples_empirical_
            first_neural_later_64.  Must satisfy 0 <= noise_steps <= t_switch.
        Other args: same as generate_samples_empirical_first_neural_later_64.
    """
    dimen_64 = 64 * 64 * 3

    noise_scheduler = NoiseScheduler(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_timesteps=num_timesteps,
    )
    sigmas_for_inference = noise_scheduler.get_sigmas_karras(
        num_timesteps,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        rho=rho,
    )

    full_dataset_64 = F.interpolate(
        full_dataset.view(-1, 3, 32, 32).float().to(device),
        size=(64, 64),
        mode='bilinear',
        align_corners=False,
    ).view(-1, dimen_64)

    num_track = min(5, num_samples)
    empirical_transformation = []
    empirical_timesteps = []

    # ── prop=0.0 fast path ────────────────────────────────────────────────────
    if t_switch == 0:
        if logger:
            logger.info("t_switch=0: skipping empirical phase, running all-neural")
        else:
            print("t_switch=0: skipping empirical phase, running all-neural")
        empirical_samples_64 = (torch.randn(num_samples, dimen_64) * sigma_max).cpu()
        neural_start = -1

    # ── Phase 1: empirical denoiser on 64×64, steps 0 → t_switch ─────────────
    else:
        data_loader_64, _ = get_cifar10_dataset_64(n=comp_batch_size, shuffle=True)
        init_batch, _ = next(iter(data_loader_64))
        init_batch = init_batch.to(device)

        if full_dataset_mode:
            curr_denoiser = EmpiricalDenoiser(
                init_batch,
                data_loader=data_loader_64,
                clipping=(empirical_sampler_type == "clipsde"),
                n_dim=dimen_64,
                C=C,
                beta_hyper=config.beta_hyper,
                non_dp=False,
                is_full_dataset=True,
                device=device,
            )
        elif without_replacement:
            curr_denoiser = BatchCyclingEmpiricalDenoiser(
                init_batch,
                data_loader=data_loader_64,
                clipping=(empirical_sampler_type == "clipsde"),
                n_dim=dimen_64,
                C=C,
                beta_hyper=config.beta_hyper,
                non_dp=False,
                device=device,
            )
        elif row_subsample:
            curr_denoiser = RowSampleEmpiricalDenoiser(
                init_batch,
                data_loader=data_loader_64,
                clipping=(empirical_sampler_type == "clipsde"),
                n_dim=dimen_64,
                C=C,
                beta_hyper=config.beta_hyper,
                non_dp=False,
                device=device,
            )
        else:
            curr_denoiser = SubSampleEmpiricalDenoiser(
                init_batch,
                data_loader=data_loader_64,
                clipping=(empirical_sampler_type == "clipsde"),
                n_dim=dimen_64,
                C=C,
                beta_hyper=config.beta_hyper,
                non_dp=False,
                device=device,
            )

        samples_all = torch.randn([num_samples, dimen_64], device=device) * sigma_max
        track_buf = {}

        if full_dataset_mode:
            query_batch = comp_batch_size
            for start in range(0, num_samples, query_batch):
                end = min(start + query_batch, num_samples)
                sample = samples_all[start:end].clone()

                with torch.no_grad():
                    for i, t in enumerate(sigmas_for_inference):
                        if i > t_switch:
                            break
                        if i < sigmas_for_inference.shape[0] - 1:
                            residual, beta_t = curr_denoiser.denoise(sample, t, i=i)
                            if empirical_sampler_type == 'ode':
                                sample, _ = noise_scheduler.euler_step(residual, t, sample, i)
                            elif empirical_sampler_type == 'sde':
                                sample, _ = noise_scheduler.euler_maruyama_step(residual, t, sample, i)
                            elif empirical_sampler_type == 'clipsde':
                                sample, _ = noise_scheduler.clipping_reverse_step(
                                    curr_denoiser.get_numerator(), beta_t, C, t, sample, i)
                            else:
                                raise NotImplementedError(
                                    f"Sampler '{empirical_sampler_type}' not supported for empirical phase")

                            if start < num_track:
                                track_end = min(end, num_track) - start
                                if i not in track_buf:
                                    track_buf[i] = (t, [])
                                track_buf[i][1].append(sample[:track_end].clone().detach().cpu())

                samples_all[start:end] = sample

                if logger:
                    logger.info(f"[SDEdit Phase 1 full-batch] {end}/{num_samples} samples done")
                else:
                    print(f"[SDEdit Phase 1 full-batch] {end}/{num_samples} samples done")
        else:
            for sample_idx in range(num_samples):
                sample = samples_all[sample_idx:sample_idx + 1].clone()

                if row_subsample:
                    curr_denoiser.get_subsample()

                with torch.no_grad():
                    for i, t in enumerate(sigmas_for_inference):
                        if i > t_switch:
                            break
                        if i < sigmas_for_inference.shape[0] - 1:
                            residual, beta_t = curr_denoiser.denoise(sample, t, i=i)
                            if empirical_sampler_type == 'ode':
                                sample, _ = noise_scheduler.euler_step(residual, t, sample, i)
                            elif empirical_sampler_type == 'sde':
                                sample, _ = noise_scheduler.euler_maruyama_step(residual, t, sample, i)
                            elif empirical_sampler_type == 'clipsde':
                                sample, _ = noise_scheduler.clipping_reverse_step(
                                    curr_denoiser.get_numerator(), beta_t, C, t, sample, i)
                            else:
                                raise NotImplementedError(
                                    f"Sampler '{empirical_sampler_type}' not supported for empirical phase")

                            if sample_idx < num_track:
                                if i not in track_buf:
                                    track_buf[i] = (t, [])
                                track_buf[i][1].append(sample.clone().detach().cpu())

                samples_all[sample_idx] = sample[0]

                if logger:
                    logger.info(f"[SDEdit Phase 1] {sample_idx + 1}/{num_samples} samples done")
                else:
                    print(f"[SDEdit Phase 1] {sample_idx + 1}/{num_samples} samples done")

        empirical_samples_64 = samples_all.cpu()

        for i_key in sorted(track_buf.keys()):
            t_val, tensors = track_buf[i_key]
            empirical_transformation.append(torch.cat(tensors, dim=0))
            empirical_timesteps.append(t_val)

        neural_start = t_switch

    # ── SDEdit bridge: re-noise the state back by noise_steps positions ───────
    # After empirical phase, the state has σ = sigmas_for_inference[t_switch + 1].
    # We add Gaussian noise to bring it to σ = sigmas_for_inference[neural_start + 1]
    # where neural_start = t_switch - noise_steps (higher σ = earlier in schedule).
    if t_switch > 0 and noise_steps > 0:
        effective_noise_steps = min(noise_steps, t_switch)
        neural_start = t_switch - effective_noise_steps

        sigmas_dev = sigmas_for_inference.to(device)
        sigma_after_empirical = sigmas_dev[t_switch + 1]
        sigma_neural_start = sigmas_dev[neural_start + 1]

        delta_sigma_sq = (sigma_neural_start ** 2 - sigma_after_empirical ** 2).clamp(min=0.0)
        re_noised = empirical_samples_64.to(device) + torch.sqrt(delta_sigma_sq) * torch.randn(
            empirical_samples_64.shape, device=device
        )
        empirical_samples_64 = re_noised.cpu()

        msg = (
            f"SDEdit bridge: σ {sigma_after_empirical.item():.4f} → "
            f"{sigma_neural_start.item():.4f} ({effective_noise_steps} steps back); "
            f"neural denoiser starts at step {neural_start}"
        )
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # ── Phase 2: neural denoiser from neural_start → end ─────────────────────
    if logger:
        logger.info(f"[SDEdit Phase 2] Neural denoiser from step {neural_start + 1} onward")
    else:
        print(f"[SDEdit Phase 2] Neural denoiser from step {neural_start + 1} onward")

    neural_result = sample_neural_denoiser_later_steps(
        denoiser=pmodel_imagenet,
        t_switch=neural_start,
        early_stop_samples=empirical_samples_64,
        dimen=dimen_64,
        batch_size=neural_batch_size if neural_batch_size is not None else comp_batch_size,
        num_samples=num_samples,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_timesteps=num_timesteps,
        sampler_type=neural_sampler_type,
        rho=rho,
        device=device,
        logger=logger,
        num_track=num_track,
    )

    # ── Downsample 64×64 → 32×32 ──────────────────────────────────────────────
    sample_tensor_64 = neural_result["samples"].clamp(-1, 1)
    sample_tensor_32 = convert_64_to_32(sample_tensor_64.to(device)).cpu()

    if save:
        sample_name = f"samples_cifar10_{experiment_name}_{neural_sampler_type}.pt"
        torch.save({"samples": sample_tensor_32}, os.path.join(samplesdir, sample_name))
        if logger:
            logger.info(f"Samples saved to {os.path.join(samplesdir, sample_name)}")
        else:
            print(f"Samples saved to {os.path.join(samplesdir, sample_name)}")

    max_vis_samples = min(100, num_samples)
    plot_rgb_grid(
        sample_tensor_32[:max_vis_samples],
        dimen=(32, 32), grid_size=(10, 10),
        imgname=f"cifar_samples_{neural_sampler_type}_{experiment_name}",
        imgdir=imgdir,
        sigma_min=sigma_min, sigma_max=sigma_max,
    )

    # ── Diffusion grid: full trajectory ───────────────────────────────────────
    neural_transformation = neural_result["sample_transformation"]
    neural_timesteps = neural_result["timesteps"]

    all_transformation_64 = empirical_transformation + neural_transformation
    all_timesteps = empirical_timesteps + neural_timesteps

    if all_transformation_64:
        all_transformation_32 = [
            convert_64_to_32(s.to(device)).cpu() for s in all_transformation_64
        ]
        n_total = len(all_transformation_32)
        indices = np.linspace(0, n_total - 1, min(11, n_total), dtype=int)
        transformation_short = [all_transformation_32[idx] for idx in indices]
        timesteps_short = [all_timesteps[idx] for idx in indices]

        plot_diffusion_grid(
            transformation_short,
            timesteps_short,
            rgb=True,
            dimen=(32, 32),
            imgname=f"diffusion_grid_{experiment_name}",
            imgdir=imgdir,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
        )

    return {
        "samples": sample_tensor_32,
    }


def generate_samples_sdedit_celeba_neural_64(
    full_dataset,
    data_loader,
    t_switch,
    noise_steps,
    pmodel_imagenet,
    num_samples=10000,
    samples_per_seed=1000,
    sigma_min=0.002,
    sigma_max=80.0,
    num_timesteps=50,
    empirical_sampler_type="sde",
    neural_sampler_type="edm",
    C=1.0,
    rho=7.0,
    save=False,
    experiment_name="sdedit_celeba_neural_64",
    samplesdir="./samples",
    imgdir="./images_app",
    logger=None,
    config=None,
    device=None,
    comp_batch_size=100,
    neural_batch_size=None,
    without_replacement=False,
    full_dataset_mode=False,
    row_subsample=False,
):
    """SDEdit-style hybrid pipeline: CelebA empirical denoiser + FFHQ-64 neural denoiser.

    Same structure as generate_samples_sdedit_empirical_neural_64, but the empirical
    denoiser is built from CelebA-64 instead of upscaled CIFAR10. CelebA is loaded
    natively at 64x64 (center-crop + resize), so both phases operate in 64x64 space
    with no resolution bridge, and the final samples stay at 64x64.

    Phase 1: CelebA empirical denoiser from step 0 → t_switch (high σ to low σ).
    Bridge:  add Gaussian noise to push the state back by noise_steps positions
             in the Karras σ schedule (SDEdit re-noising).
    Phase 2: FFHQ-64 neural denoiser from (t_switch - noise_steps) → end.

    Args:
        t_switch: Step index where the empirical denoiser stops.
        noise_steps: Karras-schedule steps to re-noise backward after the
            empirical phase. Must satisfy 0 <= noise_steps <= t_switch.
        Other args: same as generate_samples_sdedit_empirical_neural_64.
    """
    dimen_64 = 64 * 64 * 3

    noise_scheduler = NoiseScheduler(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_timesteps=num_timesteps,
    )
    sigmas_for_inference = noise_scheduler.get_sigmas_karras(
        num_timesteps,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        rho=rho,
    )

    num_track = min(5, num_samples)
    empirical_transformation = []
    empirical_timesteps = []

    # ── prop=0.0 fast path ────────────────────────────────────────────────────
    if t_switch == 0:
        if logger:
            logger.info("t_switch=0: skipping empirical phase, running all-neural")
        else:
            print("t_switch=0: skipping empirical phase, running all-neural")
        empirical_samples_64 = (torch.randn(num_samples, dimen_64) * sigma_max).cpu()
        neural_start = -1

    # ── Phase 1: CelebA empirical denoiser on 64×64, steps 0 → t_switch ──────
    else:
        data_loader_64, _ = get_celeba_dataset_64(n=comp_batch_size, shuffle=True)
        init_batch, _ = next(iter(data_loader_64))
        init_batch = init_batch.to(device)

        if full_dataset_mode:
            curr_denoiser = EmpiricalDenoiser(
                init_batch,
                data_loader=data_loader_64,
                clipping=(empirical_sampler_type == "clipsde"),
                n_dim=dimen_64,
                C=C,
                beta_hyper=config.beta_hyper,
                non_dp=False,
                is_full_dataset=True,
                device=device,
            )
        elif without_replacement:
            curr_denoiser = BatchCyclingEmpiricalDenoiser(
                init_batch,
                data_loader=data_loader_64,
                clipping=(empirical_sampler_type == "clipsde"),
                n_dim=dimen_64,
                C=C,
                beta_hyper=config.beta_hyper,
                non_dp=False,
                device=device,
            )
        elif row_subsample:
            curr_denoiser = RowSampleEmpiricalDenoiser(
                init_batch,
                data_loader=data_loader_64,
                clipping=(empirical_sampler_type == "clipsde"),
                n_dim=dimen_64,
                C=C,
                beta_hyper=config.beta_hyper,
                non_dp=False,
                device=device,
            )
        else:
            curr_denoiser = SubSampleEmpiricalDenoiser(
                init_batch,
                data_loader=data_loader_64,
                clipping=(empirical_sampler_type == "clipsde"),
                n_dim=dimen_64,
                C=C,
                beta_hyper=config.beta_hyper,
                non_dp=False,
                device=device,
            )

        samples_all = torch.randn([num_samples, dimen_64], device=device) * sigma_max
        track_buf = {}

        if full_dataset_mode:
            query_batch = comp_batch_size
            for start in range(0, num_samples, query_batch):
                end = min(start + query_batch, num_samples)
                sample = samples_all[start:end].clone()

                with torch.no_grad():
                    for i, t in enumerate(sigmas_for_inference):
                        if i > t_switch:
                            break
                        if i < sigmas_for_inference.shape[0] - 1:
                            residual, beta_t = curr_denoiser.denoise(sample, t, i=i)
                            if empirical_sampler_type == 'ode':
                                sample, _ = noise_scheduler.euler_step(residual, t, sample, i)
                            elif empirical_sampler_type == 'sde':
                                sample, _ = noise_scheduler.euler_maruyama_step(residual, t, sample, i)
                            elif empirical_sampler_type == 'clipsde':
                                sample, _ = noise_scheduler.clipping_reverse_step(
                                    curr_denoiser.get_numerator(), beta_t, C, t, sample, i)
                            else:
                                raise NotImplementedError(
                                    f"Sampler '{empirical_sampler_type}' not supported for empirical phase")

                            if start < num_track:
                                track_end = min(end, num_track) - start
                                if i not in track_buf:
                                    track_buf[i] = (t, [])
                                track_buf[i][1].append(sample[:track_end].clone().detach().cpu())

                samples_all[start:end] = sample

                if logger:
                    logger.info(f"[SDEdit CelebA Phase 1 full-batch] {end}/{num_samples} samples done")
                else:
                    print(f"[SDEdit CelebA Phase 1 full-batch] {end}/{num_samples} samples done")
        else:
            for sample_idx in range(num_samples):
                sample = samples_all[sample_idx:sample_idx + 1].clone()

                if row_subsample:
                    curr_denoiser.get_subsample()

                with torch.no_grad():
                    for i, t in enumerate(sigmas_for_inference):
                        if i > t_switch:
                            break
                        if i < sigmas_for_inference.shape[0] - 1:
                            residual, beta_t = curr_denoiser.denoise(sample, t, i=i)
                            if empirical_sampler_type == 'ode':
                                sample, _ = noise_scheduler.euler_step(residual, t, sample, i)
                            elif empirical_sampler_type == 'sde':
                                sample, _ = noise_scheduler.euler_maruyama_step(residual, t, sample, i)
                            elif empirical_sampler_type == 'clipsde':
                                sample, _ = noise_scheduler.clipping_reverse_step(
                                    curr_denoiser.get_numerator(), beta_t, C, t, sample, i)
                            else:
                                raise NotImplementedError(
                                    f"Sampler '{empirical_sampler_type}' not supported for empirical phase")

                            if sample_idx < num_track:
                                if i not in track_buf:
                                    track_buf[i] = (t, [])
                                track_buf[i][1].append(sample.clone().detach().cpu())

                samples_all[sample_idx] = sample[0]

                if logger:
                    logger.info(f"[SDEdit CelebA Phase 1] {sample_idx + 1}/{num_samples} samples done")
                else:
                    print(f"[SDEdit CelebA Phase 1] {sample_idx + 1}/{num_samples} samples done")

        empirical_samples_64 = samples_all.cpu()

        for i_key in sorted(track_buf.keys()):
            t_val, tensors = track_buf[i_key]
            empirical_transformation.append(torch.cat(tensors, dim=0))
            empirical_timesteps.append(t_val)

        neural_start = t_switch

    # ── SDEdit bridge: re-noise the state back by noise_steps positions ───────
    if t_switch > 0 and noise_steps > 0:
        effective_noise_steps = min(noise_steps, t_switch)
        neural_start = t_switch - effective_noise_steps

        sigmas_dev = sigmas_for_inference.to(device)
        sigma_after_empirical = sigmas_dev[t_switch + 1]
        sigma_neural_start = sigmas_dev[neural_start + 1]

        delta_sigma_sq = (sigma_neural_start ** 2 - sigma_after_empirical ** 2).clamp(min=0.0)
        re_noised = empirical_samples_64.to(device) + torch.sqrt(delta_sigma_sq) * torch.randn(
            empirical_samples_64.shape, device=device
        )
        empirical_samples_64 = re_noised.cpu()

        msg = (
            f"SDEdit bridge: σ {sigma_after_empirical.item():.4f} → "
            f"{sigma_neural_start.item():.4f} ({effective_noise_steps} steps back); "
            f"neural denoiser starts at step {neural_start}"
        )
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # ── Phase 2: neural denoiser from neural_start → end ─────────────────────
    if logger:
        logger.info(f"[SDEdit CelebA Phase 2] Neural denoiser from step {neural_start + 1} onward")
    else:
        print(f"[SDEdit CelebA Phase 2] Neural denoiser from step {neural_start + 1} onward")

    neural_result = sample_neural_denoiser_later_steps(
        denoiser=pmodel_imagenet,
        t_switch=neural_start,
        early_stop_samples=empirical_samples_64,
        dimen=dimen_64,
        batch_size=neural_batch_size if neural_batch_size is not None else comp_batch_size,
        num_samples=num_samples,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_timesteps=num_timesteps,
        sampler_type=neural_sampler_type,
        rho=rho,
        device=device,
        logger=logger,
        num_track=num_track,
    )

    # ── CelebA target resolution is 64×64: no downsampling ───────────────────
    sample_tensor_64 = neural_result["samples"].clamp(-1, 1).cpu()

    if save:
        sample_name = f"samples_celeba_{experiment_name}_{neural_sampler_type}.pt"
        torch.save({"samples": sample_tensor_64}, os.path.join(samplesdir, sample_name))
        if logger:
            logger.info(f"Samples saved to {os.path.join(samplesdir, sample_name)}")
        else:
            print(f"Samples saved to {os.path.join(samplesdir, sample_name)}")

    max_vis_samples = min(100, num_samples)
    plot_rgb_grid(
        sample_tensor_64[:max_vis_samples],
        dimen=(64, 64), grid_size=(10, 10),
        imgname=f"celeba_samples_{neural_sampler_type}_{experiment_name}",
        imgdir=imgdir,
        sigma_min=sigma_min, sigma_max=sigma_max,
    )

    # ── Diffusion grid: full trajectory ───────────────────────────────────────
    neural_transformation = neural_result["sample_transformation"]
    neural_timesteps = neural_result["timesteps"]

    all_transformation_64 = empirical_transformation + neural_transformation
    all_timesteps = empirical_timesteps + neural_timesteps

    if all_transformation_64:
        n_total = len(all_transformation_64)
        indices = np.linspace(0, n_total - 1, min(11, n_total), dtype=int)
        transformation_short = [all_transformation_64[idx] for idx in indices]
        timesteps_short = [all_timesteps[idx] for idx in indices]

        plot_diffusion_grid(
            transformation_short,
            timesteps_short,
            rgb=True,
            dimen=(64, 64),
            imgname=f"diffusion_grid_{experiment_name}",
            imgdir=imgdir,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
        )

    return {
        "samples": sample_tensor_64,
    }


def generate_without_row_sampling(denoising_step_hybrid, pmodel_imagenet, t_switch):
    denoise_cfg = dict(C = config.C, 
            sigma_max = config.sigma_max, 
            noise_scheduler=noise_scheduler, 
            sigmas_for_inference=sigmas_for_inference, 
            sampler_type=config.sampler_type, 
            dimen = dimen, 
            num_samples = config.num_samples,)
    result_dict_hybrid = denoising_step_hybrid(t_switch, pmodel_imagenet, pfode, empirical_toggle=False, **denoise_cfg)
    sample_tensor = result_dict_hybrid["samples"]
    sample_transformation = result_dict_hybrid["sample_transformation"]
    timesteps = result_dict_hybrid["timesteps"]
    if config.save == 1:
        sample_name = f"samples_{config.dataset}_{config.experiment_name}_{config.sampler_type}.pt"
        torch.save({"samples": sample_tensor}, os.path.join(samplesdir, sample_name))
        if logger:
            logger.info(f"Samples saved to {os.path.join(samplesdir, sample_name)}")
        else:
            print(f"Samples saved to {os.path.join(samplesdir, sample_name)}")
        num_samples = min(100, config.num_samples)
    
    if (config.dataset == "cifar10"):
            # Get indices for 10 evenly spaced elements
        proportion_indices = np.linspace(0, 50 - 1, 10 + 1, dtype=int)

        sample_transformation_short = [sample_transformation[idx] for i, idx in enumerate(proportion_indices)]
        timesteps_short = [timesteps[idx] for i, idx in enumerate(proportion_indices)]

        plot_diffusion_grid(sample_transformation_short, timesteps_short, rgb = True, dimen=(32, 32), imgname=f"denoiser_{config.sampler_type}_{config.experiment_name}", imgdir=imgdir, sigma_max=config.sigma_max, sigma_min=config.sigma_min)
        plot_rgb_grid(sample_tensor[:num_samples], dimen=(32, 32), grid_size=(10, 10),
                        imgname=f"cifar_samples_{config.sampler_type}_{config.experiment_name}", imgdir=imgdir,
                        sigma_min=config.sigma_min, sigma_max=config.sigma_max)
        
def sample_neural_denoiser_early_stop(
    pmodel_imagenet,
    t_switch,
    prop,
    denoiser,
    file_name,
    dimen = 2,
    batch_size=2000,
    num_samples=1000, 
    sigma_min=0.002, 
    sigma_max=80.0, 
    num_timesteps=50, 
    sampler_type="sde", 
    rho=7.0,
    cachedir="./cache",
    device=None
):
    """
    Generate samples using only the neural denoiser, stopping early at t_switch. Saves samples to disk.
    Method is to apply a full denoising pass up to x_0, then add corresponding SDE noise back to reach t_switch.
    
    """
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
    ).to(device)
    
    num_batches = int(np.ceil(num_samples / batch_size))
    
    # Store all samples
    all_samples = []
    
    # Visualization data
    sample_transformation = []
    timesteps = []
    num_denoise = min(5, num_samples)
    # first_run = True
    
     # Sample in batches
    for batch_idx in tqdm(range(num_batches)):
        if logger:
            logger.info(f"Sampling batch {batch_idx + 1}/{num_batches}...")
        else:
            print(f"Sampling batch {batch_idx + 1}/{num_batches}...")
            
        sample_size = min(batch_size, num_samples - batch_idx * batch_size)
        samples = torch.randn([sample_size, dimen], device=device) * sigma_max
        # Create one-hot encoding of random label classes
        class_labels = torch.randint(0, 1000, (sample_size,), device=device)
        class_labels_one_hot = F.one_hot(class_labels, num_classes=1000).float()
        
        with torch.no_grad():
            for i, t in enumerate(sigmas_for_inference):
                if i < sigmas_for_inference.shape[0] - 1:
                    # EDM
                    residual3, beta_t = denoiser.denoise_conditional(samples, t, class_labels_one_hot, i=i)
                    samples, _ = noise_scheduler.edm_stochastic_step(
                        residual3, t, sigmas_for_inference[i+1], samples, i, denoiser, class_labels_one_hot)
                    
                    # if first_run and (i <= t_switch):
                    #     timesteps.append(t)
                    #     sample_transformation.append(samples.clone().detach().cpu()[:num_denoise])
        
        # Now add noise back to reach t_switch
        # t_target = sigmas_for_inference[t_switch]
        samples_32 = convert_64_to_32(samples)
        # noise = torch.randn_like(samples_32)
        # samples = samples_32 + noise * t_target
        
        all_samples.append(samples_32.clone().detach().cpu())
        # first_run = False
        
    sample_tensor = torch.cat(all_samples, dim=0)
    # Save the samples
    
    results = {
        "samples": sample_tensor,
        "sample_transformation": sample_transformation,
        "timesteps": timesteps,
    }
    torch.save(results, os.path.join(cache_dir, file_name))
    if logger:
        logger.info(f"Neural denoiser early stop samples saved to {os.path.join(cache_dir, file_name)}")
    else:
        print(f"Neural denoiser early stop samples saved to {os.path.join(cache_dir, file_name)}")
    
    return results

def sample_empirical_denoiser_late_start(
    t_switch,
    data_loader,
    early_stop_samples,
    sample_batch_size=100,
    batch_size=2000,
    num_samples=1000, 
    sigma_min=0.002, 
    sigma_max=80.0, 
    num_timesteps=50, 
    sampler_type="sde", 
    C=1.0,
    beta_hyper=1.0,
    rho=7.0,
    samples_per_seed=1000,
    device=None,
    logger=None
):
    """Generate samples using only the empirical denoiser, starting late at t_switch with row-wise subsampling.
    
    Uses samples_per_seed to determine how many samples share the same empirical denoiser.
    For example:
    - If samples_per_seed=100 and num_samples=10000, creates 100 chunks
    - Each chunk uses sample_batch_size samples to create an empirical denoiser
    - That denoiser processes samples_per_seed samples in parallel
    """
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
    
    dimen = early_stop_samples.shape[1]
    
    # Calculate number of chunks: each chunk uses sample_batch_size samples to generate sample_batch_size outputs
    num_chunks = (num_samples + sample_batch_size - 1) // sample_batch_size
    
    all_samples = []
    all_samples_one_step = []
    
    # Track transformations for visualization
    sample_transformation = []
    timesteps = []
    num_denoise = min(5, samples_per_seed)
    first_chunk = True
    
    for chunk_idx in range(num_chunks):
        seed_value = chunk_idx + 1
        if logger:
            logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks} with seed {seed_value}")
        else:
            print(f"Processing chunk {chunk_idx + 1}/{num_chunks} with seed {seed_value}")

        generator = torch.Generator()
        generator.manual_seed(seed_value)
        
        # Get a unique subsampled dataset for this chunk
        seed_data_loader, n_samples = get_cifar10_dataset(n=sample_batch_size, 
                                                          shuffle=True, 
                                                          generator=generator,)
        subsampled_dataset, _ = next(iter(seed_data_loader))
        subsampled_dataset = subsampled_dataset.to(device)
        
        # Create empirical denoiser with this chunk's subsampled data
        curr_denoiser = RowSampleEmpiricalDenoiser(
            subsampled_dataset,
            data_loader=seed_data_loader,
            clipping=(sampler_type == "clipsde"), 
            n_dim=dimen, 
            C=C,
            beta_hyper=beta_hyper,
            non_dp=True,
            device=device
        )
        
        # Get the corresponding early stop samples for this chunk
        start_idx = chunk_idx * sample_batch_size
        end_idx = min(start_idx + sample_batch_size, num_samples)
        chunk_early_stop_samples = early_stop_samples[start_idx:end_idx]
        
        # Process each sample individually
        chunk_samples = []
        chunk_one_step_samples = []
        
        for sample_idx in range(chunk_early_stop_samples.shape[0]):
            sample = chunk_early_stop_samples[sample_idx:sample_idx+1].to(device)
            
            if logger:
                logger.info(f"Chunk {chunk_idx+1}/{num_chunks}, Sample {sample_idx+1}/{chunk_early_stop_samples.shape[0]}: Continuing denoising from timestep {t_switch + 1}/{num_timesteps}")
            
            # Continue from t_switch onward
            for i, t in enumerate(sigmas_for_inference):
                if i < t_switch + 1:
                    continue  # Skip until we reach t_switch
                if i < sigmas_for_inference.shape[0] - 1:
                    with torch.no_grad():
                        residual3, beta_t = curr_denoiser.denoise(sample, t, i=i)
                    if sampler_type == 'ode':
                        sample, _ = noise_scheduler.euler_step(residual3, t, sample, i)
                    elif sampler_type == 'clipsde':
                        sample, _ = noise_scheduler.clipping_reverse_step(
                            curr_denoiser.get_numerator(), beta_t, C, t, sample, i)
                    elif sampler_type == 'sde':
                        sample, _ = noise_scheduler.euler_maruyama_step(residual3, t, sample, i)
                    else:
                        raise NotImplementedError(f"Sampler type {sampler_type} not supported for empirical denoiser")
                    
                
                # if first_chunk and sample_idx < num_denoise:
                #     if sample_idx == 0:  # Only record timesteps once
                #         timesteps.append(t)
                #     if len(sample_transformation) <= (i - t_switch - 1):
                #         sample_transformation.append([])
                #     sample_transformation[i - t_switch - 1].append(sample.clone().detach().cpu())
                    
                if (i == t_switch + 1):
                    chunk_one_step_samples.append(sample.clone().detach().cpu())
            
            chunk_samples.append(sample.clone().detach().cpu())
            curr_denoiser.get_subsample()
        
        all_samples.append(torch.cat(chunk_samples, dim=0))
        all_samples_one_step.append(torch.cat(chunk_one_step_samples, dim=0))
        first_chunk = False
        
        if logger:
            logger.info(f"Generated {all_samples[-1].shape[0]} sample(s) for this chunk")
    
    sample_transformation_tensors = [torch.cat(step_samples, dim=0) for step_samples in sample_transformation]
    sample_tensor = torch.cat(all_samples, dim=0)[:num_samples]
    one_step_sample_tensor = torch.cat(all_samples_one_step, dim=0)[:num_samples]
    
    if logger:
        logger.info(f"Total samples generated: {sample_tensor.shape[0]}")
    
    return {
        "samples": sample_tensor,
        "one_step_samples": one_step_sample_tensor,
        "sample_transformation": sample_transformation_tensors,
        "timesteps": timesteps,
    }
    
def sample_empirical_denoiser_random_subsampling(
    t_switch,
    data_loader,
    early_stop_samples,
    without_replacement,
    sample_batch_size=100,
    batch_size=2000,
    num_samples=1000, 
    sigma_min=0.002, 
    sigma_max=80.0, 
    num_timesteps=50, 
    sampler_type="sde", 
    C=1.0,
    beta_hyper=1.0,
    rho=7.0,
    samples_per_seed=1000,
    device=None,
    logger=None
):
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

    dimen = early_stop_samples.shape[1]

    all_samples = []
    all_samples_one_step = []

    # Track transformations for visualization
    sample_transformation = []
    timesteps = []
    num_denoise = min(5, samples_per_seed)
        
    # Get a unique subsampled dataset for this chunk
    data_loader, n_samples = get_cifar10_dataset(n=sample_batch_size, 
                                                        shuffle=True)
    subsampled_dataset, _ = next(iter(data_loader))
    subsampled_dataset = subsampled_dataset.to(device)

    if (without_replacement):
        # Create an empirical denoiser without replacement
        curr_denoiser = BatchCyclingEmpiricalDenoiser(
            subsampled_dataset,
            data_loader=data_loader,
            clipping=(sampler_type == "clipsde"), 
            n_dim=dimen, 
            C=C,
            beta_hyper=beta_hyper,
            non_dp=True,
            device=device
        )
    else:
        curr_denoiser = SubSampleEmpiricalDenoiser(
            subsampled_dataset,
            data_loader=data_loader,
            clipping=(sampler_type == "clipsde"),
            n_dim=dimen, 
            C=C,
            beta_hyper=beta_hyper,
            non_dp=True,
            device=device
        )
    
    samples = early_stop_samples.to(device)
    for sample_idx in range(samples.shape[0]):
        sample = samples[sample_idx:sample_idx+1]
        
        # Continue from t_switch onward
        for i, t in enumerate(sigmas_for_inference):
            if i < t_switch + 1:
                continue  # Skip until we reach t_switch
                
            if i < sigmas_for_inference.shape[0] - 1:
                with torch.no_grad():
                    residual3, beta_t = curr_denoiser.denoise(sample, t, i=i)
                    
                if sampler_type == 'ode':
                    sample, _ = noise_scheduler.euler_step(residual3, t, sample, i)
                elif sampler_type == 'clipsde':
                    sample, _ = noise_scheduler.clipping_reverse_step(
                        curr_denoiser.get_numerator(), beta_t, C, t, sample, i)
                elif sampler_type == 'sde':
                    sample, _ = noise_scheduler.euler_maruyama_step(residual3, t, sample, i)
                else:
                    raise NotImplementedError(
                        f"Sampler type {sampler_type} not supported for empirical denoiser"
                    )
            
            # Track transformations for first few samples
            # if sample_idx < num_denoise:
            #     if sample_idx == 0:  # Only record timesteps once
            #         timesteps.append(t.item() if torch.is_tensor(t) else t)
                    
            #     if len(sample_transformation) <= (i - t_switch - 1):
            #         sample_transformation.append([])
            #     sample_transformation[i - t_switch - 1].append(
            #         sample.clone().detach().cpu()
            #     )
            
            if i == t_switch + 1:
                all_samples_one_step.append(sample.clone().detach().cpu())
    
        all_samples.append(sample.clone().detach().cpu())
        if logger and (sample_idx + 1) % 100 == 0:
            logger.info(f"Generated {sample_idx + 1}/{samples.shape[0]} samples")

    # Concatenate all samples
    sample_transformation_tensors = [
        torch.cat(step_samples, dim=0) for step_samples in sample_transformation
    ]
    sample_tensor = torch.cat(all_samples, dim=0)[:num_samples]
    one_step_sample_tensor = torch.cat(all_samples_one_step, dim=0)[:num_samples]

    if logger:
        logger.info(f"Total samples generated: {sample_tensor.shape[0]}")

    return {
        "samples": sample_tensor,
        "one_step_samples": one_step_sample_tensor,
        "sample_transformation": sample_transformation_tensors,
        "timesteps": timesteps,
    }
        
    
    
def sample_neural_denoiser_later_steps(
    denoiser,
    t_switch,
    early_stop_samples,
    dimen=2,
    batch_size=2000,
    num_samples=1000,
    sigma_min=0.002,
    sigma_max=80.0,
    num_timesteps=50,
    sampler_type="sde",
    rho=7.0,
    device=None,
    logger=None,
    num_track=5,
):
    """Complete generation with a neural denoiser starting from empirical denoiser one-step samples.

    Takes one-step samples produced by the empirical denoiser at t_switch and runs
    the neural denoiser for the remaining (later, lower-sigma) timesteps.

    Args:
        denoiser: Neural denoiser (e.g. ImageNet64NeuralDenoiser).
        t_switch: Timestep index where the empirical denoiser stopped.
        early_stop_samples: Tensor of shape (num_samples, dimen) from the empirical denoiser.
        dimen: Flattened spatial dimension of the samples.
        batch_size: Number of samples to process per forward pass.
        num_samples: Total samples to generate (truncates early_stop_samples if needed).
        sigma_min/sigma_max: Noise schedule bounds.
        num_timesteps: Number of diffusion timesteps.
        sampler_type: 'edm' uses EDM stochastic step; 'sde' uses Euler-Maruyama; 'ode' uses Euler.
        rho: Rho for the Karras sigma schedule.
        device: Torch device.
        logger: Optional DPLogger instance.

    Returns:
        dict with keys: samples, sample_transformation, timesteps.
    """
    noise_scheduler = NoiseScheduler(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_timesteps=num_timesteps,
    )

    sigmas_for_inference = noise_scheduler.get_sigmas_karras(
        num_timesteps,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        rho=rho,
    ).to(device)

    samples_in = early_stop_samples[:num_samples]  # keep on CPU; move per-batch below
    num_batches = int(np.ceil(samples_in.shape[0] / batch_size))

    all_samples = []
    sample_transformation = []
    timesteps = []

    for batch_idx in tqdm(range(num_batches)):
        start = batch_idx * batch_size
        end = min(start + batch_size, samples_in.shape[0])
        samples = samples_in[start:end].clone().to(device)

        sample_size = samples.shape[0]
        class_labels = torch.randint(0, 1000, (sample_size,), device=device)
        class_labels_one_hot = F.one_hot(class_labels, num_classes=1000).float()

        with torch.no_grad(), autocast():
            for i, t in enumerate(sigmas_for_inference):
                if i <= t_switch:
                    continue  # empirical denoiser already handled these steps

                if i < sigmas_for_inference.shape[0] - 1:
                    # EDM
                    residual3, beta_t = denoiser.denoise_conditional(samples, t, class_labels_one_hot, i=i)
                    residual3 = residual3.float()
                    samples, _ = noise_scheduler.edm_stochastic_step(
                        residual3, t, sigmas_for_inference[i+1], samples, i, denoiser, class_labels_one_hot)

                    # samples, _ = noise_scheduler.euler_maruyama_step(
                    #     residual3, t, samples, i
                    # )

                    if batch_idx == 0:
                        sample_transformation.append(samples[:num_track].clone().detach().cpu())
                        timesteps.append(t)
                    # if i < sigmas_for_inference.shape[0] - 1:
                    #     # if sampler_type == 'edm':
                    #     residual3, beta_t = denoiser.denoise_conditional(
                    #         samples, t, class_labels_one_hot, i=i
                    #     )
                    #     samples, _ = noise_scheduler.edm_stochastic_step(
                    #         residual3, t, sigmas_for_inference[i + 1], samples, i,
                    #         denoiser, class_labels_one_hot,
                    #     )
                    # elif sampler_type == 'sde':
                    #     residual3, beta_t = denoiser.denoise(samples, t, i=i)
                    #     samples, _ = noise_scheduler.euler_maruyama_step(
                    #         residual3, t, samples, i
                    #     )
                    # elif sampler_type == 'ode':
                    #     residual3, beta_t = denoiser.denoise(samples, t, i=i)
                    #     samples, _ = noise_scheduler.euler_step(residual3, t, samples, i)
                    # else:
                    #     raise NotImplementedError(
                    #         f"Sampler type '{sampler_type}' not supported in sample_neural_denoiser_later_steps"
                    #     )

        all_samples.append(samples.clone().detach().cpu())

        if logger:
            logger.info(f"Neural later-steps: batch {batch_idx + 1}/{num_batches} done")
        else:
            print(f"Neural later-steps: batch {batch_idx + 1}/{num_batches} done")

    sample_tensor = torch.cat(all_samples, dim=0)

    return {
        "samples": sample_tensor,
        "sample_transformation": sample_transformation,
        "timesteps": timesteps,
    }


def convert_64_to_32(neural_samples):
    neural_samples_resized = neural_samples.view(-1, 3, 64, 64)
    neural_samples_resized = F.interpolate(neural_samples_resized, size=(32, 32), mode='bilinear', align_corners=False)
    neural_samples_resized = neural_samples_resized.view(-1, 3*32*32)
    return neural_samples_resized

if __name__ == "__main__":
    
    # Script to run hybrid diffusion denoising experiments for multiple samples and save results
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10", "celeba"])
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--sampler_type", type=str, default='sde', choices = ["sde", "ode", "clipsde"])
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=80.)
    parser.add_argument("--beta_hyper", type=float, default=None)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples is L in our write up")
    parser.add_argument("--batch_size", type=int, default=2000, help="Batch size is m in our write up")
    parser.add_argument("--subsample_batch_size", type=int, default=50, help="Batch size for subsampling (n in q=m/n)")
    parser.add_argument("--subsample", type=int, default=0)
    parser.add_argument("--row_subsample", type=int, default=0, help="Use row-wise subsampling (1=yes, 0=no).")
    parser.add_argument("--samples_per_seed", type=int, default=1000, help="Number of samples to generate per seed")
    parser.add_argument("--verbosity", type=int, default=3, help="Verbosity level: 0=silent, 1=errors, 2=warnings, 3=info, 4=debug, 5=trace")
    parser.add_argument("--rho", type=float, default=7.0, help="Rho value for Karras schedule")
    parser.add_argument("--batch_seed", type=int, default=0, help="Specific seed for data loader batch sampling")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model (if any)")
    parser.add_argument("--prop", type=float, default=0.9, help="Proportion of sampling to switch denoisers")
    parser.add_argument("--save", type=int, default=1, help="Whether to save samples to disk (1=yes, 0=no)")
    parser.add_argument("--pretrained64", action = "store_true", help="Use pretrained EDM model for neural denoiser")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory to cached samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--without_replacement", type=int, default=0, help="Use without replacement empirical denoiser (1=yes, 0=no)")
    parser.add_argument("--neural_later_steps", type=int, default=0, help="After empirical one-step samples, complete generation with neural denoiser (1=yes, 0=no)")
    parser.add_argument("--empirical_first", type=int, default=0, help="Run empirical denoiser first then switch to neural denoiser (1=yes, 0=no)")
    parser.add_argument("--empirical_without_replacement", type=int, default=0, help="Use BatchCyclingEmpiricalDenoiser (without replacement) for empirical phase (1=yes, 0=no)")
    parser.add_argument("--empirical_per_sample_subsample", type=int, default=0, help="Pure empirical denoising with per-sample subsampling (1=yes, 0=no). Overrides hybrid modes when enabled.")
    parser.add_argument("--neural_batch_size", type=int, default=None, help="Batch size for the neural denoiser phase (defaults to subsample_batch_size if not set)")
    parser.add_argument("--full_dataset_mode", type=int, default=0, help="Use base EmpiricalDenoiser with is_full_dataset=True for the empirical phase (1=yes, 0=no)")
    parser.add_argument("--sdedit_empirical_neural", type=int, default=0, help="SDEdit-style hybrid: empirical → re-noise → neural (1=yes, 0=no). Requires --pretrained64.")
    parser.add_argument("--noise_steps", type=int, default=5, help="Number of Karras sigma-schedule steps to re-noise backward in the SDEdit bridge (used with --sdedit_empirical_neural 1)")

    config = parser.parse_args()
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    logger = DPLogger(name="dp-denoiser", log_to_file=True, verbosity=config.verbosity)
    device = get_torch_device(config.device, logger=logger)
    os.makedirs(config.outdir, exist_ok=True)
    imgdir = f"{config.outdir}/images_app"
    os.makedirs(imgdir, exist_ok=True)
    samplesdir = f"{config.outdir}/samples"
    os.makedirs(samplesdir, exist_ok=True)
 
    batch_size = config.batch_size
 
    # Get data (with no subsampling first!)
    if config.dataset == "celeba":
        data_loader, n_samples = get_celeba_dataset_64(n=batch_size, shuffle=True)
    else:
        data_loader, n_samples = get_cifar10_dataset(n = batch_size, shuffle = True)
 
    dataset, init_label = next(iter(data_loader))
    dataset = dataset.to(device)

    dimen = dataset.shape[1]
 
    if (not config.pretrained64):
        # Get neural denoiser from pretrained model
        pmodel_imagenet = get_imagenet32_neural_denoiser(config.model_path, device=device)
        
        noise_scheduler = NoiseScheduler(
            num_timesteps=config.num_timesteps,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max
        )
        
        sigmas_for_inference = noise_scheduler.get_sigmas_karras(
            config.num_timesteps,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            rho=config.rho,
            device = device
        )
        
        # Get t_switch from the proportion
        t_switch = int(config.num_timesteps * config.prop)
        
        if config.row_subsample == 1:
            sampling_kwargs = dict(
                full_dataset=dataset,
                data_loader=data_loader,
                t_switch=t_switch,
                pmodel_imagenet=pmodel_imagenet,
                num_samples=config.num_samples,
                samples_per_seed=config.samples_per_seed,
                dataset_name=config.dataset,
                sigma_min=config.sigma_min,
                sigma_max=config.sigma_max,
                num_timesteps=config.num_timesteps,
                sampler_type=config.sampler_type,
                C=config.C,
                rho=config.rho,
                save=(config.save == 1),
                experiment_name=config.experiment_name,
                samplesdir=samplesdir,
                imgdir=imgdir,
                logger=logger,
                config=config,
                device=device,
            )
            if config.empirical_per_sample_subsample == 1:
                result_dict_hybrid = generate_samples_empirical_per_sample_subsampling(
                    full_dataset=dataset,
                    data_loader=data_loader,
                    num_samples=config.num_samples,
                    samples_per_seed=config.samples_per_seed,
                    dataset_name=config.dataset,
                    sigma_min=config.sigma_min,
                    sigma_max=config.sigma_max,
                    num_timesteps=config.num_timesteps,
                    sampler_type=config.sampler_type,
                    C=config.C,
                    rho=config.rho,
                    save=(config.save == 1),
                    experiment_name=config.experiment_name,
                    samplesdir=samplesdir,
                    imgdir=imgdir,
                    logger=logger,
                    beta_schedule=None,
                    comp_batch_size=1,
                    config=config,
                    device=device,
                )
            elif config.empirical_first == 1:
                result_dict_hybrid = generate_samples_empirical_then_neural_row_subsampling(
                    **sampling_kwargs,
                    full_dataset_mode=(config.full_dataset_mode == 1),
                )
            else:
                result_dict_hybrid = generate_samples_hybrid_model_row_subsampling(**sampling_kwargs)
        else:
            generate_without_row_sampling(denoising_step_hybrid, pmodel_imagenet, t_switch)
    else:
        noise_scheduler = NoiseScheduler(
            num_timesteps=config.num_timesteps,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max
        )
        
        sigmas_for_inference = noise_scheduler.get_sigmas_karras(
            config.num_timesteps,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            rho=config.rho,
            device = device
        )
        
        raw_model = get_imagenet64_neural_denoiser(config.model_path, device=device)
        # Unconditional checkpoints (label_dim=0, e.g. FFHQ-64) need the FFHQ wrapper
        if getattr(raw_model, "label_dim", None) == 0:
            pmodel_imagenet = FFHQ64NeuralDenoiser(raw_model, device=device)
        else:
            pmodel_imagenet = ImageNet64NeuralDenoiser(raw_model, device=device)

        # Get t_switch from the proportion
        t_switch = int(config.num_timesteps * config.prop)

        if config.sdedit_empirical_neural == 1:
            if config.dataset == "celeba":
                sdedit_fn = generate_samples_sdedit_celeba_neural_64
            else:
                sdedit_fn = generate_samples_sdedit_empirical_neural_64
            result = sdedit_fn(
                full_dataset=dataset,
                data_loader=data_loader,
                t_switch=t_switch,
                noise_steps=config.noise_steps,
                pmodel_imagenet=pmodel_imagenet,
                num_samples=config.num_samples,
                samples_per_seed=config.samples_per_seed,
                sigma_min=config.sigma_min,
                sigma_max=config.sigma_max,
                num_timesteps=config.num_timesteps,
                empirical_sampler_type="sde",
                neural_sampler_type="edm",
                C=config.C,
                rho=config.rho,
                save=(config.save == 1),
                experiment_name=config.experiment_name,
                samplesdir=samplesdir,
                imgdir=imgdir,
                logger=logger,
                config=config,
                device=device,
                comp_batch_size=config.subsample_batch_size,
                neural_batch_size=config.neural_batch_size,
                without_replacement=(config.empirical_without_replacement == 1),
                full_dataset_mode=(config.full_dataset_mode == 1),
                row_subsample=(config.row_subsample == 1),
            )
            logger.info("SDEdit empirical-neural 64x64 hybrid pipeline complete.")
        elif config.empirical_first == 1:
            result = generate_samples_empirical_first_neural_later_64(
                full_dataset=dataset,
                data_loader=data_loader,
                t_switch=t_switch,
                pmodel_imagenet=pmodel_imagenet,
                num_samples=config.num_samples,
                samples_per_seed=config.samples_per_seed,
                sigma_min=config.sigma_min,
                sigma_max=config.sigma_max,
                num_timesteps=config.num_timesteps,
                empirical_sampler_type="sde",
                neural_sampler_type="edm",
                C=config.C,
                rho=config.rho,
                save=(config.save == 1),
                experiment_name=config.experiment_name,
                samplesdir=samplesdir,
                imgdir=imgdir,
                logger=logger,
                config=config,
                device=device,
                comp_batch_size=config.subsample_batch_size,
                neural_batch_size=config.neural_batch_size,
                without_replacement=(config.empirical_without_replacement == 1),
                full_dataset_mode=(config.full_dataset_mode == 1),
            )
            logger.info("Empirical-first 64x64 hybrid pipeline complete.")
        else:  # neural-first (existing behavior)
            cache_dir = os.path.join(config.cache_dir, "imagenet")
            os.makedirs(cache_dir, exist_ok=True)
            neural_samples_name = f"samples_imagenet_{config.num_samples}.pt"

            if os.path.exists(os.path.join(cache_dir, neural_samples_name)):
                logger.info(f"Loading existing neural samples from {neural_samples_name}")
                neural_result = torch.load(os.path.join(cache_dir, neural_samples_name))
            else:
                logger.info(f"Generating neural samples and saving to {neural_samples_name}")
                neural_result = sample_neural_denoiser_early_stop(
                    pmodel_imagenet=pmodel_imagenet,
                    t_switch=t_switch,
                    prop=config.prop,
                    denoiser=pmodel_imagenet,
                    file_name=neural_samples_name,
                    dimen=64*64*3,
                    batch_size=batch_size,
                    num_samples=config.num_samples,
                    sigma_min=config.sigma_min,
                    sigma_max=config.sigma_max,
                    num_timesteps=config.num_timesteps,
                    sampler_type=config.sampler_type,
                    rho=config.rho,
                    device=device
                )

            neural_samples = neural_result["samples"].to(device)
            neural_transformation = neural_result["sample_transformation"]
            neural_timesteps = neural_result["timesteps"]

            # Add noise to neural samples to reach t_switch if needed
            if t_switch < config.num_timesteps - 1:
                t_target = sigmas_for_inference[t_switch]
                noise = torch.randn_like(neural_samples)
                neural_samples = neural_samples + noise * t_target

            # Visualize neural samples
            max_vis_samples = min(100, config.num_samples)

            plot_rgb_grid(neural_samples[:max_vis_samples], dimen=(32, 32), grid_size=(10, 10),
                              imgname=f"hybrid_neural_samples_prop{config.prop}", imgdir=imgdir,
                              sigma_min=config.sigma_min, sigma_max=config.sigma_max)

            if (config.without_replacement == 1):
                logger.info("Using without replacement empirical denoiser for hybrid sampling")
            else:
                logger.info("Using standard empirical denoiser with replacement for hybrid sampling")

            hybrid_samples = sample_empirical_denoiser_random_subsampling(
                t_switch=t_switch,
                data_loader=data_loader,
                early_stop_samples=neural_samples,
                sample_batch_size=config.subsample_batch_size,
                without_replacement=(config.without_replacement == 1),
                batch_size=batch_size,
                num_samples=config.num_samples,
                sigma_min=config.sigma_min,
                sigma_max=config.sigma_max,
                num_timesteps=config.num_timesteps,
                sampler_type=config.sampler_type,
                C=config.C,
                beta_hyper=config.beta_hyper,
                rho=config.rho,
                samples_per_seed=config.samples_per_seed,
                device=device,
                logger=logger
            )

            final_sample_name = f"samples_{config.dataset}_{config.experiment_name}_{config.sampler_type}.pt"
            final_sample_one_step_name = f"samples_one_step_{config.dataset}_{config.experiment_name}_{config.sampler_type}.pt"

            if config.save == 1:
                torch.save({"samples": hybrid_samples["samples"]}, os.path.join(samplesdir, final_sample_name))
                torch.save({"samples": hybrid_samples["one_step_samples"]}, os.path.join(samplesdir, final_sample_one_step_name))
                logger.info(f"Hybrid samples saved to {os.path.join(samplesdir, final_sample_name)}")
                logger.info(f"Hybrid one step samples saved to {os.path.join(samplesdir, final_sample_one_step_name)}")

            # Visualize hybrid samples
            hybrid_sample_tensor = hybrid_samples["samples"]
            one_step_sample_tensor = hybrid_samples["one_step_samples"]
            sample_transformation_hybrid = hybrid_samples["sample_transformation"]
            timesteps_hybrid = hybrid_samples["timesteps"]

            if config.neural_later_steps == 1:
                logger.info("Running neural denoiser on empirical one-step samples (neural_later_steps=1)")
                _n = one_step_sample_tensor.shape[0]
                _samples_64 = one_step_sample_tensor.reshape(_n, 3, 32, 32)
                _samples_64 = F.interpolate(_samples_64.float(), size=(64, 64), mode='bilinear', align_corners=False)
                _samples_64 = _samples_64.reshape(_n, 3 * 64 * 64)
                neural_later_result = sample_neural_denoiser_later_steps(
                    denoiser=pmodel_imagenet,
                    t_switch=t_switch,
                    early_stop_samples=_samples_64,
                    dimen=64 * 64 * 3,
                    batch_size=batch_size,
                    num_samples=config.num_samples,
                    sigma_min=config.sigma_min,
                    sigma_max=config.sigma_max,
                    num_timesteps=config.num_timesteps,
                    sampler_type="edm",
                    rho=config.rho,
                    device=device,
                    logger=logger,
                )
                _out = neural_later_result["samples"].reshape(-1, 3, 64, 64)
                _out = F.interpolate(_out.float(), size=(32, 32), mode='bilinear', align_corners=False)
                neural_later_tensor = _out.reshape(-1, 3 * 32 * 32)
                if config.save == 1:
                    neural_later_name = f"samples_neural_later_{config.dataset}_{config.experiment_name}_{config.sampler_type}.pt"
                    torch.save({"samples": neural_later_tensor}, os.path.join(samplesdir, neural_later_name))
                    logger.info(f"Neural later-steps samples saved to {os.path.join(samplesdir, neural_later_name)}")
                plot_rgb_grid(neural_later_tensor[:max_vis_samples], dimen=(32, 32), grid_size=(10, 10),
                              imgname=f"neural_later_samples_prop{config.prop}_seed{config.seed}", imgdir=imgdir,
                              sigma_min=config.sigma_min, sigma_max=config.sigma_max)

            plot_rgb_grid(hybrid_sample_tensor[:max_vis_samples], dimen=(32, 32), grid_size=(10, 10),
                              imgname=f"hybrid_samples_prop{config.prop}_seed{config.seed}", imgdir=imgdir,
                              sigma_min=config.sigma_min, sigma_max=config.sigma_max)
            plot_rgb_grid(one_step_sample_tensor[:max_vis_samples], dimen=(32, 32), grid_size=(10, 10),
                              imgname=f"hybrid_one_step_samples_prop{config.prop}_seed{config.seed}", imgdir=imgdir,
                              sigma_min=config.sigma_min, sigma_max=config.sigma_max)

            # neural_transformation = [convert_64_to_32(step) for step in neural_transformation]

            # combined_transformations = neural_transformation + sample_transformation_hybrid
            # combined_timesteps = neural_timesteps + timesteps_hybrid

            # proportion_indices = np.linspace(0, len(combined_transformations) - 1, 10 + 1, dtype=int)
            # combined_transformations = [combined_transformations[idx] for i, idx in enumerate(proportion_indices)]
            # combined_timesteps = [combined_timesteps[idx] for i, idx in enumerate(proportion_indices)]

            # plot_diffusion_grid(combined_transformations,
            #                     combined_timesteps,
            #                     rgb = True,
            #                     dimen=(32, 32),
            #                     imgname=f"hybrid_denoiser_{config.sampler_type}_{config.experiment_name}_prop{config.prop}",
            #                     imgdir=imgdir,
            #                     sigma_max=config.sigma_max,
            #                     sigma_min=config.sigma_min)
