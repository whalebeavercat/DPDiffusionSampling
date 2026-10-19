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
from cm.unet import UNetModel
import dnnlib
from empirical_denoiser import CIFAR10NeuralDenoiser, EmpiricalDenoiser, ImageNet64NeuralDenoiser, ImageNet32NeuralDenoiser, RandomFourierFeatureDenoiser, SubSampleEmpiricalDenoiser
from noise_scheduler import NoiseScheduler
from utils.diffusion_utils import generate_samples_standard, generate_samples_with_row_subsampling, get_celeba_dataset_64, get_cifar10_dataset, get_cifar10_dataset_64, get_mnist_dataset, get_moons_dataset, perform_PCA_2d
from utils.visualization import (
    plot_mnist_grid, 
    plot_rgb_grid, 
    plot_diffusion_grid, 
    plot_2d_graph,
)
from utils.logger import DPLogger, LogLevel
import gc
import sys
from torch_utils import distributed as dist

def get_torch_device(device_str, logger=None):
    """Convert device string to valid PyTorch device."""
    if device_str.lower() in ["gpu", "cuda"]:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            if logger:
                logger.warning("GPU requested but not available, falling back to CPU")
            else:
                print("Warning: GPU requested but not available, falling back to CPU")
            return torch.device("cpu")
    return torch.device("cpu")

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dataset", type=str, default="moons", choices=["circle", "dino", "line", "moons", "mog", "mnist", "cifar10", "imagenet32", "imagenet64"])
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--sampler_type", type=str, default='sde', choices = ["sde", "ode", "clipsde", "edm"])
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=80.)
    parser.add_argument("--save", type=int, default=0)
    parser.add_argument("--make2d", type=int, default=0)
    parser.add_argument("--beta_hyper", type=float, default=None)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples is L in our write up")
    parser.add_argument("--batch_size", type=int, default=2000, help="Batch size is m in our write up")
    parser.add_argument("--subsample", type=int, default=0)
    parser.add_argument("--row_subsample", type=int, default=0, help="Use row-wise subsampling (1=yes, 0=no).  Works with --compare_epsilons=1.")
    parser.add_argument("--samples_per_seed", type=int, default=1000, help="Number of samples to generate per seed")
    parser.add_argument("--verbosity", type=int, default=3, help="Verbosity level: 0=silent, 1=errors, 2=warnings, 3=info, 4=debug, 5=trace")
    parser.add_argument("--non_dp", type=int, default=0, help="Use non-DP schedule (1=yes, 0=no)")
    parser.add_argument("--rho", type=float, default=7.0, help="Rho value for Karras schedule")
    parser.add_argument("--batch_seed", type=int, default=0, help="Specific seed for data loader batch sampling")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model (if any)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    config = parser.parse_args()

    logger = DPLogger(name="dp-denoiser", log_to_file=True, verbosity=config.verbosity)
    device = get_torch_device(config.device, logger=logger)
    os.makedirs(config.outdir, exist_ok=True)
    imgdir = f"{config.outdir}/images_app"
    os.makedirs(imgdir, exist_ok=True)
    samplesdir = f"{config.outdir}/samples"
    os.makedirs(samplesdir, exist_ok=True)
    
    # set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    torch.backends.cudnn.benchmark = True
    n_samples = -1
    # for subsampling, set the see (always sample one, but with a different seed number, up to L (num_samples))
    if (config.dataset == "mnist"):
        data_loader, n_samples = get_mnist_dataset(n = config.batch_size)
    elif (config.dataset == "cifar10"):
        if (config.subsample):
            data_loader, n_samples = get_cifar10_dataset(n = config.batch_size, shuffle = True)
        else:
            data_loader, n_samples = get_cifar10_dataset(n = config.batch_size, shuffle = False)
    # Set others for moon for now
    else:
        data_loader, n_samples = get_moons_dataset(n = config.batch_size)

    # Get the dataset sample or initial sample for SubSampleDenoiser
    dataset, init_label = next(iter(data_loader))

    dataset = dataset.to(device)
    
    if (config.make2d and (config.dataset == "mnist" or config.dataset == "cifar10")):
        dimen = 2
        reduced_dataset, _ = perform_PCA_2d(dataset)
        pfode = EmpiricalDenoiser(
            reduced_dataset, 
            data_loader=data_loader,
            clipping = (config.sampler_type == "clipsde"), 
            n_dim = dimen, 
            C = config.C, 
            beta_hyper = config.beta_hyper,
            non_dp=config.non_dp,
            device=device
        )
    elif (config.model_path is not None):
        # Load model from pickle or PyTorch checkpoint
        if config.model_path.endswith('.pkl'):
            with dnnlib.util.open_url(config.model_path, verbose=(dist.get_rank() == 0)) as f:
                model_data = pickle.load(f)['ema'].to(device)
        else:
            raise ValueError(f"Unsupported model file format: {config.model_path}. Use .pkl, .pt, or .pth")
        
        if (config.dataset == "cifar10"):
            dimen = 3072 # 32x32x3
            pfode = CIFAR10NeuralDenoiser(
                model_data,
                device = device,
            )
        elif (config.dataset == "imagenet64"):
            dimen = 64*64*3
            pfode = ImageNet64NeuralDenoiser(
                model_data,
                device = device,
            )
        elif (config.dataset == "imagenet32"):
            dimen = 32*32*3
            pfode = ImageNet32NeuralDenoiser(
                model_data,
                device = device,
            )
        else:
            raise NotImplementedError(f"Denoiser model for dataset {config.dataset} not implemented.")
    else:
        dimen = dataset.shape[1]
        if (config.subsample == 1):
            pfode = SubSampleEmpiricalDenoiser(
                dataset, 
                data_loader, 
                clipping = (config.sampler_type == "clipsde"), 
                n_dim = dimen, 
                C = config.C,
                non_dp=config.non_dp,
                device = device
            )
        else:
            pfode = EmpiricalDenoiser(
                dataset,
                data_loader=data_loader,
                clipping = True, 
                n_dim = dimen, 
                C = config.C,
                beta_hyper = config.beta_hyper,
                non_dp=config.non_dp,
                is_full_dataset=True,
                device=device
            )

    noise_scheduler = NoiseScheduler(sigma_min=config.sigma_min, sigma_max=config.sigma_max, num_timesteps = config.num_timesteps)
    sigmas_for_inference = noise_scheduler.get_sigmas_karras(
        config.num_timesteps, 
        sigma_max=config.sigma_max, 
        sigma_min=config.sigma_min,
        rho=config.rho,
    )

    print("Sigma schedule:", sigmas_for_inference)

    if config.row_subsample == 1 and config.non_dp == 1:
        logger.info("Row-wise subsampling with non-DP schedule")
        sample_tensor = generate_samples_with_row_subsampling(
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
            save=config.save,
            experiment_name=config.experiment_name,
            samplesdir=samplesdir,
            imgdir=imgdir,
            logger=logger,
            config=config,
            device=device,
        )
        # FOR TESTING
        # Visualize 2d graph between the samples and the dataset
        plot_2d_graph(perform_PCA_2d, config, dataset, init_label, sample_tensor["samples"], imgdir)
    else:        
        sampling_results = generate_samples_standard(
            denoiser=pfode,
            dimen=dimen,
            num_samples=config.num_samples,
            batch_size=config.batch_size,
            dataset_name=config.dataset,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            num_timesteps=config.num_timesteps,
            sampler_type=config.sampler_type,
            C=config.C,
            rho=config.rho,
            save=config.save,
            experiment_name=config.experiment_name,
            samplesdir=samplesdir,
            imgdir=imgdir,
            config=config,
            device=device,
            logger=logger
        )
        
        sample_tensor = sampling_results["samples"]
        sample_transformation = sampling_results["sample_transformation"]
        timesteps = sampling_results["timesteps"]
        sigma_beta_t = sampling_results["sigma_beta_t"]

        # Visualize 2d graph between the samples and the dataset
        plot_2d_graph(perform_PCA_2d, config, dataset, init_label, sample_tensor, imgdir)
        
        # Visualize denoiser intermediate steps and samples
        if (not config.make2d):
            # Only show top 100 samples for visualization
            num_samples = min(100, config.num_samples)
            if (config.dataset == "mnist"):
                plot_diffusion_grid(sample_transformation, timesteps, dimen=(28, 28), imgname=f"denoiser_{config.sampler_type}_{config.experiment_name}", imgdir=imgdir, sigma_max=config.sigma_max, sigma_min=config.sigma_min)
                plot_mnist_grid(sample_tensor[:num_samples], dimen=(28, 28), grid_size=(10, 10), 
                        imgname=f"mnist_samples_{config.sampler_type}_{config.experiment_name}", imgdir=imgdir,
                        sigma_min=config.sigma_min, sigma_max=config.sigma_max)
            if (config.dataset == "cifar10"):
                # Get indices for 10 evenly spaced elements
                # proportion_indices = np.linspace(0, config.num_timesteps - 1, 10 + 1, dtype=int)

                # sample_transformation_short = [sample_transformation[idx] for i, idx in enumerate(proportion_indices)]
                # timesteps_short = [timesteps[idx] for i, idx in enumerate(proportion_indices)]
                sigmas_for_inference = noise_scheduler.get_sigmas_karras(
                    config.num_timesteps, 
                    sigma_max=config.sigma_max, 
                    sigma_min=config.sigma_min,
                    rho=config.rho,
                )
                
                deltaT = noise_scheduler.get_sigma_diffs(sigmas_for_inference)
                
                noise_scales = []
                for i, sigma in enumerate(sigmas_for_inference[:-1]):
                    noise_scale = sigma * torch.sqrt(sigma / deltaT[i])
                    noise_scales.append(noise_scale)

                plot_diffusion_grid(sample_transformation, timesteps, rgb = True, dimen=(32, 32), imgname=f"denoiser_{config.sampler_type}_{config.experiment_name}", imgdir=imgdir, sigma_max=config.sigma_max, sigma_min=config.sigma_min)
                plot_rgb_grid(sample_tensor[:num_samples], dimen=(32, 32), grid_size=(10, 10),
                            imgname=f"cifar_samples_{config.sampler_type}_{config.experiment_name}", imgdir=imgdir,
                            sigma_min=config.sigma_min, sigma_max=config.sigma_max)
            if (config.dataset == "imagenet64"):
                plot_rgb_grid(sample_tensor[:num_samples], dimen=(64, 64), grid_size=(10, 10),
                            imgname=f"imagenet64_samples_{config.sampler_type}_{config.experiment_name}", imgdir=imgdir,
                            sigma_min=config.sigma_min, sigma_max=config.sigma_max)
            if (config.dataset == "imagenet32"):
                plot_rgb_grid(sample_tensor[:num_samples], dimen=(32, 32), grid_size=(10, 10),
                            imgname=f"imagenet32_samples_{config.sampler_type}_{config.experiment_name}", imgdir=imgdir,
                            sigma_min=config.sigma_min, sigma_max=config.sigma_max)
                plot_diffusion_grid(sample_transformation, timesteps, rgb = True, dimen=(32, 32), imgname=f"denoiser_{config.sampler_type}_{config.experiment_name}", imgdir=imgdir, sigma_max=config.sigma_max, sigma_min=config.sigma_min)
            
        plt.close()