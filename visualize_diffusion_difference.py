import argparse
import os
from diffusion_2d import generate_samples_standard, get_cifar10_dataset, get_mnist_dataset
from empirical_denoiser import EmpiricalDenoiser, SubSampleEmpiricalDenoiser
from noise_scheduler import NoiseScheduler
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.visualization import vector_to_RGB


def visualize_no_subsampling(config, dimension, denoiser):
    sigma_min_grid = config.sigma_min_grid
    sampler_type = config.sampler_type

    sigma_min_grid = config.sigma_min_grid

    # Same noise
    
    # Reseed for each sigma_min for same noise
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    sample_size = 1
    base_samples = torch.randn([sample_size, dimension], device=config.device) * config.sigma_max
    
    transformations = []
    for sigma_min in tqdm(sigma_min_grid, desc="Sigma min grid search for diffusion difference"):
        noise_scheduler = NoiseScheduler(
            num_timesteps=config.num_timesteps,
            sigma_min=sigma_min,
            sigma_max=config.sigma_max,
        )

        sigma_values = noise_scheduler.get_sigmas_karras(
            config.num_timesteps, 
            sigma_max=config.sigma_max, 
            sigma_min=sigma_min,
            rho=config.rho,
            device=config.device
        )

        sigmas_diff = noise_scheduler.get_sigma_diffs(sigma_values)
        samples = base_samples.clone().detach()

        timesteps = []
        sample_transformation = []
        
        for i, t in enumerate(tqdm(sigma_values)):
            if (i == 0):
                sample_transformation.append(samples.clone().detach().cpu())
                timesteps.append(t)
            if i < sigma_values.shape[0] - 1:
                with torch.no_grad():
                    residual3, beta_t = denoiser.denoise(samples, t, i=i)
                if sampler_type == 'ode':
                    samples, _ = noise_scheduler.euler_step(residual3, t, samples, i)
                elif sampler_type == 'sde':
                    samples, _ = noise_scheduler.euler_maruyama_step(residual3, t, samples, i)
                elif sampler_type == 'clipsde':
                    samples, _ = noise_scheduler.clipping_reverse_step(
                            denoiser.get_numerator(), beta_t, config.C, t, samples, i)
                else:
                    raise NotImplementedError
                if (i != 0):
                    if (i % (config.num_timesteps // config.num_transforms) == 0) or (i == sigma_values.shape[0] - 2):
                        sample_transformation.append(samples.clone().detach().cpu())
                        timesteps.append(t)
        transformation = {"sigma_min": sigma_min,
                        "transformations": torch.stack(sample_transformation, dim=0),
                        "timesteps": timesteps}
        transformations.append(transformation)

    # Save
    torch.save(transformations, f"./checkpoints/{config.experiment_name}_vary_p_nosubsampling.pt")
    print(f"Saved to ./checkpoints/{config.experiment_name}_vary_p_nosubsampling.pt")
    

def get_data_loader(batch_size, shuffle = False):
    if (config.dataset == "cifar10"):
        data_loader, n_samples = get_cifar10_dataset(n = batch_size, shuffle = shuffle)
        dimension = 3*32*32
    elif (config.dataset == "mnist"):
        data_loader, n_samples = get_mnist_dataset(n = batch_size)
        dimension = 1*28*28
    return data_loader,dimension

def visualize_varying_subsampling(config, device, noise_scheduler, dimension):
    sample_size = 1
    samples = torch.randn([sample_size, dimension], device=config.device) * config.sigma_max
    p_transformations = []
    for p_i in config.p_grid:
            # Create a new denoiser
        num_samples = int(50000 * p_i)
        data_loader_i, dimension = get_data_loader(num_samples, True)
        dataset_i, init_label = next(iter(data_loader_i))
        dataset_i = dataset_i.to(device)
        denoiser = SubSampleEmpiricalDenoiser(dataset_i, 
                                                data_loader_i,
                                                clipping=False, 
                                                n_dim=dimension, 
                                                # C=config.C, 
                                                beta_hyper=config.beta, 
                                                non_dp=True,
                                                device=config.device)
        sigma_values = noise_scheduler.get_sigmas_karras(
                config.num_timesteps, 
                sigma_max=config.sigma_max, 
                sigma_min=config.sigma_min,
                rho=config.rho,
                device=config.device
            )
        samples = samples.detach().clone()
        sample_transformation = []
        timesteps = []
        # Add the same set of noise
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        for i, t in enumerate(tqdm(sigma_values)):
            if i < sigma_values.shape[0] - 1:
                with torch.no_grad():
                    residual3, beta_t = denoiser.denoise(samples, t, i=i)
                if config.sampler_type == 'ode':
                    samples, _ = noise_scheduler.euler_step(residual3, t, samples, i)
                elif config.sampler_type == 'sde':
                    samples, _ = noise_scheduler.euler_maruyama_step(residual3, t, samples, i)
                elif config.sampler_type == 'clipsde':
                    samples, _ = noise_scheduler.clipping_reverse_step(
                                denoiser.get_numerator(), beta_t, config.C, t, samples, i)
                else:
                    raise NotImplementedError
                if (i % (config.num_timesteps // config.num_transforms) == 0) or (i == sigma_values.shape[0] - 2):
                    sample_transformation.append(samples.clone().detach().cpu())
                    timesteps.append(t)
        p_transformation = {"p": p_i,
                            "transformations": torch.stack(sample_transformation, dim=0),
                            "timesteps": timesteps}
        p_transformations.append(p_transformation)

    torch.save(p_transformations, f"./checkpoints/{config.experiment_name}_vary_sigma_min_subsampling.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--sigma_min_grid", type=float, nargs='+', default=[0.002, 0.01, 0.1, 0.5, 0.99], help="Grid of sigma_min values to search over")
    parser.add_argument("--p_grid", type = float, nargs ='+', default = [0.001, 0.01, 0.5], help = "Subsampling rate")
    parser.add_argument("--sigma_min", type = float, default = 0.002)
    parser.add_argument("--sigma_max", type=float, default=80.)
    parser.add_argument("--batch_size", type=int, default=2000, help="Batch size is m in our write up")
    parser.add_argument("--rho", type=float, default=7.0, help="Rho value for Karras schedule")
    parser.add_argument("--verbosity", type=int, default=3, help="Verbosity level: 0=silent, 1=errors, 2=warnings, 3=info, 4=debug, 5=trace")
    parser.add_argument("--seed", type=int, default=67, help="Random seed for sampling")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"])
    parser.add_argument("--subsample", type=int, default=0, help = "To subsample or not")
    parser.add_argument("--sampler_type", type=str, default="sde", choices=["ode", "sde", "clipsde"], help="Sampler type")
    parser.add_argument("--C", type=float, default=1.0, help="Clipping constant for clipsde sampler")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta value for clipsde sampler")
    parser.add_argument("--num_transforms", type=int, default=10, help="Number of images to show in the transformation")
    config = parser.parse_args()

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Use a single noise to create the comparison diagram
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    privacydir = f"./checkpoints/"
    os.makedirs(privacydir, exist_ok=True)

    data_loader, dimension = get_data_loader(config.batch_size, False)

    # Get the dataset sample or initial sample for SubSampleDenoiser
    dataset, init_label = next(iter(data_loader))
    dataset = dataset.to(device)

    if (config.subsample):
        # Same initial noise
        noise_scheduler = NoiseScheduler(
            num_timesteps=config.num_timesteps,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
        )

        visualize_varying_subsampling(config, device, noise_scheduler, dimension)

    else:
        denoiser = EmpiricalDenoiser(dataset, 
                                     data_loader=data_loader,
                                    clipping=False, 
                                    n_dim=dimension, 
                                    # C=config.C, 
                                    beta_hyper=config.beta, 
                                    non_dp=True,
                                    is_full_dataset=False,
                                    device=config.device)
        visualize_no_subsampling(config, dimension, denoiser)

        