import os
import argparse
from privacy_accounting import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from noise_scheduler import NoiseScheduler
from tqdm import tqdm

if __name__ == "__main__":
    # Performs a grid search for Gaussian DP and creates plots, without subsampling
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=80.)
    parser.add_argument("--batch_size", type=int, default=2000, help="Batch size is m in our write up")
    parser.add_argument("--rho", type=float, default=7.0, help="Rho value for Karras schedule")
    parser.add_argument("--verbosity", type=int, default=3, help="Verbosity level: 0=silent, 1=errors, 2=warnings, 3=info, 4=debug, 5=trace")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Target epsilon value")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta value for GDP sensitivity")
    parser.add_argument("--num_intervals", type=int, default=10, help="Number of intervals in the diffusion process to analyze")
    parser.add_argument("--epsilon_max", type = float, default = 300.0, help="Max epsilon value to plot to")
    parser.add_argument("--sigma_min_grid", type=float, nargs='+', default=[0.002, 0.01, 0.1, 0.5, 1, 5], help="Grid of sigma_min values to search over")
    parser.add_argument("--seed", type=int, default=67, help="Random seed for sampling")
    # parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"])
    config = parser.parse_args()

    privacydir = f"{config.outdir}/privacy/{config.experiment_name}"
    os.makedirs(privacydir, exist_ok=True)

    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # GDP Without subsampling for varying sigma_min

    sigma_min_grid = config.sigma_min_grid
    eps_vals = np.linspace(0.01, config.epsilon_max, int(config.epsilon_max * 10))

    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
    )

    for sigma_min in tqdm(sigma_min_grid, desc="Sigma min grid search for GDP no subsampling"):
        sigma_values = noise_scheduler.get_sigmas_karras(
            config.num_timesteps, 
            sigma_max=config.sigma_max, 
            sigma_min=sigma_min,
            rho=config.rho,
            device=config.device
        )
        sigma_diffs = torch.tensor(noise_scheduler.get_sigma_diffs(sigma_values))
        mus = calculate_mu_value(sigma_values, sigma_diffs, beta = config.beta)

        total_gdp = composition_gdp(mus.cpu())
        print(f"Total mu-GDP value from sigma_max to sigma_min {sigma_min}: {total_gdp}")
        delta_vals = [gdp_to_eps_delta(total_gdp, e) for e in eps_vals]
        plt.plot(eps_vals, delta_vals, label=f"sigma_min={sigma_min}")

    # Visualize epsilon-delta tradeoff without subsampling
    plt.xlabel("ε")
    plt.ylabel("δ(ε)")
    plt.axhline(y=10**(-5), color="red", linestyle=":", linewidth=2)
    plt.yscale('log')
    plt.ylim(1e-10, 1.2)
    plt.title(f"G-DP to (ε, δ)-DP without subsampling with β {config.beta}")
    plt.legend(loc ='upper right')
    plt.grid(True)
    plt.savefig(f"{privacydir}/{config.experiment_name}_epsilon_delta_no_subsampling_plot.png")
    print(f"Saved epsilon-delta no subsampling plot to {privacydir}/{config.experiment_name}_epsilon_delta_no_subsampling_plot.png")
    plt.close()


    # For every interval between sigma_max and timesteps t, calculate the mu-GDP value
    # NOTE: We only schedule once using the argument

    if (config.num_intervals > config.num_timesteps):
        config.num_intervals = config.num_timesteps

    sigma_values = noise_scheduler.get_sigmas_karras(
        config.num_timesteps, 
        sigma_max=config.sigma_max, 
        sigma_min=config.sigma_min,
        rho=config.rho,
        device=config.device
    )

    sigma_diffs = torch.tensor(noise_scheduler.get_sigma_diffs(sigma_values))
    mus = calculate_mu_value(sigma_values, sigma_diffs, beta = config.beta)

    proportion_indices = np.linspace(0, config.num_timesteps - 1, config.num_intervals + 1, dtype=int)
    for i in range(1, len(proportion_indices)):
        t = proportion_indices[i]
        sigma_t = sigma_values[t]
        mus_t = mus[:t + 1]
        total_gdp_t = composition_gdp(mus_t.cpu())
        proportion = t/(config.num_timesteps - 1)
        # print(f"Total mu-GDP value from sigma_max to sigma {sigma_t} with fraction of generation completed {proportion}: {total_gdp_t}")
        delta_vals_t = [gdp_to_eps_delta(total_gdp_t, e) for e in eps_vals]
        plt.plot(eps_vals, delta_vals_t, label=f"{proportion:.2f} Complete at σ={sigma_t:.4f}")

    # Visualize epsilon-delta tradeoff without subsampling
    plt.xlabel("ε")
    plt.axhline(y=10**(-5), color="red", linestyle=":", linewidth=2)
    plt.ylabel("δ(ε)")
    plt.yscale('log')
    plt.ylim(1e-10, 1.2)
    plt.title(f"G-DP to (ε, δ)-DP without subsampling with β {config.beta}, sigma_min={config.sigma_min}")
    plt.legend(loc ='upper right')
    plt.grid(True)
    plt.savefig(f"{privacydir}/{config.experiment_name}_epsilon_delta_no_subsampling_fraction_plot.png")
    print(f"Saved epsilon-delta no subsampling plot to {privacydir}/{config.experiment_name}_epsilon_delta_no_subsampling_fraction_plot.png")
    plt.close()