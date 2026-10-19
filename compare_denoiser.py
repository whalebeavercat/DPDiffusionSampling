import dnnlib
from empirical_denoiser import EmpiricalDenoiser
from noise_scheduler import NoiseScheduler
import torch
import pickle
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import scienceplots
from diffusion_2d import get_cifar10_dataset
from torch_utils import distributed as dist

def load_model_from_pickle(file_path, device):
    with dnnlib.util.open_url(file_path, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)
    return net

def get_xt_from_x0(inputs, t, device):
    noise_flatten = torch.randn_like(inputs).to(device)
    xt_flatten = inputs + noise_flatten * t
    xt = xt_flatten.view(-1, 3, 32, 32).to(device)
    return xt_flatten,xt

def get_denoiser_values(empirical_denoiser, pretrained_model, xt, xt_flatten, t, device):
    pfode, _ = empirical_denoiser.denoise(xt_flatten, t, i=None)
    print("PFODE stats - min:", pfode.min().item(), "max:", pfode.max().item(), "mean:", pfode.mean().item())
    with torch.no_grad():
        pmodel = pretrained_model(xt, torch.tensor([t], device=device)).to(torch.float64)
    pmodel_flatten = pmodel.view(pmodel.size(0), -1)
    print("PModel stats - min:", pmodel_flatten.min().item(), "max:", pmodel_flatten.max().item(), "mean:", pmodel_flatten.mean().item())
    return pfode,pmodel_flatten

def monte_carlos_relative_L2_distance(pfode, pmodel_flatten):
    '''
    Estimate the relative L2 distance between the empirical denoiser and the pretrained model using Monte Carlo sampling.

    Returns:
        Estimated L2 distance at time t.
        Standard deviation of the L2 distance at time t.
    '''
    pmodel_norm = torch.norm(pmodel_flatten, p=2, dim=1, keepdim=True)
    print("Pmodel shape:", pmodel_flatten.shape, "Pmodel norm shape:", pmodel_norm.shape)
    
    epsilon = 1e-8  # Small constant to prevent division by zero
    l2_distances = (torch.norm(pfode - pmodel_flatten, p=2, dim = 1) / (pmodel_norm + epsilon))
    return l2_distances.mean().item(), l2_distances.std().item()


def monte_carlos_cosine_similarity(pfode, pmodel_flatten):
    '''
    Estimate the Cosine similarity between the empirical denoiser and the pretrained model using Monte Carlo sampling.

    Returns:
        Estimated Cosine similarity at time t.
        Standard deviation of the Cosine similarity at time t.
    '''
    cosine_similarity = torch.nn.functional.cosine_similarity(pfode, pmodel_flatten, dim=1)
    return cosine_similarity.sum().item(), cosine_similarity.std().item()

def monte_carlos_difference_cosine_similarity(pfode, pmodel_flatten, xt_flatten):
    '''
    Estimate the Cosine similarity between the empirical denoiser and the pretrained model using Monte Carlo sampling.

    Returns:
        Estimated Cosine similarity at time t.
        Standard deviation of the Cosine similarity at time t.
    '''
    cosine_similarity = torch.nn.functional.cosine_similarity(xt_flatten - pfode, xt_flatten - pmodel_flatten, dim=1)
    return cosine_similarity.sum().item(), cosine_similarity.std().item()

if __name__ == "__main__":
    # Entry point for the script
    parser = argparse.ArgumentParser(description="Compare denoiser models.")
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model pickle file.")
    parser.add_argument('--device', type=str, default='cuda', help="Device to load the model onto.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"])
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for testing the model.")
    parser.add_argument("--beta_hyper", type=float, default=1.0, help="Beta hyperparameter for the empirical denoiser.")
    parser.add_argument("--sigma_min", type=float, default=0.002, help="Minimum sigma for the noise scheduler.")
    parser.add_argument("--sigma_max", type=float, default=80, help="Maximum sigma for the noise scheduler.")
    parser.add_argument("--num_timesteps", type=int, default=50, help="Number of timesteps for the noise scheduler.")
    parser.add_argument("--rho", type=float, default=7.0, help="Rho parameter for the Karras noise schedule.")
    parser.add_argument("--seed", type=int, default=67, help="Random seed for reproducibility.")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for results.")
    config = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Load the dataset
    if (config.dataset == "cifar10"):
        data_loader, n_samples = get_cifar10_dataset(n = config.batch_size, shuffle=False)
        random_loader, n_samples = get_cifar10_dataset(n = config.batch_size, shuffle=True)

    dataset, init_label = next(iter(random_loader))
    dataset = dataset.to(device)

    # Load the pretrained model 
    pretrained_model = load_model_from_pickle(config.model_path, config.device)
    print("Model loaded successfully.")
    
    denoiser = EmpiricalDenoiser(dataset, 
                                data_loader=data_loader,
                                 clipping=False, 
                                 n_dim=3*32*32, 
                                 non_dp=True,
                                 is_full_dataset=True,
                                 device=config.device)
    
    # Set noise scheduler for diffusion based sampling
    noise_scheduler = NoiseScheduler(sigma_min=config.sigma_min, 
                                     sigma_max=config.sigma_max, 
                                     num_timesteps = config.num_timesteps)
    
    sigma_values = noise_scheduler.get_sigmas_karras(
        config.num_timesteps, 
        sigma_max=config.sigma_max, 
        sigma_min=config.sigma_min,
        rho=config.rho,
        device=config.device
    )

    print("Sigma values:", sigma_values)

    distance_results = []
    cosine_similarity_results = []
    diff_cosine_similarity_results = []
    empirical_norm = []
    pretrained_norm = []

    for sigma in tqdm(sigma_values, desc="Calculating L2 distances and cosine similarities"):
        cosine_similarity_sum = 0.0
        diff_cosine_similarity_sum = 0.0
        inputs = dataset
        inputs = inputs.to(device)
        xt_flatten, xt = get_xt_from_x0(inputs, sigma, device)
        pfode, pmodel_flatten = get_denoiser_values(denoiser, pretrained_model, xt, xt_flatten, sigma, device)
        l2_distance_mean, l2_distance_std = monte_carlos_relative_L2_distance(pfode, pmodel_flatten)
        cosine_similarity, cosine_similarity_std = monte_carlos_cosine_similarity(pfode, pmodel_flatten)
        diff_cosine_similarity, diff_cosine_similarity_std = monte_carlos_difference_cosine_similarity(pfode, pmodel_flatten, xt_flatten)
        cosine_similarity_sum += cosine_similarity
        diff_cosine_similarity_sum += diff_cosine_similarity
        cosine_similarity_mean = cosine_similarity_sum / config.batch_size
        diff_cosine_similarity_mean = diff_cosine_similarity_sum / config.batch_size
        distance_results.append((sigma.item(), l2_distance_mean, l2_distance_std))
        cosine_similarity_results.append((sigma.item(), cosine_similarity_mean, cosine_similarity_std))
        diff_cosine_similarity_results.append((sigma.item(), diff_cosine_similarity_mean, diff_cosine_similarity_std))
        empirical_norm.append((sigma.item(), torch.norm(pfode, p=2, dim=1).mean().item()))
        pretrained_norm.append((sigma.item(), torch.norm(pmodel_flatten, p=2, dim=1).mean().item()))
        print(f"Sigma: {sigma.item():.4f}, L2 Distance Relative: {l2_distance_mean:.6f}")

    # AT least save results
    torch.save({
        'distance_results': distance_results,
        'cosine_similarity_results': cosine_similarity_results,
        'diff_cosine_similarity_results': diff_cosine_similarity_results,
    }, f"./{config.outdir}/{config.experiment_name}_denoiser_comparison_results.pt")
    
    plt.figure(figsize=(5,5))
    plt.style.use(['science', "ieee"])
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'DejaVu Serif'

    # Plot distance results for each t
    sigmas, distances, distance_errors = zip(*distance_results)
    plt.errorbar(sigmas, distances, yerr=distance_errors, fmt='-o',  markersize=3, capsize=3, color = "black", ecolor= "black")
    plt.xlabel(r'$\sigma$ (Noise Level)')
    plt.xscale('log')
    plt.ylabel('Relative L2 Distance')
    plt.grid(True)
    plt.savefig(f"./{config.outdir}/{config.experiment_name}_l2_distance_plot.pdf", dpi = 300, format = "pdf", bbox_inches='tight')
    print(f"Saved L2 distance plot to ./{config.outdir}/{config.experiment_name}_l2_distance_plot.pdf")
    plt.close()

    # Plot cosine similarity results for each t
    sigmas, cosine_sims, cosine_sims_std = zip(*cosine_similarity_results)
    plt.errorbar(sigmas, cosine_sims, yerr=cosine_sims_std, fmt='-o', markersize=3, capsize=2, color = "black", ecolor= "black")
    plt.xscale('log')
    plt.xlabel(r'$\sigma$ (Noise Level)')
    plt.ylabel('Cosine Similarity')
    plt.grid(True)
    plt.savefig(f"./{config.outdir}/{config.experiment_name}_cosine_similarity_plot.pdf", dpi = 300, format = "pdf", bbox_inches='tight')
    print(f"Saved Cosine Similarity plot to ./{config.outdir}/{config.experiment_name}_cosine_similarity_plot.pdf")
    plt.close()

    # Plot difference cosine similarity results for each t
    sigmas, diff_cosine_sims, diff_cosine_sims_std = zip(*diff_cosine_similarity_results)
    plt.errorbar(sigmas, diff_cosine_sims, yerr=diff_cosine_sims_std, fmt='-o', markersize=3, capsize=3, color = "black", ecolor= "black")
    plt.xscale('log')
    plt.xlabel(r'$\sigma$ (Noise Level)')
    plt.ylabel('Cosine Similarity')
    plt.grid(True)
    plt.savefig(f"./{config.outdir}/{config.experiment_name}_diff_cosine_similarity_plot.pdf", dpi = 300, format = "pdf", bbox_inches='tight')
    print(f"Saved Difference Cosine Similarity plot to ./{config.outdir}/{config.experiment_name}_diff_cosine_similarity_plot.pdf")
    plt.close()

    # Plot empirical and pretrained norms
    sigmas, emp_norms = zip(*empirical_norm)
    _, pre_norms = zip(*pretrained_norm)
    plt.plot(sigmas, emp_norms, marker='o', label='Empirical Denoiser Norm')
    plt.plot(sigmas, pre_norms, marker='x', label='Pretrained Model Norm')
    plt.xscale('log')
    plt.xlabel(r'$\sigma$ (Noise Level)')
    plt.ylabel('Average L2 Norm')
    plt.title(f'Average L2 Norms of Denoisers, Full Batch')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./{config.outdir}/{config.experiment_name}_denoiser_norms_plot.pdf", dpi = 300, format = "pdf", bbox_inches='tight')
    print(f"Saved Denoiser Norms plot to ./{config.outdir}/{config.experiment_name}_denoiser_norms_plot.pdf")
    plt.close()