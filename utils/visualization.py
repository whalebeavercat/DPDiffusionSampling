import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_mnist_grid(images, dimen = (28, 28), grid_size=(10, 10), 
                    imgname = "mnist_samples", imgdir = "./images_app", 
                    sigma_max = 80., sigma_min = 0.01, 
                    max_samples=100):
    fig, axes = plt.subplots(*grid_size, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(images):
            reshape_image = images[i].reshape(dimen)
            ax.imshow(reshape_image, cmap = 'gray', vmin = -1.0, vmax = 1.0)
            ax.axis('off')
        else:
            ax.axis('off')
    plt.suptitle(f"MNIST with sigma_max: {sigma_max} and sigma_min: {sigma_min}", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{imgdir}/{imgname}.png", dpi=300)
    plt.close()

def plot_rgb_grid(images, dimen = (32, 32), grid_size=(10, 10), 
                      imgname = "cifar_samples", imgdir = "./images_app", 
                      sigma_max = 80., sigma_min = 0.01,
                      max_samples=100):
    fig, axes = plt.subplots(*grid_size, figsize=(10, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(images):
            reshape_image = vector_to_RGB(images[i], dimen)
            ax.imshow(reshape_image)
            ax.axis('off')
        else:
            ax.axis('off')
    plt.suptitle(f"CIFAR10 with sigma_max: {sigma_max} and sigma_min: {sigma_min}", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{imgdir}/{imgname}.png", dpi=300)
    plt.close()

def plot_beta_sigma_graph(
        betas, 
        timestamp, 
        lower_bounds=None, 
        upper_bounds=None, 
        imgname = "beta_sigma", 
        imgdir = "./images_app"
    ):
    fig, ax = plt.subplots()
    if (len(betas) == 0):
        return
    
    # add the valid region given the bounds
    if lower_bounds is not None and upper_bounds is not None:
        ax.fill_between(
            timestamp,
            lower_bounds,
            upper_bounds,
            color="green",
            alpha=0.2,
            label="Valid β region"
        )

    max_beta = max([max(beta) for beta in betas])
    for i in range(len(betas)):
        ax.plot(timestamp, betas[i], marker='o', linestyle='-', label=f'Example {i+1}')
    ax.set_yscale('log')
    ax.set_xlabel('Sigma value (t)')
    # ax.set_xscale('log')
    ax.set_ylabel('Beta value')
    ax.set_title('Sigma vs. Beta')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    fig.tight_layout()
    fig.savefig(f"{imgdir}/{imgname}.png", dpi=300)
    plt.close(fig)

def plot_sigma_delta_sigma(sigmas, delta_sigmas, imgname = "beta_sigma", imgdir = "./images_app"):
    fig, ax = plt.subplots()
    if (len(sigmas) == 0):
        return
    ax.plot(sigmas, delta_sigmas, marker='o', linestyle='-')
    ax.set_xlabel('Sigma value (t)')
    ax.set_ylabel('Delta Sigma value')
    ax.set_title('Sigma vs. Delta Sigma')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    fig.tight_layout()
    fig.savefig(f"{imgdir}/{imgname}.png", dpi=300)

def plot_diffusion_grid(transformations, timestamp, rgb = False, 
                        dimen = (32, 32), imgname = "diffusion_viz", 
                        imgdir = "./images_app", sigma_max = 80., sigma_min = 0.01,
                        max_samples=10):
    T = len(transformations)
    if (T < 0):
        return
    N = min(transformations[0].shape[0], max_samples)

    fig, axs = plt.subplots(N, T, figsize=(T * 2, N * 2))

    for i in range(N):         
        for j in range(T):     
            ax = axs[i, j] if N > 1 else axs[j] 
            img = transformations[j][i]
            if (rgb):
                reshape_image = vector_to_RGB(img, dimen)
                ax.imshow(reshape_image)
            else:
                reshape_image = img.reshape(dimen)
                ax.imshow(reshape_image, cmap = 'gray', vmin = -1.0, vmax = 1.0)
            ax.axis("off")
            if i == 0:
                ax.set_title(f"t={round(timestamp[j].item(), 3)}")

    plt.suptitle(f"Diffusion process with sigma_max: {sigma_max} and sigma_min: {sigma_min}", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{imgdir}/{imgname}.pdf", dpi=300, format='pdf', bbox_inches='tight')
    plt.close()

def vector_to_RGB(image, dimen):
    flatten_size = dimen[0] * dimen[1]
    reshape_image = (image + 1)/2
    red = reshape_image[0:flatten_size].reshape(dimen)
    green = reshape_image[flatten_size: flatten_size * 2].reshape(dimen)
    blue = reshape_image[flatten_size * 2: ].reshape(dimen)
    reshape_image = torch.stack((red, green, blue), dim=2).cpu().numpy()
    reshape_image = np.clip(reshape_image, 0, 1)
    return reshape_image

def plot_2d_graph(perform_PCA_2d, config, dataset, init_label, samples, imgdir = "./images_app"):
    # return
    plt.figure(figsize=(16, 10))
    plt.xticks([])
    plt.yticks([])
    if (config.dataset == "moons"):
        # Only apply to the moons dataset
        xmin, xmax = -1.1, 1.1
        ymin, ymax = -1.05, 0.9
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

    data_2d, vector_space = perform_PCA_2d(dataset.detach().clone())
    if (config.dataset in ["mnist", "cifar10"]):
        plt.scatter(data_2d.cpu().numpy()[:, 0], data_2d.cpu().numpy()[:, 1], 5, c = init_label, cmap = 'tab10', alpha=0.8)
    else:
        plt.scatter(dataset.cpu().numpy()[:, 0], dataset.cpu().numpy()[:, 1], 5, color='black', alpha=0.8)
        plt.scatter(100, 100, 100, color='black', label='Data')

    if (config.dataset == "mnist" and not config.make2d):
        pass
        # samples_2d = torch.matmul(samples.detach().clone() - mean_dataset, vector_space)
        # plt.scatter(samples_2d.cpu().numpy()[:, 0], samples_2d.cpu().numpy()[:, 1], facecolors='none', edgecolors='red', s=25, alpha=0.3)
    else:
        plt.scatter(samples.cpu().numpy()[:,
                                           0], samples.cpu().numpy()[:, 1], facecolors='none', edgecolors='red', s=25, alpha=0.3)
        plt.scatter(100, 100, s=100, color='red', label=(f'{config.sampler_type.upper()} Samples'), alpha=0.5)

    plt.legend(loc='lower left', fontsize=40, frameon=False)
    plt.tight_layout()
    if config.save:
        name = f"{config.sampler_type}_{config.experiment_name}_toy"
        plt.savefig(f"{imgdir}/{name}.pdf")
        plt.savefig(f"{imgdir}/{name}.png", dpi=300)
    else:
        plt.show()
    plt.close()


def plot_epsilon_vs_beta(betas, epsilons, min_beta_index, condition_I, condition_II, m, n, delta, k, outpath):
    plt.figure(figsize=(10, 6))
    
    valid_indices = []
    invalid_indices = []
    
    for i, beta in enumerate(betas):
        if condition_II.item() <= beta <= condition_I.item():
            valid_indices.append(i)
        else:
            invalid_indices.append(i)
    
    if invalid_indices:
        invalid_eps = [epsilons[i] for i in invalid_indices]
        invalid_beta = [betas[i] for i in invalid_indices]
        plt.plot(invalid_eps, invalid_beta, 'bo-', alpha=0.5, label=r"$\beta$ vs. $\varepsilon$ (Invalid)")
    
    if valid_indices:
        valid_eps = [epsilons[i] for i in valid_indices]
        valid_beta = [betas[i] for i in valid_indices]
        plt.plot(valid_eps, valid_beta, 'go-', linewidth=3, markersize=8, label=r"$\beta$ vs. $\varepsilon$ (Valid)")
        
        if valid_indices:
            valid_min_idx = np.argmin([epsilons[i] for i in valid_indices])
            min_valid_eps = epsilons[valid_indices[valid_min_idx]]
            min_valid_beta = betas[valid_indices[valid_min_idx]]
            plt.plot(min_valid_eps, min_valid_beta, 'rs', markersize=10, label=f"Min $\\varepsilon$ (Valid) = {min_valid_eps:.2f}, $\\beta$ = {min_valid_beta:.2e}")
    
    plt.axhline(condition_I.item(), color='green', linestyle='--', label=f"Upper Bound (Cond. I) = {condition_I.item():.2e}")
    plt.axhline(condition_II.item(), color='red', linestyle='--', label=f"Lower Bound (Cond. II) = {condition_II.item():.2e}")
    
    plt.axhspan(condition_II.item(), condition_I.item(), alpha=0.1, color='green', label="Valid β Region")
    
    plt.plot(epsilons[min_beta_index], betas[min_beta_index], 'ks', markersize=8, label=f"Overall Min $\\varepsilon$ = {epsilons[min_beta_index]:.2f}, $\\beta$ = {betas[min_beta_index]:.2e}")

    plt.ylabel(r"$\beta$")
    plt.xlabel(r"$\varepsilon$")
    plt.yscale('log')
    plt.title(f"Optimal β vs. ε for m={m}, n={n}, δ={delta:.0e}, k={k}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_beta_bounds_given_eps(epsilon_range, beta_lb_vals, beta_ub_vals, m, outpath):
    plt.figure(figsize=(6, 5))
    plt.plot(epsilon_range, beta_ub_vals, 'bs-', label='Upper Bound (Cond. 3)')
    plt.plot(epsilon_range, beta_lb_vals, 'r^-', label='Lower Bound (Cond. 2)')
    plt.yscale('log')
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r'$\beta$')
    plt.title(f'Bounds on β for batch size = {m}')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_lambda_vs_beta(betas, lambdas, condition_I, condition_II, outpath):
    """
    Plot lambda values vs. beta values
    
    Args:
        betas: List of beta values
        lambdas: List of corresponding lambda values
        condition_I: Upper bound for beta (condition I)
        condition_II: Lower bound for beta (condition II)
        outpath: Output path for the plot
    """
    plt.figure(figsize=(10, 6))
    
    valid_data = [(lam, beta) for lam, beta in zip(lambdas, betas) if not np.isnan(lam)]
    
    if valid_data:
        lam_vals, beta_vals = zip(*valid_data)
        
        valid_indices = [i for i, beta in enumerate(beta_vals) 
                        if condition_II.item() <= beta <= condition_I.item()]
        invalid_indices = [i for i, beta in enumerate(beta_vals) 
                          if beta < condition_II.item() or beta > condition_I.item()]
        
        if invalid_indices:
            invalid_lam = [lam_vals[i] for i in invalid_indices]
            invalid_beta = [beta_vals[i] for i in invalid_indices]
            plt.scatter(invalid_beta, invalid_lam, color='blue', alpha=0.5, 
                       label=r"$\lambda$ for invalid $\beta$")
        
        if valid_indices:
            valid_lam = [lam_vals[i] for i in valid_indices]
            valid_beta = [beta_vals[i] for i in valid_indices]
            plt.scatter(valid_beta, valid_lam, color='green', 
                       label=r"$\lambda$ for valid $\beta$")
            
            sorted_indices = np.argsort(valid_beta)
            sorted_beta = [valid_beta[i] for i in sorted_indices]
            sorted_lam = [valid_lam[i] for i in sorted_indices]
            plt.plot(sorted_beta, sorted_lam, 'g-', alpha=0.7)
        
        plt.axvline(condition_I.item(), color='green', linestyle='--', 
                   label=f"Upper Bound (Cond. I) = {condition_I.item():.2e}")
        plt.axvline(condition_II.item(), color='red', linestyle='--', 
                   label=f"Lower Bound (Cond. II) = {condition_II.item():.2e}")
        
        plt.axvspan(condition_II.item(), condition_I.item(), alpha=0.1, color='green', 
                   label="Valid β Region")
        
        plt.xlabel(r"$\beta$")
        plt.ylabel(r"$\lambda$")
        plt.title(r"Optimal $\lambda$ vs. $\beta$")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(outpath, dpi=300)
        plt.close()
    else:
        print("No valid lambda data to plot")

def plot_beta_schedules_comparison(
        timesteps, 
        beta_schedules, 
        epsilon_values, 
        optimal_epsilons=None,
        lower_bounds=None, 
        upper_bounds=None,
        regular_beta=None,
        imgname="beta_comparison", 
        imgdir="./images_app"
    ):
    """
    Plot multiple beta schedules for different epsilon values for comparison
    
    Args:
        timesteps: List of timestep values
        beta_schedules: Dictionary mapping epsilon values to beta schedules
        epsilon_values: List of epsilon values to include in the plot
        optimal_epsilons: Dictionary mapping epsilon keys to achieved epsilon values
        lower_bounds: Lower bounds for beta values
        upper_bounds: Upper bounds for beta values
        regular_beta: Regular beta values from direct kernel computation (no DP)
        imgname: Base filename for the output image
        imgdir: Directory to save the image
    """
    # creating two plots - one log-log and one linear
    for log_scale in [True, False]:
        plt.figure(figsize=(12, 8))
        
        if lower_bounds is not None and upper_bounds is not None:
            plt.fill_between(
                timesteps,
                lower_bounds,
                upper_bounds,
                color="green",
                alpha=0.2,
                label="Valid β region"
            )
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(epsilon_values)))
        
        for i, eps in enumerate(epsilon_values):
            key = str(eps)
            if key in beta_schedules:
                beta_schedule = beta_schedules[key]

                if optimal_epsilons and key in optimal_epsilons:
                    actual_eps = optimal_epsilons[key]
                    label = f'Target ε={eps} (Achieved/Optimal ε={actual_eps:.4f})'
                else:
                    label = f'Target ε={eps}'
                    
                plt.plot(
                    timesteps, 
                    beta_schedule, 
                    'o-', 
                    color=colors[i], 
                    linewidth=2, 
                    label=label
                )
        
        if "non_dp" in beta_schedules:
            plt.plot(
                timesteps, 
                beta_schedules["non_dp"], 
                'k--', 
                linewidth=3, 
                label='Non-DP (bound-based)'
            )
        
        if regular_beta is not None:
            plt.plot(
                timesteps,
                regular_beta,
                'ro-',
                linewidth=2,
                label='Regular kernel β'
            )
        
        plt.xlabel('Timestep (t)')
        plt.ylabel('β_t value')
        plt.title('Beta Schedules for Different Privacy Levels')
        
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
            scale_suffix = "_log"
        else:
            scale_suffix = "_linear"
            
        plt.grid(True, which='both' if log_scale else 'major', linestyle='--', linewidth=0.5)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f"{imgdir}/{imgname}{scale_suffix}.png", dpi=300)
        plt.close()

def plot_sigma_dp_schedules_comparison(
        timesteps,
        sigma_dp_schedules,
        epsilon_values,
        optimal_epsilons=None,
        lower_bound_betas=None,
        upper_bound_betas=None,
        delta_t_list=None,
        imgname="sigma_dp_comparison",
        imgdir="./images_app"
    ):
    """
    Plot multiple sigma_dp schedules for different epsilon values for comparison
    
    Args:
        timesteps: List of timestep values
        sigma_dp_schedules: Dictionary mapping epsilon values to sigma_dp values
        epsilon_values: List of epsilon values to include in the plot
        lower_bound_betas: Lower bounds for beta values (to calculate sigma_dp bounds)
        upper_bound_betas: Upper bounds for beta values (to calculate sigma_dp bounds)
        delta_t_list: Delta t values for each timestep (needed for bound calculation)
        imgname: Base filename for the output image
        imgdir: Directory to save the image
    """
    from utils.dp_analysis import calculate_sigma_dp
    
    # create two plots - one log-log and one linear
    for log_scale in [True, False]:
        plt.figure(figsize=(12, 8))
        
        if lower_bound_betas is not None and upper_bound_betas is not None and delta_t_list is not None:
            t_tensor = torch.tensor(timesteps)
            dt_tensor = torch.tensor(delta_t_list)
            
            # upper bound for sigma_dp comes from lower bound for beta
            # (since they are inversely related)
            sigma_dp_upper_bounds = calculate_sigma_dp(
                torch.tensor(lower_bound_betas), 
                t_tensor, 
                dt_tensor
            )
            
            # lower bound for sigma_dp comes from upper bound for beta
            sigma_dp_lower_bounds = calculate_sigma_dp(
                torch.tensor(upper_bound_betas), 
                t_tensor, 
                dt_tensor
            )
            
            plt.fill_between(
                timesteps,
                sigma_dp_lower_bounds.numpy(),
                sigma_dp_upper_bounds.numpy(),
                color="green",
                alpha=0.2,
                label="Valid σ_dp region"
            )
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(epsilon_values)))
        
        for i, eps in enumerate(epsilon_values):
            key = str(eps)
            if key in sigma_dp_schedules:
                sigma_dp_schedule = sigma_dp_schedules[key]
                if optimal_epsilons and key in optimal_epsilons:
                    actual_eps = optimal_epsilons[key]
                    label = f'Target ε={eps} (Achieved/Optimal ε={actual_eps:.4f})'
                else:
                    label = f'Target ε={eps}'
                plt.plot(
                    timesteps, 
                    sigma_dp_schedule, 
                    'o-', 
                    color=colors[i], 
                    linewidth=2, 
                    label=label
                )
        
        if "non_dp" in sigma_dp_schedules:
            plt.plot(
                timesteps, 
                sigma_dp_schedules["non_dp"], 
                'k--', 
                linewidth=3, 
                label='Non-DP'
            )
        
        plt.xlabel('Timestep (t)')
        plt.ylabel('σ_dp,t value')
        plt.title('Privacy Noise Parameters for Different Privacy Levels')
        
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
            scale_suffix = "_log"
        else:
            scale_suffix = "_linear"
            
        plt.grid(True, which='both' if log_scale else 'major', linestyle='--', linewidth=0.5)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f"{imgdir}/{imgname}{scale_suffix}.png", dpi=300)
        plt.close()

def plot_sigma_dp_vs_beta_comparison(sigma_dp_values, beta_values, timesteps, epsilon, 
                                    lower_bounds=None, upper_bounds=None, imgdir="./images_app"):
    """
    Plot sigma_dp vs beta values with epsilon information
    
    Args:
        sigma_dp_values: List of sigma_dp values
        beta_values: List of beta values  
        timesteps: List of timesteps
        epsilon: Privacy parameter epsilon value
        lower_bounds: Lower bounds for beta (optional)
        upper_bounds: Upper bounds for beta (optional)
        imgdir: Directory to save the image
    """
    import os
    os.makedirs(imgdir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(timesteps, sigma_dp_values, 'b-', linewidth=2, label='σ_dp,t')
    plt.xlabel('Timestep (t)')
    plt.ylabel('σ_dp,t')
    plt.title(f'Privacy Noise Parameters (ε={epsilon:.4f})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    
    if lower_bounds is not None and upper_bounds is not None:
        plt.fill_between(
            timesteps,
            lower_bounds,
            upper_bounds,
            color="green",
            alpha=0.2,
            label="Valid β region"
        )
    
    plt.plot(timesteps, beta_values, 'r-', linewidth=2, label='β_t')
    plt.xlabel('Timestep (t)')
    plt.xscale('log')
    plt.ylabel('β_t')
    plt.yscale('log')
    plt.title(f'Beta Schedule (ε={epsilon:.4f})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{imgdir}/sigma_dp_vs_beta_eps{epsilon:.2f}.png", dpi=300)
    plt.close()


def save_individual_sample_steps(transformations, timestamp, sample_idx, rgb=False, 
                                dimen=(28, 28), imgname="sample", imgdir="./images_app"):
    """
    Save denoising steps for an individual sample
    
    Args:
        transformations: List of transformations for a single sample
        timestamp: List of timestep values
        sample_idx: Index of the sample
        rgb: Whether the image is RGB
        dimen: Dimensions of the image
        imgname: Base filename for the output image
        imgdir: Directory to save the image
    """
    import os
    os.makedirs(f"{imgdir}/individual_samples", exist_ok=True)
    
    T = len(transformations)
    if T <= 0:
        return
        
    fig, axs = plt.subplots(1, T, figsize=(T * 2, 2))
    
    for j in range(T):
        ax = axs[j] if T > 1 else axs
        img = transformations[j]
        if rgb:
            reshape_image = vector_to_RGB(img, dimen)
            ax.imshow(reshape_image)
        else:
            reshape_image = img.reshape(dimen)
            ax.imshow(reshape_image, cmap='gray', vmin=-1.0, vmax=1.0)
        ax.axis("off")
        ax.set_title(f"t={round(timestamp[j].item(), 2)}")
    
    plt.tight_layout()
    plt.savefig(f"{imgdir}/individual_samples/{imgname}_sample{sample_idx}.png", dpi=300)
    plt.close()
