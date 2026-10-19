import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, transforms
import torch.nn.functional as F
from tqdm.auto import tqdm
import json

def get_transformed_mnist_dataset(n=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img.repeat(3,1,1))  # gray→RGB
    ])
    real_ds = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    real_loader = DataLoader(real_ds, batch_size=100, shuffle=False)
    return real_loader, len(real_ds)

def get_transformed_cifar10_dataset(n=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    real_ds = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    real_loader = DataLoader(real_ds, batch_size=n, shuffle=False)
    return real_loader, len(real_ds)

def get_transformed_celeba_dataset(n=128):
    # Same preprocessing as get_celeba_dataset_64, minus the [-1,1] normalization
    transform = transforms.Compose([
        transforms.CenterCrop(140),
        transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    real_ds = datasets.CelebA(root='data', split='train', download=True, transform=transform)
    real_loader = DataLoader(real_ds, batch_size=n, shuffle=False)
    return real_loader, len(real_ds)

def prepare_MNIST_inception(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N, 784] float tensor in [-1,1]
    returns: [N, 3, 299, 299] float tensor normalized for InceptionV3
    """
    x = (x + 1)/2

    N = x.shape[0]
    x = x.view(N, 1, 28, 28)
    
    x = x.repeat(1, 3, 1, 1)
    
    return x

def prepare_CIFAR_for_inception(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N, 3072] float tensor in [-1,1] (flattened CIFAR images)
    returns: [N, 3, 299, 299] float tensor normalized for InceptionV3
    """
    x = (x + 1)/2 # Second operation to ensure the x is between 0 and 1
    # print("Value Range (to 0, 1):", x.min(), x.max())
    N = x.shape[0]
    x = x.view(N, 3, 32, 32)  # Reshape flattened tensor to [N, 3, 32, 32]

    return x

def prepare_CelebA_for_inception(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N, 12288] float tensor in [-1,1] (flattened 64x64 CelebA images)
    returns: [N, 3, 64, 64] float tensor in [0,1]
    """
    x = (x + 1)/2
    N = x.shape[0]
    x = x.view(N, 3, 64, 64)

    return x


def compute_and_save_real_stats(dataset_name, batch_size=100, cache_dir='./fid_cache', recompute=False):
    """Pre-compute and save the real image statistics for FID calculation"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{dataset_name}_real_stats.pt")
    
    if os.path.exists(cache_file) and not recompute:
        print(f"✓ Found cached real statistics for {dataset_name}")
        return cache_file
        
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    if dataset_name == "mnist":
        data_loader, _ = get_transformed_mnist_dataset(n=batch_size)
    elif dataset_name == "cifar10":
        data_loader, _ = get_transformed_cifar10_dataset(n=batch_size)
    elif dataset_name == "celeba":
        data_loader, _ = get_transformed_celeba_dataset(n=batch_size)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    print(f"Processing real {dataset_name} images for FID statistics...")
    for batch, _ in tqdm(data_loader, desc="Real images → Feature Space", total=len(data_loader)):
        fid.update(batch.to(device), real=True)
    
    real_stats = {
        'real_features_sum': fid.real_features_sum,
        'real_features_cov_sum': fid.real_features_cov_sum,
        'real_features_num_samples': fid.real_features_num_samples
    }
    
    torch.save(real_stats, cache_file)
    print(f"✓ Saved real statistics to {cache_file}")
    return cache_file

def compute_fid_with_cached_stats(samples, dataset_name, batch_size=128, cache_dir='./fid_cache'):
    """Compute FID using cached statistics for real images"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cache_file = compute_and_save_real_stats(dataset_name, batch_size, cache_dir)
    real_stats = torch.load(cache_file, map_location=device)  # This loads a dictionary, not a tensor
    
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    fid.real_features_sum = real_stats['real_features_sum']
    fid.real_features_cov_sum = real_stats['real_features_cov_sum']
    fid.real_features_num_samples = real_stats['real_features_num_samples']
    
    if dataset_name == "mnist":
        processed_samples = prepare_MNIST_inception(samples.to(device))
    elif dataset_name == "celeba":
        processed_samples = prepare_CelebA_for_inception(samples.to(device))
    else:  # cifar10
        processed_samples = prepare_CIFAR_for_inception(samples.to(device))
        
    batches = torch.split(processed_samples, batch_size)
    for batch in tqdm(batches, desc="Generated samples → Feature Space", total=len(batches)):
        fid.update(batch.to(device), real=False)
    
    fid_value = fid.compute()
    return fid_value

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser('')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size per GPU')
    parser.add_argument('--sampledir', type=str, default='', help='Directory to load samples')
    parser.add_argument('--fid_dir', type=str, default='.', help='Directory to store fid stats')
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10", "celeba"])
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--force_recompute', action='store_true', help='Force recomputing real stats')

    config = parser.parse_args()

    # sanity check
    if os.path.exists(config.sampledir) and os.path.isfile(config.sampledir):
        try:
            samples = torch.load(config.sampledir, map_location=device, weights_only=True)
            
            # sample format check
            if "samples" not in samples:
                raise ValueError(f"Sample file does not contain 'samples' key. Found keys: {list(samples.keys())}")
            sample_tensor = samples["samples"]
            
            if torch.isnan(sample_tensor).any() or torch.isinf(sample_tensor).any():
                raise ValueError(f"Sample tensor contains NaN or infinite values! Please check data integrity.")
            
            expected_dims = None
            if config.dataset == "mnist":
                expected_dims = 784  # 28*28
            elif config.dataset == "cifar10":
                expected_dims = 3072  # 3*32*32
            elif config.dataset == "celeba":
                expected_dims = 12288  # 3*64*64
                
            if expected_dims and sample_tensor.shape[1] != expected_dims:
                raise ValueError(f"Expected {expected_dims} dimensions for {config.dataset} samples, but got {sample_tensor.shape[1]}")
            
            print(f"✓ Loaded {sample_tensor.shape[0]} samples of dimension {sample_tensor.shape[1]}")
            print(f"✓ Value range: [{sample_tensor.min().item():.4f}, {sample_tensor.max().item():.4f}]")
            print(f"✓ Mean: {sample_tensor.mean().item():.4f}, Std: {sample_tensor.std().item():.4f}")
            
            # show random sample indices for verification
            rand_indices = torch.randint(0, sample_tensor.shape[0], (5,))
            print(f"✓ Random sample indices for verification: {rand_indices.tolist()}")
            print(f"✓ Sample validation passed! Proceeding with FID calculation...")
            
        except Exception as e:
            raise RuntimeError(f"Error validating samples: {str(e)}")
    else:
        raise FileNotFoundError(f"No such file: {config.sampledir}")

    # Use the cached stats computation function instead of recalculating every time
    cache_dir = os.path.join('./cache', f"{config.dataset}_cache")
    os.makedirs(cache_dir, exist_ok=True)

    if config.force_recompute:
        print("Forcing recomputation of real statistics...")
        real_stats_file = compute_and_save_real_stats(
            config.dataset, 
            config.batch_size, 
            cache_dir,
            recompute=True
        )
    else:
        real_stats_file = compute_and_save_real_stats(
            config.dataset, 
            config.batch_size, 
            cache_dir,
            recompute=False
        )

    # Initialize FID with cached real statistics
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    real_stats = torch.load(real_stats_file, map_location=device)

    # Set the statistics in the FID object
    fid.real_features_sum = real_stats['real_features_sum']
    fid.real_features_cov_sum = real_stats['real_features_cov_sum'] 
    fid.real_features_num_samples = real_stats['real_features_num_samples']

    print(f"Using cached real statistics from {real_stats_file}")
    print(f"Real features: {real_stats['real_features_num_samples']} samples with dimension {real_stats['real_features_sum'].shape[0]}")
    
    with torch.no_grad():
        all_samples = samples["samples"]
        # print all_samples stats
        print(f"Sample tensor shape: {all_samples.shape}")
        print(f"Sample tensor value range: [{all_samples.min().item():.4f}, {all_samples.max().item():.4f}]")
        print(f"Sample tensor mean: {all_samples.mean().item():.4f}, std: {all_samples.std().item():.4f}")

        total_samples = all_samples.shape[0]
        
        processing_batch_size = config.batch_size

        num_batches = (total_samples + processing_batch_size - 1) // processing_batch_size
        
        print(f"Processing {total_samples} samples in {num_batches} small batches to avoid memory issues")
        
        for i in tqdm(range(num_batches), desc="Processing sample batches"):
            start_idx = i * processing_batch_size
            end_idx = min((i + 1) * processing_batch_size, total_samples)
            batch = all_samples[start_idx:end_idx].to(device)
            
            if config.dataset == "mnist":
                processed_batch = prepare_MNIST_inception(batch)
            elif config.dataset == "celeba":
                processed_batch = prepare_CelebA_for_inception(batch)
            elif config.dataset == "cifar10":
                # check the range is between 0-1
                processed_batch = prepare_CIFAR_for_inception(batch)
                
                if i <= 2:
                    import torchvision.utils as vutils
                    import os
                    
                    debug_dir = os.path.join(config.fid_dir, "debug_images")
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    orig_images = batch.view(-1, 3, 32, 32)
                    for j in range(min(5, orig_images.shape[0])):
                        vutils.save_image(
                            orig_images[j], 
                            os.path.join(debug_dir, f"original_batch_{i}_{j}.png"),
                            normalize=True,
                        )
                        
                    # save processed 299x299 images
                    for j in range(min(5, processed_batch.shape[0])):
                        vutils.save_image(
                            processed_batch[j], 
                            os.path.join(debug_dir, f"processed_batch_{i}_{j}.png"),
                            normalize=True,
                        )
                    
                    print(f"Debug images saved to {debug_dir}")
                    print(f"Processed image stats: min={processed_batch.min().item():.4f}, max={processed_batch.max().item():.4f}, mean={processed_batch.mean().item():.4f}")
            
            fid.update(processed_batch, real=False)
            
            del batch
            del processed_batch
            torch.cuda.empty_cache()
    
    fid_value = fid.compute()
    print("Final FID Score:", fid_value.item())
    torch.save(fid_value, os.path.join(config.fid_dir, f'{config.experiment_name}_fid_score.pt'))
    # save as txt
    with open(os.path.join(config.fid_dir, f'{config.experiment_name}_fid_score.txt'), 'w') as f:
        f.write(f"FID Score: {fid_value.item()}\n")
        
    #save as json
    with open(os.path.join(config.fid_dir, f'{config.experiment_name}_fid_score.json'), 'w') as f:
        json.dump({"FID Score": fid_value.item()}, f, indent=4)
