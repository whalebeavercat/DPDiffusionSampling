import os
import argparse
import torch
from glob import glob

def consolidate_samples(base_dir, experiment_name, sampler_type, output_name=None):
    """
    Consolidate samples from multiple batch runs into a single tensor file.
    
    Args:
        base_dir: Base directory where sample files are stored
        experiment_name: Base name of the experiment (will match with *_batch*)
        sampler_type: Type of sampler used (sde, ode, etc.)
        output_name: Name for the consolidated file
    """
    sample_pattern = f"{base_dir}/samples/samples_cifar10_{experiment_name}_batch*_{sampler_type}.pt"
    sample_files = glob(sample_pattern)
    
    if not sample_files:
        print(f"No sample files found matching pattern: {sample_pattern}")
        return
    
    print(f"Found {len(sample_files)} sample files to consolidate")
    
    all_samples = []
    total_count = 0
    
    for file_path in sorted(sample_files):
        print(f"Loading samples from {file_path}")
        data = torch.load(file_path)
        batch_samples = data["samples"]
        all_samples.append(batch_samples)
        total_count += batch_samples.shape[0]
        print(f"- Added {batch_samples.shape[0]} samples (total: {total_count})")
    
    consolidated_samples = torch.cat(all_samples, dim=0)
    print(f"Consolidated tensor shape: {consolidated_samples.shape}")
    
    if output_name is None:
        output_name = f"samples_cifar10_{experiment_name}_{sampler_type}.pt"
    
    output_path = os.path.join(base_dir, "samples", output_name)
    
    torch.save({"samples": consolidated_samples}, output_path)
    print(f"Consolidated samples saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate generated samples from parallel runs")
    parser.add_argument("--base_dir", type=str, default="./results/cifar10_sde_50k",
                        help="Base directory containing the sample files")
    parser.add_argument("--experiment_name", type=str, default="non_dp_sde_50k",
                        help="Base experiment name")
    parser.add_argument("--sampler_type", type=str, default="sde",
                        help="Sampler type used")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Name for the output consolidated file")
    
    args = parser.parse_args()
    
    output_path = consolidate_samples(
        args.base_dir, 
        args.experiment_name, 
        args.sampler_type,
        args.output_name
    )
    
    print("\nTo compute FID score, run:")
    print(f"python fid/compute_fid.py \\\n"
          f"    --batch_size 128 \\\n"
          f"    --sampledir \"{output_path}\" \\\n"
          f"    --fid_dir \"./results/{args.base_dir}/fid\" \\\n"
          f"    --dataset cifar10 \\\n"
          f"    --experiment_name \"{args.experiment_name}\"")
