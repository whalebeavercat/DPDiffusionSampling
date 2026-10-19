import os
import json
import csv
from pathlib import Path
import sys
import torch

def combine_results(root_folder):
    """
    Combine FID scores, privacy metrics, and hyperparameters from subfolders.
    
    Args:
        root_folder: Path to the root folder containing experiment subfolders
    """
    results = []
    
    for subfolder in Path(root_folder).iterdir():
        if not subfolder.is_dir():
            continue
            
        result = {'folder': subfolder.name}
        
        # Extract hyperparameters from folder name
        # Format: cifar10_hybrid_n{num_samples}_bs{batch_size}_prop{proportion}
        folder_name = subfolder.name
        try:
            if '_n' in folder_name and '_bs' in folder_name and '_prop' in folder_name:
                parts = folder_name.split('_')
                for i, part in enumerate(parts):
                    if part == 'n' and i + 1 < len(parts):
                        result['num_samples'] = int(parts[i + 1].replace('bs', '').replace('prop', ''))
                    elif part.startswith('n'):
                        result['num_samples'] = int(part[1:])
                    elif part == 'bs' and i + 1 < len(parts):
                        result['batch_size'] = int(parts[i + 1].replace('prop', ''))
                    elif part.startswith('bs'):
                        result['batch_size'] = int(part[2:])
                    elif part == 'prop' and i + 1 < len(parts):
                        result['proportion'] = int(parts[i + 1])
                    elif part.startswith('prop'):
                        result['proportion'] = int(part[4:])
        except (ValueError, IndexError):
            pass  # Skip if parsing fails
        
        # Load FID scores from fid subfolder
        fid_folder = subfolder / 'fid'
        if fid_folder.exists() and fid_folder.is_dir():
            # Find JSON files in fid subfolder
            for json_file in fid_folder.glob('*_fid_score.json'):
                with open(json_file, 'r') as f:
                    fid_data = json.load(f)
                    # Distinguish between regular and one_step FID scores
                    if 'one_step' in json_file.name:
                        result.update({f'fid_one_step_{k}': v for k, v in fid_data.items()})
                    else:
                        result.update({f'fid_{k}': v for k, v in fid_data.items()})
            
            # If no JSON files found, search for PT files
            if not any(fid_folder.glob('*_fid_score.json')):
                for pt_file in fid_folder.glob('*_fid_score.pt'):
                    fid_data = torch.load(pt_file)
                    # Handle both dict and tensor formats
                    if isinstance(fid_data, dict):
                        if 'one_step' in pt_file.name:
                            result.update({f'fid_one_step_{k}': v for k, v in fid_data.items()})
                        else:
                            result.update({f'fid_{k}': v for k, v in fid_data.items()})
                    else:
                        # If it's a tensor, assume it's the FID score value
                        fid_value = fid_data.item() if fid_data.numel() == 1 else float(fid_data)
                        if 'one_step' in pt_file.name:
                            result['fid_one_step_score'] = fid_value
                        else:
                            result['fid_score'] = fid_value
                
        # Also check for fid_scores.json in root (fallback)
        fid_file = subfolder / 'fid_scores.json'
        if fid_file.exists():
            with open(fid_file, 'r') as f:
                fid_data = json.load(f)
                result.update({f'fid_{k}': v for k, v in fid_data.items()})
        
        # Load privacy metrics and hyperparameters from privacy subfolder
        privacy_folder = subfolder / 'privacy'
        if privacy_folder.exists() and privacy_folder.is_dir():
            # Find JSON files in privacy subfolder
            for json_file in privacy_folder.glob('*_privacy.json'):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    # Extract privacy metrics (epsilon, delta, etc.)
                    for k, v in data.items():
                        if k in ['epsilon', 'delta', 'noise_multiplier', 'sample_rate', 'steps']:
                            result[k] = v
                        elif not k.startswith('fid_'):
                            result[f'privacy_{k}'] = v
        
        # Also check for privacy_metrics.json in root (fallback)
        privacy_file = subfolder / 'privacy_metrics.json'
        if privacy_file.exists():
            with open(privacy_file, 'r') as f:
                privacy_data = json.load(f)
                result.update({f'privacy_{k}': v for k, v in privacy_data.items()})
        
        # Load hyperparameters
        hparams_file = subfolder / 'hyperparameters.json'
        if hparams_file.exists():
            with open(hparams_file, 'r') as f:
                hparams = json.load(f)
                result.update(hparams)
        
        results.append(result)
    
    return results


def write_to_csv(results, output_file='combined_results.csv'):
    """Write combined results to CSV file."""
    if not results:
        print("No results to write")
        return
    
    # Get all unique keys across all results
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())
    
    fieldnames = ['folder'] + sorted([k for k in all_keys if k != 'folder'])
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results written to {output_file}")


if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("Usage: python create_results_csv.py <root_folder> [output_file]")
        sys.exit(1)
    
    root_folder = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'combined_results.csv'
    
    results = combine_results(root_folder)
    write_to_csv(results, output_file)