# Diffusion-based Sampling with Differential Privacy

This repository contains the code used for differentially private diffusion sampling with empirical and hybrid denoisers. The codebase includes:

- sample generation for MNIST, CIFAR-10, CelebA, and ImageNet variants,
- hybrid sampling that switches between empirical and neural denoisers,
- privacy analysis utilities,
- FID evaluation scripts

## Setup

We recommend Python 3.10+ and a PyTorch installation that matches your CUDA setup.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

MNIST, CIFAR-10, and CelebA are downloaded automatically into `./data` when first used. CelebA is center-cropped to 140x140 and resized to 64x64.

## Main Files

- `diffusion_2d.py`: main entry point for empirical / DP diffusion sampling.
- `hybrid_diffusion.py`: hybrid sampler that combines neural and empirical denoisers.
- `fid/compute_fid.py`: computes FID from saved sample tensors.
- `privacy_accounting.py`: privacy accounting utilities.

## Running Experiments

### 1. Generate samples with the empirical sampler

Example CIFAR-10 run (full-batch):

```bash
python diffusion_2d.py \
  --dataset cifar10 \
  --sampler_type sde \
  --batch_size 256 \
  --num_samples 10000 \
  --num_timesteps 50 \
  --rho 7.0 \
  --sigma_min 0.002 \
  --sigma_max 80 \
  --C 50 \
  --save 1 \
  --beta_hyper 1.0 \
  --outdir ./results/example_run \
  --experiment_name example_run \
  --device cuda
```

### 2. Run the hybrid sampler sdedit

Example hybrid CIFAR-10 run:

```bash
python hybrid_diffusion.py \
  --dataset cifar10 \
  --sampler_type sde \
  --batch_size 256 \
  --num_samples 10000 \
  --subsample_batch_size 25 \
  --prop 0.7 \
  --num_timesteps 39 \
  --rho 7 \
  --sigma_min 0.002 \
  --sigma_max 80 \
  --C 50 \
  --beta_hyper 1.0 \
  --save 1 \
  --row_subsample 1 \
  --model_path ./models/edm-imagenet-64x64-cond-adm.pkl \
  --pretrained64 \
  --outdir ./results/hybrid_example \
  --experiment_name hybrid_example \
  --device cuda \
  --row_subsample 1\
  --sdedit_empirical_neural 1
```

Example hybrid CelebA run, with the empirical denoiser on CelebA-64 and the FFHQ-64 model as the neural denoiser:

```bash
python hybrid_diffusion.py \
  --dataset celeba \
  --sampler_type sde \
  --batch_size 256 \
  --num_samples 10000 \
  --subsample_batch_size 25 \
  --neural_batch_size 500 \
  --prop 0.65 \
  --noise_steps 3 \
  --num_timesteps 39 \
  --rho 7 \
  --sigma_min 0.002 \
  --sigma_max 80 \
  --C 50 \
  --beta_hyper 1.0 \
  --save 1 \
  --model_path ./models/edm-ffhq-64x64-uncond-ve.pkl \
  --pretrained64 \
  --outdir ./results/hybrid_celeba_example \
  --experiment_name hybrid_celeba_example \
  --device cuda \
  --row_subsample 1 \
  --sdedit_empirical_neural 1
```

For ImageNet- and FFHQ-based hybrid runs, a pretrained model checkpoint must be provided via `--model_path`, not included.

### 3. Compute FID

```bash
python fid/compute_fid.py \
  --batch_size 512 \
  --sampledir ./results/example_run/samples/samples_cifar10_example_run_clipsde.pt \
  --fid_dir ./results/example_run/fid \
  --dataset cifar10 \
  --experiment_name example_run
```

Use `--dataset celeba` for CelebA samples. Real statistics are computed on the CelebA train split and cached under `./cache/celeba_cache`.

### 4. Run privacy accounting

Example privacy accounting run:

```bash
python privacy_accounting.py \
  --experiment_name example_run \
  --outdir ./results/example_run \
  --device cpu \
  --num_timesteps 50 \
  --sigma_min 0.01 \
  --sigma_max 80 \
  --batch_size 100 \
  --rho 7.0 \
  --p 0.01 \
  --epsilon 10.0 \
  --beta 1.0 \
  --delta 1e-5
```

## Outputs

Typical outputs are written under the chosen `--outdir`:

- `samples/`: saved sample tensors,
- `fid/`: FID scores and cached evaluation outputs,
- `images_app/`: visualizations of generated samples and denoising trajectories.

## Notes
- Some scripts assume access to a GPU for practical runtimes.
- Cached statistics and intermediate artifacts may be reused across runs through the `cache/` directory.
