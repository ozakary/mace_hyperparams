# MACE Potential Hyperparameter Optimization for Xe-Water Systems


📄 Authors: **Ouail Zakary**  
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)  
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)  
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)  
- 📁 Portfolio: [GitHub Portfolio](https://ozakary.github.io/)

---

This repository contains the methodology and tools for optimizing a MACE (Equivariant Message Passing Neural Network) architecture for Xenon-water systems.

## Table of Contents
1. [Overview](#overview)
2. [Setup Instructions](#setup-instructions)
3. [Dataset Preparation](#dataset-preparation)
4. [Hyperparameter Testing Strategy](#hyperparameter-testing-strategy)
5. [Testing Workflow](#testing-workflow)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Recommended Parameter Combinations](#recommended-parameter-combinations)
8. [Results Visualization](#results-visualization)
9. [References](#references)

## Overview

This project develops a machine learning interatomic potential (MLIP) for Xenon-water systems using the MACE framework.

The workflow consists of:
1. Generating initial ab initio molecular dynamics (AIMD) data using VASP
2. Exploring hyperparameter space to optimize model performance

## Setup Instructions

### Prerequisites
- Access to CSC's Puhti or Mahti supercomputer
- CSC account with project allocation
- Basic knowledge of SLURM job scheduling

### Installation

1. Connect to Puhti or Mahti:
```bash
ssh your-username@puhti.csc.fi
```

2. Load the required modules:
```bash
module load pytorch/2.2
```

3. Create and activate a virtual environment in the ./projappl/plantto/<user_name>/ directory (if you run out of quota, use the ./scratch/plantto/<user_name>/ directory):
```bash
cd ./projappl/plantto/<user_name>/packages/
# or
cd ./scratch/plantto/<user_name>/packages/
python3 -m venv mace_env
source mace_env/bin/activate
```

4. Install MACE and dependencies:
```bash
pip install --upgrade pip
pip install mace-torch
pip install wandb

# get the MACE repository for the running scripts
git clone https://github.com/ACEsuit/mace.git
```

5. Set up Weights & Biases for tracking experiments:
```bash
wandb login
```
Follow the instructions to complete the login process.

### Project Structure Setup

Create the necessary directory structure:
```bash
cd ..
mkdir -p mace_calcs/hyperparams_tests/cutoff_redius_tests/
cd mace_calcs/hyperparams_tests/cutoff_redius_tests/
```

## Dataset Preparation

1. Upload your AIMD trajectory to the `mlip_data_xe-water/` directory.

```bash
mkdir mlip_data_xe-water
cd mlip_data_xe-water

# Go to your local machine terminal, locate the `mlip_data_xe-water.xyz` file and then upload it to Puhti using:
scp mlip_data_xe-water.xyz <user_nqme>@puhti.csc.fi:./scratch/plantto/<user_name>/mace_calcs/hyperparams_tests/cutoff_redius_tests/mlip_data_xe-water/
```

2. Now back to the supercomputer terminal, split the dataset into training, validation, and testing sets:
```bash
# Load the data analysis python module
module load python-data
python3 code_split.py  # Dataset splitting script
cd ..
```

3. The splitting script should create:
   - `mlip_data_xe-water_train.xyz` (typically 80% of data)
   - `mlip_data_xe-water_valid.xyz` (typically 10% of data)
   - `mlip_data_xe-water_test.xyz` (typically 10% of data)

## Hyperparameter Testing Strategy

Our hyperparameter optimization strategy follows a systematic approach, testing parameters in order of their expected impact on model performance.

### Phase 1: Architecture Parameters

These parameters define the fundamental architecture of the model and have the most significant impact on accuracy and computational cost.

| Parameter | Description | Values to Test | Default |
|-----------|-------------|----------------|---------|
| `num_interactions` | Number of message-passing interactions | 1, 2, 3, 4 | 1 |
| `max_L` | Maximum degree of spherical harmonics | 1, 2, 3 | 1 |
| `correlation` | Order of tensor product correlations | 2, 3, 4 | 3 |
| `hidden_irreps` | Irreducible representations in hidden layers | Various combinations | '128x0e + 128x1o' |

### Phase 2: Interaction Parameters

These parameters define how atoms interact within the cutoff radius.

| Parameter | Description | Values to Test | Default |
|-----------|-------------|----------------|---------|
| `r_max` | Cutoff radius in Ångstroms | 4.0, 5.0, 6.0 | 5.0 |
| `num_radial_basis` | Number of radial basis functions | 8, 12, 16 | 8 |
| `num_cutoff_basis` | Number of cutoff basis functions | 8, 12, 16 | 8 |
| `radial_type` | Type of radial basis | 'bessel', 'gaussian' | 'bessel' |

### Phase 3: Network Size Parameters

These parameters control the capacity and expressivity of the model.

| Parameter | Description | Values to Test | Default |
|-----------|-------------|----------------|---------|
| `num_channels` | Number of channels in equivariant layers | 64, 128, 192, 256 | 128 |
| `MLP_irreps` | Irreps for the MLP output layer | '8x0e', '16x0e' | '8x0e' |
| `radial_MLP` | Architecture of radial MLP | '[32, 64, 128]', '[64, 128, 256]' | '[32, 64, 128]' |

### Phase 4: Training Parameters

These parameters control the optimization process.

| Parameter | Description | Values to Test | Default |
|-----------|-------------|----------------|---------|
| `lr` | Learning rate | 1e-4, 1e-5, 1e-6 | 1e-5 |
| `batch_size` | Batch size for training | 1, 2, 4, 8 | 1 |
| `forces_weight` | Weight of forces in the loss function | 100, 500, 1000, 2000 | 1000 |
| `stress_weight` | Weight of stress in the loss function | 1, 10, 100 | 10 |
| `max_num_epochs` | Maximum number of training epochs | 100, 200, 500 | 100 |

## Testing Workflow

### Step 1: Create Job Scripts

Create a separate job script for each hyperparameter combination you want to test. Use the template provided in this repository and modify the relevant parameters.

Example for testing different cutoff radii:
```bash
vi mlip_xe-water_test-r_max-4.job
vi mlip_xe-water_test-r_max-5.job
vi mlip_xe-water_test-r_max-6.job
```

### Step 2: Initial Testing on gputest

Always start with a short run on the `gputest` partition to ensure your job works correctly:

```bash
sbatch mlip_xe-water_test-r_max-4.job
```

Review the output and error files to ensure everything is working properly.

### Step 3: Full Parameter Sweep

Submit jobs to the `gpu` partition for full training runs:

```bash
# Modify the job script to use the gpu partition and longer time limit
sed -i 's/gputest/gpu/g' mlip_xe-water_test-r_max-4.job
sed -i 's/0-00:15:00/0-12:00:00/g' mlip_xe-water_test-r_max-4.job

# Submit the job
sbatch mlip_xe-water_test-r_max-4.job
```

### Step 4: Track and Analyze Results

Use Weights & Biases to track training progress and compare models:

1. Open your Weights & Biases project in a web browser
2. Compare models based on training and validation errors
3. Look for trends in how different hyperparameters affect model performance

## Evaluation Metrics

We evaluate model performance using several metrics:

1. **Energy RMSE**: Root mean square error of predicted energies (eV)
2. **Forces RMSE**: Root mean square error of predicted forces (eV/Å)
3. **Stress RMSE**: Root mean square error of predicted stress (GPa)
4. **Learning Curve**: How quickly the model converges during training
5. **Computational Efficiency**: Training time per epoch and inference time

## Recommended Parameter Combinations

Based on our preliminary testing, we recommend starting with these parameter combinations:

### Quick Test Model
```
num_interactions=1, max_L=1, num_channels=64, r_max=5.0
```

### Balanced Model
```
num_interactions=2, max_L=2, num_channels=128, r_max=5.0
```

### High-Accuracy Model
```
num_interactions=3, max_L=3, num_channels=256, r_max=6.0
```

## Results Visualization

### Force Component Analysis

Plot predicted vs. reference forces for each component (F<sub>x</sub>, F<sub>y</sub>, F<sub>z</sub>) to identify any systematic errors.

### Energy Scatter Plot

Plot predicted vs. reference energies to assess overall accuracy.

### Error Distribution

Analyze the distribution of errors to identify potential outliers or systematic biases.

### Visualization Commands

```python
import matplotlib.pyplot as plt
import numpy as np

# Example: Plot energy errors
plt.figure(figsize=(10, 8))
plt.scatter(reference_energies, predicted_energies, alpha=0.6)
plt.plot([min(reference_energies), max(reference_energies)], 
         [min(reference_energies), max(reference_energies)], 'r--')
plt.xlabel('Reference Energy (eV)')
plt.ylabel('Predicted Energy (eV)')
plt.title('MACE Potential Energy Prediction')
plt.savefig('energy_prediction.png', dpi=300)
```

## References

1. Batatia, I., Kovács, D.P., Simm, G.N. et al. (2022). MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields. *NeurIPS*. [https://doi.org/10.48550/arXiv.2206.07697](https://doi.org/10.48550/arXiv.2206.07697)

2. Batzner, S., Musaelian, A., Sun, L. et al. (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. *Nature Communications*, 13, 2453. [https://doi.org/10.1038/s41467-022-29939-5](https://doi.org/10.1038/s41467-022-29939-5)

3. Drautz, R. (2019). Atomic cluster expansion for accurate and transferable interatomic potentials. *Physical Review B*, 99, 014104. [https://doi.org/10.1103/PhysRevB.99.014104](https://doi.org/10.1103/PhysRevB.99.014104)

---

For further details, please refer to the respective folders or contact the author via the provided email.
