# MACE Potential Hyperparameter Optimization for Xe-Water Systems


📄 Author: **Ouail Zakary**  
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)  
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)  
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)  
- 📁 Portfolio: [GitHub Portfolio](https://ozakary.github.io/)

---

This repository contains the methodology and tools for optimizing the hyperparameters of a MACE (Equivariant Message Passing Neural Network) architecture for Xenon-water systems.

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

1. Connect to Puhti (it has V100 GPUs, which is slower than A100 GPUs in Mahti) or Mahti:
```bash
ssh your-username@puhti.csc.fi

# or

ssh your-username@mahti.csc.fi
```

2. Load the required modules:
```bash
module load pytorch/2.0
```

3. Create and activate a virtual environment in the ./projappl/plantto/<user_name>/ directory (if you run out of quota, use the ./scratch/plantto/<user_name>/ directory):
```bash
cd ./projappl/plantto/<user_name>/packages/
# or
cd ./scratch/plantto/<user_name>/packages/

# Create and activate a python virtual environment where all packages will be installed
python3 -m venv mace_env
source mace_env/bin/activate
```

4. Install MACE and dependencies:
```bash
pip install --upgrade pip
pip install wandb

# Install MACE from source
git clone https://github.com/ACEsuit/mace.git
pip install ./mace
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

1. Upload your AIMD trajectory to the `water_xe_dataset/` directory.

```bash
mkdir water_xe_dataset
cd water_xe_dataset

# Go to your local machine terminal, locate the `mlip_data_xe-water_0-5ps_sampled-20.xyz` and `code_split.py` files, and then upload it to Puhti using:
scp mlip_data_xe-water_0-5ps_sampled-20.xyz code_split.py <user_nqme>@puhti.csc.fi:/scratch/plantto/<user_name>/mace_calcs/hyperparams_tests/cutoff_redius_tests/water_xe_dataset/
```

2. Now back to the supercomputer terminal, split the dataset into training, validation, and testing sets:
```bash
python3 code_split.py  # Dataset splitting script
cd ..
```

3. The splitting script should create:
   - `water_xe_dataset_train_val.xyz` (typically 90% of data)
   - `water_xe_dataset_test.xyz` (typically 10% of data)

## Hyperparameter Testing Strategy

Our hyperparameter optimization strategy follows the approach:

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
| `r_max` | Cutoff radius in Ångstroms | 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 | 5.0 |
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
| `energy_weight` | Weight of energy in the loss function | N/A | 1 |
| `forces_weight` | Weight of forces in the loss function | N/A | 1 |
| `stress_weight` | Weight of stress in the loss function | N/A | 10 |
| `max_num_epochs` | Maximum number of training epochs | 100, 500, 1000 | 500 |

## Testing Workflow

### Step 1: Create Job Scripts

Create a separate job script for each hyperparameter combination you want to test. Use the template provided in this repository and modify the relevant parameters.

There sould be two `.job` files per test, the first file concerns the training MACE and the second file is for testing the trained MACE model

Example for testing different cutoff radii:
```bash
# You copy the content of the 'mlip-mace_script_test-0.job' and then adjust the hyperparameters you want to optimize. The following is an example for the parameter 'r_max'
# r_max = 4
vi script_mace_training_test-r_max-4.job
vi script_mace_testing_test-r_max-4.job
# r_max = 5
vi script_mace_training_test-r_max-5.job
vi script_mace_testing_test-r_max-5.job
# r_max = 6
vi script_mace_training_test-r_max-6.job
vi script_mace_testing_test-r_max-6.job
```

### Step 2: Initial Testing on gputest

Start with a small number of epochs (e.g., 20) and always start with a short run on the `gputest` partition to ensure your job works correctly:

```bash
sbatch script_mace_training_test-r_max-4.job
```

Review the output and error files to ensure everything is working properly.

### Step 3: Full Parameter Sweep

Now, for the training `.job` script, increase the number of epochs (e.g., 500 or 1000) and submit jobs to the `gpu` partition for full training runs:

```bash
# Modify the job script to use the gpu partition and longer time limit
sed -i 's/gputest/gpu/g' script_mace_training_test-r_max-4.job
sed -i 's/0-00:15:00/0-36:00:00/g' script_mace_training_test-r_max-4.job

# Submit the job
sbatch script_mace_test-r_max-4.job
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

## Appendix (MACE different commands and options)

```bash
usage: run_train.py [-h] [--config CONFIG] --name NAME [--seed SEED]
                    [--work_dir WORK_DIR] [--log_dir LOG_DIR]
                    [--model_dir MODEL_DIR]
                    [--checkpoints_dir CHECKPOINTS_DIR]
                    [--results_dir RESULTS_DIR]
                    [--downloads_dir DOWNLOADS_DIR]
                    [--device {cpu,cuda,mps,xpu}]
                    [--default_dtype {float32,float64}] [--distributed]
                    [--log_level LOG_LEVEL] [--plot PLOT]
                    [--plot_frequency PLOT_FREQUENCY]
                    [--error_table {PerAtomRMSE,TotalRMSE,PerAtomRMSEstressvirials,PerAtomMAEstressvirials,PerAtomMAE,TotalMAE,DipoleRMSE,DipoleMAE,EnergyDipoleRMSE}]
                    [--model {BOTNet,MACE,ScaleShiftMACE,ScaleShiftBOTNet,AtomicDipolesMACE,EnergyDipolesMACE}]
                    [--r_max R_MAX]
                    [--radial_type {bessel,gaussian,chebyshev}]
                    [--num_radial_basis NUM_RADIAL_BASIS]
                    [--num_cutoff_basis NUM_CUTOFF_BASIS] [--pair_repulsion]
                    [--distance_transform {None,Agnesi,Soft}]
                    [--interaction {RealAgnosticResidualInteractionBlock,RealAgnosticAttResidualInteractionBlock,RealAgnosticInteractionBlock,RealAgnosticDensityInteractionBlock,RealAgnosticDensityResidualInteractionBlock}]
                    [--interaction_first {RealAgnosticResidualInteractionBlock,RealAgnosticInteractionBlock,RealAgnosticDensityInteractionBlock,RealAgnosticDensityResidualInteractionBlock}]
                    [--max_ell MAX_ELL] [--correlation CORRELATION]
                    [--num_interactions NUM_INTERACTIONS]
                    [--MLP_irreps MLP_IRREPS] [--radial_MLP RADIAL_MLP]
                    [--hidden_irreps HIDDEN_IRREPS]
                    [--num_channels NUM_CHANNELS] [--max_L MAX_L]
                    [--gate {silu,tanh,abs,None}]
                    [--scaling {std_scaling,rms_forces_scaling,no_scaling}]
                    [--avg_num_neighbors AVG_NUM_NEIGHBORS]
                    [--compute_avg_num_neighbors COMPUTE_AVG_NUM_NEIGHBORS]
                    [--compute_stress COMPUTE_STRESS]
                    [--compute_forces COMPUTE_FORCES]
                    [--train_file TRAIN_FILE] [--valid_file VALID_FILE]
                    [--valid_fraction VALID_FRACTION] [--test_file TEST_FILE]
                    [--test_dir TEST_DIR]
                    [--multi_processed_test MULTI_PROCESSED_TEST]
                    [--num_workers NUM_WORKERS] [--pin_memory PIN_MEMORY]
                    [--atomic_numbers ATOMIC_NUMBERS] [--mean MEAN]
                    [--std STD] [--statistics_file STATISTICS_FILE]
                    [--E0s E0S]
                    [--foundation_filter_elements FOUNDATION_FILTER_ELEMENTS]
                    [--heads HEADS]
                    [--multiheads_finetuning MULTIHEADS_FINETUNING]
                    [--foundation_head FOUNDATION_HEAD]
                    [--weight_pt_head WEIGHT_PT_HEAD]
                    [--num_samples_pt NUM_SAMPLES_PT]
                    [--force_mh_ft_lr FORCE_MH_FT_LR]
                    [--subselect_pt {fps,random}]
                    [--filter_type_pt {none,combinations,inclusive,exclusive}]
                    [--pt_train_file PT_TRAIN_FILE]
                    [--pt_valid_file PT_VALID_FILE]
                    [--foundation_model_elements FOUNDATION_MODEL_ELEMENTS]
                    [--keep_isolated_atoms KEEP_ISOLATED_ATOMS]
                    [--energy_key ENERGY_KEY] [--forces_key FORCES_KEY]
                    [--virials_key VIRIALS_KEY] [--stress_key STRESS_KEY]
                    [--dipole_key DIPOLE_KEY] [--head_key HEAD_KEY]
                    [--charges_key CHARGES_KEY]
                    [--skip_evaluate_heads SKIP_EVALUATE_HEADS]
                    [--loss {ef,weighted,forces_only,virials,stress,dipole,huber,universal,energy_forces_dipole,l1l2energyforces}]
                    [--forces_weight FORCES_WEIGHT]
                    [--swa_forces_weight SWA_FORCES_WEIGHT]
                    [--energy_weight ENERGY_WEIGHT]
                    [--swa_energy_weight SWA_ENERGY_WEIGHT]
                    [--virials_weight VIRIALS_WEIGHT]
                    [--swa_virials_weight SWA_VIRIALS_WEIGHT]
                    [--stress_weight STRESS_WEIGHT]
                    [--swa_stress_weight SWA_STRESS_WEIGHT]
                    [--dipole_weight DIPOLE_WEIGHT]
                    [--swa_dipole_weight SWA_DIPOLE_WEIGHT]
                    [--config_type_weights CONFIG_TYPE_WEIGHTS]
                    [--huber_delta HUBER_DELTA]
                    [--optimizer {adam,adamw,schedulefree}] [--beta BETA]
                    [--batch_size BATCH_SIZE]
                    [--valid_batch_size VALID_BATCH_SIZE] [--lr LR]
                    [--swa_lr SWA_LR] [--weight_decay WEIGHT_DECAY]
                    [--amsgrad] [--scheduler SCHEDULER]
                    [--lr_factor LR_FACTOR]
                    [--scheduler_patience SCHEDULER_PATIENCE]
                    [--lr_scheduler_gamma LR_SCHEDULER_GAMMA] [--swa]
                    [--start_swa START_SWA] [--lbfgs] [--ema]
                    [--ema_decay EMA_DECAY] [--max_num_epochs MAX_NUM_EPOCHS]
                    [--patience PATIENCE]
                    [--foundation_model FOUNDATION_MODEL]
                    [--foundation_model_readout]
                    [--eval_interval EVAL_INTERVAL] [--keep_checkpoints]
                    [--save_all_checkpoints] [--restart_latest] [--save_cpu]
                    [--clip_grad CLIP_GRAD] [--dry_run]
                    [--enable_cueq ENABLE_CUEQ] [--wandb]
                    [--wandb_dir WANDB_DIR] [--wandb_project WANDB_PROJECT]
                    [--wandb_entity WANDB_ENTITY] [--wandb_name WANDB_NAME]
                    [--wandb_log_hypers WANDB_LOG_HYPERS]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       config file to aggregate options (default: None)
  --name NAME           experiment name (default: None)
  --seed SEED           random seed (default: 123)
  --work_dir WORK_DIR   set directory for all files and folders (default: .)
  --log_dir LOG_DIR     directory for log files (default: None)
  --model_dir MODEL_DIR
                        directory for final model (default: None)
  --checkpoints_dir CHECKPOINTS_DIR
                        directory for checkpoint files (default: None)
  --results_dir RESULTS_DIR
                        directory for results (default: None)
  --downloads_dir DOWNLOADS_DIR
                        directory for downloads (default: None)
  --device {cpu,cuda,mps,xpu}
                        select device (default: cpu)
  --default_dtype {float32,float64}
                        set default dtype (default: float64)
  --distributed         train in multi-GPU data parallel mode (default: False)
  --log_level LOG_LEVEL
                        log level (default: INFO)
  --plot PLOT           Plot results of training (default: True)
  --plot_frequency PLOT_FREQUENCY
                        Set plotting frequency: '0' for only at the end or an
                        integer N to plot every N epochs. (default: 0)
  --error_table {PerAtomRMSE,TotalRMSE,PerAtomRMSEstressvirials,PerAtomMAEstressvirials,PerAtomMAE,TotalMAE,DipoleRMSE,DipoleMAE,EnergyDipoleRMSE}
                        Type of error table produced at the end of the
                        training (default: PerAtomRMSE)
  --model {BOTNet,MACE,ScaleShiftMACE,ScaleShiftBOTNet,AtomicDipolesMACE,EnergyDipolesMACE}
                        model type (default: MACE)
  --r_max R_MAX         distance cutoff (in Ang) (default: 5.0)
  --radial_type {bessel,gaussian,chebyshev}
                        type of radial basis functions (default: bessel)
  --num_radial_basis NUM_RADIAL_BASIS
                        number of radial basis functions (default: 8)
  --num_cutoff_basis NUM_CUTOFF_BASIS
                        number of basis functions for smooth cutoff (default:
                        5)
  --pair_repulsion      use pair repulsion term with ZBL potential (default:
                        False)
  --distance_transform {None,Agnesi,Soft}
                        use distance transform for radial basis functions
                        (default: None)
  --interaction {RealAgnosticResidualInteractionBlock,RealAgnosticAttResidualInteractionBlock,RealAgnosticInteractionBlock,RealAgnosticDensityInteractionBlock,RealAgnosticDensityResidualInteractionBlock}
                        name of interaction block (default:
                        RealAgnosticResidualInteractionBlock)
  --interaction_first {RealAgnosticResidualInteractionBlock,RealAgnosticInteractionBlock,RealAgnosticDensityInteractionBlock,RealAgnosticDensityResidualInteractionBlock}
                        name of interaction block (default:
                        RealAgnosticResidualInteractionBlock)
  --max_ell MAX_ELL     highest \ell of spherical harmonics (default: 3)
  --correlation CORRELATION
                        correlation order at each layer (default: 3)
  --num_interactions NUM_INTERACTIONS
                        number of interactions (default: 2)
  --MLP_irreps MLP_IRREPS
                        hidden irreps of the MLP in last readout (default:
                        16x0e)
  --radial_MLP RADIAL_MLP
                        width of the radial MLP (default: [64, 64, 64])
  --hidden_irreps HIDDEN_IRREPS
                        irreps for hidden node states (default: None)
  --num_channels NUM_CHANNELS
                        number of embedding channels (default: None)
  --max_L MAX_L         max L equivariance of the message (default: None)
  --gate {silu,tanh,abs,None}
                        non linearity for last readout (default: silu)
  --scaling {std_scaling,rms_forces_scaling,no_scaling}
                        type of scaling to the output (default:
                        rms_forces_scaling)
  --avg_num_neighbors AVG_NUM_NEIGHBORS
                        normalization factor for the message (default: 1)
  --compute_avg_num_neighbors COMPUTE_AVG_NUM_NEIGHBORS
                        normalization factor for the message (default: True)
  --compute_stress COMPUTE_STRESS
                        Select True to compute stress (default: False)
  --compute_forces COMPUTE_FORCES
                        Select True to compute forces (default: True)
  --train_file TRAIN_FILE
                        Training set file, format is .xyz or .h5 (default:
                        None)
  --valid_file VALID_FILE
                        Validation set .xyz or .h5 file (default: None)
  --valid_fraction VALID_FRACTION
                        Fraction of training set used for validation (default:
                        0.1)
  --test_file TEST_FILE
                        Test set .xyz pt .h5 file (default: None)
  --test_dir TEST_DIR   Path to directory with test files named as test_*.h5
                        (default: None)
  --multi_processed_test MULTI_PROCESSED_TEST
                        Boolean value for whether the test data was
                        multiprocessed (default: False)
  --num_workers NUM_WORKERS
                        Number of workers for data loading (default: 0)
  --pin_memory PIN_MEMORY
                        Pin memory for data loading (default: True)
  --atomic_numbers ATOMIC_NUMBERS
                        List of atomic numbers (default: None)
  --mean MEAN           Mean energy per atom of training set (default: None)
  --std STD             Standard deviation of force components in the training
                        set (default: None)
  --statistics_file STATISTICS_FILE
                        json file containing statistics of training set
                        (default: None)
  --E0s E0S             Dictionary of isolated atom energies (default: None)
  --foundation_filter_elements FOUNDATION_FILTER_ELEMENTS
                        Filter element during fine-tuning (default: True)
  --heads HEADS         Dict of heads: containing individual files and E0s
                        (default: None)
  --multiheads_finetuning MULTIHEADS_FINETUNING
                        Boolean value for whether the model is multiheaded
                        (default: True)
  --foundation_head FOUNDATION_HEAD
                        Name of the head to use for fine-tuning (default:
                        None)
  --weight_pt_head WEIGHT_PT_HEAD
                        Weight of the pretrained head in the loss function
                        (default: 1.0)
  --num_samples_pt NUM_SAMPLES_PT
                        Number of samples in the pretrained head (default:
                        10000)
  --force_mh_ft_lr FORCE_MH_FT_LR
                        Force the multiheaded fine-tuning to use arg_parser lr
                        (default: False)
  --subselect_pt {fps,random}
                        Method to subselect the configurations of the
                        pretraining set (default: random)
  --filter_type_pt {none,combinations,inclusive,exclusive}
                        Filtering method for collecting the pretraining set
                        (default: none)
  --pt_train_file PT_TRAIN_FILE
                        Training set file for the pretrained head (default:
                        None)
  --pt_valid_file PT_VALID_FILE
                        Validation set file for the pretrained head (default:
                        None)
  --foundation_model_elements FOUNDATION_MODEL_ELEMENTS
                        Keep all elements of the foundation model during fine-
                        tuning (default: False)
  --keep_isolated_atoms KEEP_ISOLATED_ATOMS
                        Keep isolated atoms in the dataset, useful for
                        transfer learning (default: False)
  --energy_key ENERGY_KEY
                        Key of reference energies in training xyz (default:
                        REF_energy)
  --forces_key FORCES_KEY
                        Key of reference forces in training xyz (default:
                        REF_forces)
  --virials_key VIRIALS_KEY
                        Key of reference virials in training xyz (default:
                        REF_virials)
  --stress_key STRESS_KEY
                        Key of reference stress in training xyz (default:
                        REF_stress)
  --dipole_key DIPOLE_KEY
                        Key of reference dipoles in training xyz (default:
                        dipole)
  --head_key HEAD_KEY   Key of head in training xyz (default: head)
  --charges_key CHARGES_KEY
                        Key of atomic charges in training xyz (default:
                        REF_charges)
  --skip_evaluate_heads SKIP_EVALUATE_HEADS
                        Comma-separated list of heads to skip during final
                        evaluation (default: pt_head)
  --loss {ef,weighted,forces_only,virials,stress,dipole,huber,universal,energy_forces_dipole,l1l2energyforces}
                        type of loss (default: weighted)
  --forces_weight FORCES_WEIGHT
                        weight of forces loss (default: 100.0)
  --swa_forces_weight SWA_FORCES_WEIGHT, --stage_two_forces_weight SWA_FORCES_WEIGHT
                        weight of forces loss after starting Stage Two
                        (previously called swa) (default: 100.0)
  --energy_weight ENERGY_WEIGHT
                        weight of energy loss (default: 1.0)
  --swa_energy_weight SWA_ENERGY_WEIGHT, --stage_two_energy_weight SWA_ENERGY_WEIGHT
                        weight of energy loss after starting Stage Two
                        (previously called swa) (default: 1000.0)
  --virials_weight VIRIALS_WEIGHT
                        weight of virials loss (default: 1.0)
  --swa_virials_weight SWA_VIRIALS_WEIGHT, --stage_two_virials_weight SWA_VIRIALS_WEIGHT
                        weight of virials loss after starting Stage Two
                        (previously called swa) (default: 10.0)
  --stress_weight STRESS_WEIGHT
                        weight of stress loss (default: 1.0)
  --swa_stress_weight SWA_STRESS_WEIGHT, --stage_two_stress_weight SWA_STRESS_WEIGHT
                        weight of stress loss after starting Stage Two
                        (previously called swa) (default: 10.0)
  --dipole_weight DIPOLE_WEIGHT
                        weight of dipoles loss (default: 1.0)
  --swa_dipole_weight SWA_DIPOLE_WEIGHT, --stage_two_dipole_weight SWA_DIPOLE_WEIGHT
                        weight of dipoles after starting Stage Two (previously
                        called swa) (default: 1.0)
  --config_type_weights CONFIG_TYPE_WEIGHTS
                        String of dictionary containing the weights for each
                        config type (default: {"Default":1.0})
  --huber_delta HUBER_DELTA
                        delta parameter for huber loss (default: 0.01)
  --optimizer {adam,adamw,schedulefree}
                        Optimizer for parameter optimization (default: adam)
  --beta BETA           Beta parameter for the optimizer (default: 0.9)
  --batch_size BATCH_SIZE
                        batch size (default: 10)
  --valid_batch_size VALID_BATCH_SIZE
                        Validation batch size (default: 10)
  --lr LR               Learning rate of optimizer (default: 0.01)
  --swa_lr SWA_LR, --stage_two_lr SWA_LR
                        Learning rate of optimizer in Stage Two (previously
                        called swa) (default: 0.001)
  --weight_decay WEIGHT_DECAY
                        weight decay (L2 penalty) (default: 5e-07)
  --amsgrad             use amsgrad variant of optimizer (default: True)
  --scheduler SCHEDULER
                        Type of scheduler (default: ReduceLROnPlateau)
  --lr_factor LR_FACTOR
                        Learning rate factor (default: 0.8)
  --scheduler_patience SCHEDULER_PATIENCE
                        Learning rate factor (default: 50)
  --lr_scheduler_gamma LR_SCHEDULER_GAMMA
                        Gamma of learning rate scheduler (default: 0.9993)
  --swa, --stage_two    use Stage Two loss weight, which decreases the
                        learning rate and increases the energy weight at the
                        end of the training to help converge them (default:
                        False)
  --start_swa START_SWA, --start_stage_two START_SWA
                        Number of epochs before changing to Stage Two loss
                        weights (default: None)
  --lbfgs               Switch to L-BFGS optimizer (default: False)
  --ema                 use Exponential Moving Average (default: False)
  --ema_decay EMA_DECAY
                        Exponential Moving Average decay (default: 0.99)
  --max_num_epochs MAX_NUM_EPOCHS
                        Maximum number of epochs (default: 2048)
  --patience PATIENCE   Maximum number of consecutive epochs of increasing
                        loss (default: 2048)
  --foundation_model FOUNDATION_MODEL
                        Path to the foundation model for transfer learning
                        (default: None)
  --foundation_model_readout
                        Use readout of foundation model for transfer learning
                        (default: True)
  --eval_interval EVAL_INTERVAL
                        evaluate model every <n> epochs (default: 1)
  --keep_checkpoints    keep all checkpoints (default: False)
  --save_all_checkpoints
                        save all checkpoints (default: False)
  --restart_latest      restart optimizer from latest checkpoint (default:
                        False)
  --save_cpu            Save a model to be loaded on cpu (default: False)
  --clip_grad CLIP_GRAD
                        Gradient Clipping Value (default: 10.0)
  --dry_run             Run all steps upto training to test settings.
                        (default: False)
  --enable_cueq ENABLE_CUEQ
                        Enable cuequivariance acceleration (default: False)
  --wandb               Use Weights and Biases for experiment tracking
                        (default: False)
  --wandb_dir WANDB_DIR
                        An absolute path to a directory where Weights and
                        Biases metadata will be stored (default: None)
  --wandb_project WANDB_PROJECT
                        Weights and Biases project name (default: )
  --wandb_entity WANDB_ENTITY
                        Weights and Biases entity name (default: )
  --wandb_name WANDB_NAME
                        Weights and Biases experiment name (default: )
  --wandb_log_hypers WANDB_LOG_HYPERS
                        The hyperparameters to log in Weights and Biases
                        (default: ['num_channels', 'max_L', 'correlation',
                        'lr', 'swa_lr', 'weight_decay', 'batch_size',
                        'max_num_epochs', 'start_swa', 'energy_weight',
                        'forces_weight'])

Args that start with '--' can also be set in a config file (specified via
--config). The config file uses YAML syntax and must represent a YAML
'mapping' (for details, see http://learn.getgrav.org/advanced/yaml). In
general, command-line values override config file values which override
defaults.
```

---

For further details, please refer to the respective folders or contact the author via the provided email.
