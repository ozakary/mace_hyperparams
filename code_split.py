import random

# Set random seed for reproducibility
random.seed(42)

# Open the original file
with open('mlip_data_xe-water_0-5ps_sampled-20.xyz', 'r') as f:
    lines = f.readlines()

# Find the indices of the lines starting with 'Lattice'
lattice_indices = [i for i, line in enumerate(lines) if line.startswith('Lattice')]

# Calculate the total number of configurations
total_configs = len(lattice_indices)

# Calculate the number of configurations for each split
train_size = int(total_configs * 0.8)
valid_size = int(total_configs * 0.1)
# test_size will be the remainder

# Create a list of all configuration indices
all_indices = list(range(total_configs))

# Shuffle the indices to ensure random sampling
random.shuffle(all_indices)

# Split the indices into training, validation, and testing sets
train_indices = sorted(all_indices[:train_size])
valid_indices = sorted(all_indices[train_size:train_size + valid_size])
test_indices = sorted(all_indices[train_size + valid_size:])

# Function to extract configuration blocks
def extract_blocks(indices):
    blocks = []
    for idx in indices:
        start_idx = 0 if idx == 0 else lattice_indices[idx - 1]
        end_idx = lattice_indices[idx] - 1 if idx < len(lattice_indices) - 1 else len(lines)
        blocks.extend(lines[start_idx:end_idx])
    return blocks

# Extract blocks for each set
train_blocks = extract_blocks(train_indices)
valid_blocks = extract_blocks(valid_indices)
test_blocks = extract_blocks(test_indices)

# Write the parts to separate files
with open('mlip_data_xe-water_train.xyz', 'w') as f:
    f.writelines(train_blocks)

with open('mlip_data_xe-water_valid.xyz', 'w') as f:
    f.writelines(valid_blocks)

with open('mlip_data_xe-water_test.xyz', 'w') as f:
    f.writelines(test_blocks)

# Print statistics
print(f"Total configurations: {total_configs}")
print(f"Training set: {len(train_indices)} configurations ({len(train_indices)/total_configs:.1%})")
print(f"Validation set: {len(valid_indices)} configurations ({len(valid_indices)/total_configs:.1%})")
print(f"Testing set: {len(test_indices)} configurations ({len(test_indices)/total_configs:.1%})")
