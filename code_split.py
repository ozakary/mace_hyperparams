import random
import re

# Set random seed for reproducibility
random.seed(42)

# Open the original file
with open('mlip_data_xe-water_0-5ps_sampled-20.xyz', 'r') as f:
    lines = f.readlines()

# Find the indices of the structure count lines (lines with just a number)
# These lines appear before each configuration
count_line_indices = []
for i, line in enumerate(lines):
    if i > 0 and line.strip().isdigit():
        count_line_indices.append(i)
    # Also include the first line (first atom count)
    elif i == 0 and line.strip().isdigit():
        count_line_indices.append(i)

# Calculate the total number of configurations
total_configs = len(count_line_indices)

# Calculate the number of configurations for the test set (10%)
test_size = int(total_configs * 0.1)
# train_val_size will be the remainder (90%)

# Create a list of all configuration indices
all_indices = list(range(total_configs))

# Shuffle the indices to ensure random sampling
random.shuffle(all_indices)

# Split the indices into training+validation (90%) and testing (10%) sets
train_val_indices = sorted(all_indices[:-test_size])  # 90% for train+val
test_indices = sorted(all_indices[-test_size:])       # 10% for test

# Function to extract configuration blocks and make the specified replacements
def extract_blocks(indices):
    blocks = []
    for idx in indices:
        start_idx = count_line_indices[idx]
        # If this is the last configuration, extract to end of file
        if idx == len(count_line_indices) - 1:
            end_idx = len(lines)
        else:
            end_idx = count_line_indices[idx + 1]
            
        # Get the block lines
        block = lines[start_idx:end_idx]
        
        # Modify the header line (which is the line after the atom count)
        if len(block) >= 2:
            # The second line in each block is the header line
            header_line = block[1]
            
            # Make all the required replacements
            modified_header = header_line
            
            # Remove free_energy completely
            modified_header = re.sub(r'\s+free_energy=-?\d+\.\d+', '', modified_header)
            
            # Replace energy with REF_energy
            modified_header = re.sub(r'energy=', 'energy=', modified_header)
            
            # Replace forces with REF_forces in Properties field
            modified_header = re.sub(r'forces:', 'forces:', modified_header)
            
            # Replace stress with REF_stress
            modified_header = re.sub(r'stress=', 'stress=', modified_header)
            
            block[1] = modified_header
        
        blocks.extend(block)
    return blocks

# Extract blocks for each set
train_val_blocks = extract_blocks(train_val_indices)
test_blocks = extract_blocks(test_indices)

# Write the parts to separate files
with open('water_xe_dataset_train_val.xyz', 'w') as f:
    f.writelines(train_val_blocks)

with open('water_xe_dataset_test.xyz', 'w') as f:
    f.writelines(test_blocks)

# Print statistics
print(f"Total configurations: {total_configs}")
print(f"Training+Validation set: {len(train_val_indices)} configurations ({len(train_val_indices)/total_configs:.1%})")
print(f"Testing set: {len(test_indices)} configurations ({len(test_indices)/total_configs:.1%})")
