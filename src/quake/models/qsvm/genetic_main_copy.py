import numpy as np

def generate_random_array_with_exclusion(min_val, max_val, size, exclude_array=None):
    if exclude_array is not None:
        valid_values_per_index = [np.setdiff1d(np.arange(min_val, max_val), [exclude_array[i]]) for i in range(size)]
        random_indices = [np.random.choice(valid_values_per_index[i]) for i in range(size)]
    else:
        random_indices = np.random.randint(min_val, max_val, size)
    return random_indices

# Example usage
min_val = 1
max_val = 10
size = 5
exclude_array = [3, 5, 7, 2, 9]

import pdb; pdb.set_trace()
random_numbers = generate_random_array_with_exclusion(min_val, max_val, size, exclude_array)
print(random_numbers)
