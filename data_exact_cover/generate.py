import numpy as np

def generate_exact_cover_dataset(n=100, d=30, k=5, noise_density=0.2):
    # Initialize empty dataset
    A = np.zeros((n, d), dtype=np.int32)
    
    # 1. Generate the Ground Truth (Exact Cover)
    # Randomly partition the n rows into k mutually exclusive sets
    row_indices = np.arange(n)
    column_assignments = np.random.randint(0, k, size=n)
    A[row_indices, column_assignments] = 1
    
    # 2. Generate the Decoys (The Overlap Trap)
    # Fill the remaining columns with random 1s
    noise_matrix = np.random.binomial(1, noise_density, size=(n, d - k))
    A[:, k:] = noise_matrix
    
    # 3. Shuffle the columns
    np.random.shuffle(A.T)
    
    # The target vector is always exactly 1 for every row
    y = np.ones(n, dtype=np.int32)
    
    return A, y

# Generate your sandbox dataset
A, y = generate_exact_cover_dataset(n=100, d=90, k=5)

import pandas as pd 
for i in range(A.shape[1]):
    pd.Series(A[:,i]).to_csv(f"x{i}.csv",index=False,header=False)
pd.Series(y).to_csv("y.csv",index=False,header=False)