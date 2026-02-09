import numpy as np
import pandas as pd

# Load sequences
print("Loading sequences...")
sequences = np.load('training_data_sequences.npy', allow_pickle=True)
print(f"Sequences type: {type(sequences)}")
print(f"Sequences shape: {sequences.shape if hasattr(sequences, 'shape') else 'N/A'}")
print(f"Number of sequences: {len(sequences)}")

# Check first few sequences
print("\nChecking first sequence...")
first_seq = sequences[0]
if isinstance(first_seq, list):
    first_seq = np.array(first_seq, dtype=float)
print(f"First sequence shape: {first_seq.shape}")
print(f"First sequence dtype: {first_seq.dtype}")
print(f"First sequence min: {np.min(first_seq)}")
print(f"First sequence max: {np.max(first_seq)}")
print(f"First sequence has NaN: {np.isnan(first_seq).any()}")
print(f"First sequence has Inf: {np.isinf(first_seq).any()}")

# Check all sequences for NaN/Inf
print("\nChecking all sequences...")
has_nan = False
has_inf = False
for i, seq in enumerate(sequences[:100]):  # Check first 100
    if isinstance(seq, list):
        seq = np.array(seq, dtype=float)
    if np.isnan(seq).any():
        print(f"Sequence {i} has NaN")
        has_nan = True
        break
    if np.isinf(seq).any():
        print(f"Sequence {i} has Inf")
        has_inf = True
        break

if not has_nan:
    print("No NaN found in first 100 sequences")
if not has_inf:
    print("No Inf found in first 100 sequences")

# Load labels
print("\nLoading labels...")
df = pd.read_csv('training_data.csv')
labels = df['profitable_5d'].values
print(f"Labels shape: {labels.shape}")
print(f"Labels dtype: {labels.dtype}")
print(f"Labels unique values: {np.unique(labels)}")
print(f"Labels min: {labels.min()}")
print(f"Labels max: {labels.max()}")
print(f"Labels has NaN: {np.isnan(labels).any()}")
print(f"Number of 1s: {(labels == 1).sum()}")
print(f"Number of 0s: {(labels == 0).sum()}")
print(f"Number of other values: {((labels != 0) & (labels != 1)).sum()}")
