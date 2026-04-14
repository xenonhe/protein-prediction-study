import numpy as np
from .constants import SS3_MAP

def one_hot_encode_window(window_seq, aa_to_idx):
    """Encode a single window as flattened one‑hot vector."""
    window_size = len(window_seq)
    one_hot = np.zeros((window_size, 20))
    for j, aa in enumerate(window_seq):
        if aa in aa_to_idx:
            one_hot[j, aa_to_idx[aa]] = 1
    return one_hot.flatten()

def prepare_data_onehot(sequences, structures, window_size=15):
    """
    Convert a list of sequences and SS3 strings into one‑hot features and labels.
    Returns (X, y) where X.shape = (n_windows, window_size*20).
    """
    pad_len = window_size // 2
    aa_order = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}
    
    X_list, y_list = [], []
    for seq, ss in zip(sequences, structures):
        padded_seq = 'X' * pad_len + seq + 'X' * pad_len
        padded_ss  = 'C' * pad_len + ss + 'C' * pad_len
        for i in range(len(seq)):
            window_seq = padded_seq[i:i+window_size]
            X_list.append(one_hot_encode_window(window_seq, aa_to_idx))
            y_list.append(SS3_MAP[padded_ss[i+pad_len]])
    return np.array(X_list), np.array(y_list)