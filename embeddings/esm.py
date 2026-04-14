import numpy as np
import torch
from .constants import SS3_MAP

def prepare_data_esm(sequences, structures, esm_model, batch_converter,
                     device, window_size=15):
    """
    Extract ESM embeddings for a list of proteins (single batch).
    Only use if total number of proteins is small (e.g., < 50).
    """
    pad_len = window_size // 2
    protein_list = [(f'prot_{i}', seq) for i, seq in enumerate(sequences)]
    batch_labels, batch_strs, batch_tokens = batch_converter(protein_list)
    num_layers = esm_model.num_layers
    last_layer = num_layers
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[last_layer])
        if 33 in results["representations"]:
            embeddings = results["representations"][last_layer].cpu().numpy()
        else:
            last_key = max(results["representations"].keys())
            print(f"Layer {last_layer} not found, using layer {last_key} instead.")
            embeddings = results["representations"][last_key].cpu().numpy()
    
    X_list, y_list = [], []
    for idx, (seq, ss) in enumerate(zip(sequences, structures)):
        prot_emb = embeddings[idx][:len(seq)]          # trim to actual length
        pad_vec = np.zeros((1, 1280))
        padded_emb = np.vstack([pad_vec]*pad_len + [prot_emb] + [pad_vec]*pad_len)
        padded_ss = 'C'*pad_len + ss + 'C'*pad_len
        for i in range(len(seq)):
            window_emb = padded_emb[i:i+window_size]   # (window_size, 1280)
            X_list.append(window_emb.flatten())
            y_list.append(SS3_MAP[padded_ss[i+pad_len]])
    return np.array(X_list), np.array(y_list)


def prepare_data_esm_chunked(sequences, structures, esm_model, batch_converter,
                             device, window_size=15, chunk_size=100):
    """
    Extract ESM embeddings in chunks to avoid memory overflow.
    Recommended for large datasets.
    """
    pad_len = window_size // 2
    all_X, all_y = [], []
    n_proteins = len(sequences)
    
    for start in range(0, n_proteins, chunk_size):
        end = min(start + chunk_size, n_proteins)
        print(f"Processing proteins {start} to {end-1}...")
        
        chunk_seqs = sequences[start:end]
        chunk_ss = structures[start:end]
        protein_list = [(f'prot_{i}', seq) for i, seq in enumerate(chunk_seqs)]
        
        batch_labels, batch_strs, batch_tokens = batch_converter(protein_list)
        num_layers = esm_model.num_layers
        last_layer = num_layers
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[last_layer])
            if 33 in results["representations"]:
                embeddings = results["representations"][last_layer].cpu().numpy()
            else:
                last_key = max(results["representations"].keys())
                print(f"Layer {last_layer} not found, using layer {last_key} instead.")
                embeddings = results["representations"][last_key].cpu().numpy()
        
        for idx, (seq, ss) in enumerate(zip(chunk_seqs, chunk_ss)):
            prot_emb = embeddings[idx][:len(seq)]
            pad_vec = np.zeros((1, 1280))
            padded_emb = np.vstack([pad_vec]*pad_len + [prot_emb] + [pad_vec]*pad_len)
            padded_ss = 'C'*pad_len + ss + 'C'*pad_len
            for i in range(len(seq)):
                window_emb = padded_emb[i:i+window_size]
                all_X.append(window_emb.flatten())
                all_y.append(SS3_MAP[padded_ss[i+pad_len]])
    
    return np.array(all_X), np.array(all_y)