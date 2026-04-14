# embeddings/prott5.py
import re
import torch
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm
from .constants import SS3_MAP

def get_prott5_model(device='cuda'):
    """Load the ProtT5-XL-U50 encoder model and its tokenizer."""
    model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name).to(device)
    model = model.eval()
    if device.type == 'cuda':
        model = model.half()
    return model, tokenizer

def prepare_sequence_for_prott5(seq):
    """
    Convert a raw amino acid sequence into the format expected by ProtT5:
    - Replace rare/ambiguous amino acids (U, Z, O, B) with 'X'
    - Insert a space between every character
    """
    # Replace non-standard amino acids with 'X'
    seq_clean = re.sub(r"[UZOB]", "X", seq)
    # Insert spaces
    return " ".join(seq_clean)

def prepare_data_prott5_chunked(sequences, structures, model, tokenizer,
                                device, window_size=15, chunk_size=50):
    """
    Extract ProtT5 embeddings for many proteins in chunks.
    Returns (X, y) where X.shape = (n_windows, window_size, 1024).
    """
    pad_len = window_size // 2
    all_X, all_y = [], []
    n_proteins = len(sequences)
    embedding_dim = 1024  # ProtT5 output dimension

    for start in tqdm(range(0, n_proteins, chunk_size), desc="ProtT5 chunks"):
        end = min(start + chunk_size, n_proteins)
        chunk_seqs = sequences[start:end]
        chunk_ss = structures[start:end]

        # Preprocess sequences for ProtT5 (spaces + rare aa replacement)
        preprocessed_seqs = [prepare_sequence_for_prott5(seq) for seq in chunk_seqs]

        # Tokenize
        ids = tokenizer.batch_encode_plus(
            preprocessed_seqs,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt"
        )
        input_ids = ids['input_ids'].to(device)
        attention_mask = ids['attention_mask'].to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
            # last_hidden_state: (batch, seq_len, 1024)
            per_residue_embs = embedding_repr.last_hidden_state.detach().cpu().numpy()

        # Process each protein in the chunk
        for idx, (seq, ss) in enumerate(zip(chunk_seqs, chunk_ss)):
            seq_len = len(seq)
            # Trim special tokens (first and last token are <cls> and <sep>)
            # ProtT5 adds a start and end token; we need only the residues
            prot_emb = per_residue_embs[idx][1:seq_len+1]  # shape (L, 1024)

            # Pad with zero vectors
            pad_vec = np.zeros((1, embedding_dim))
            padded_emb = np.vstack([pad_vec] * pad_len + [prot_emb] + [pad_vec] * pad_len)
            padded_ss = 'C' * pad_len + ss + 'C' * pad_len

            # Slide window
            for i in range(len(seq)):
                window_emb = padded_emb[i:i + window_size]  # (window_size, 1024)
                all_X.append(window_emb)
                all_y.append(SS3_MAP[padded_ss[i + pad_len]])

    return np.array(all_X), np.array(all_y)