"""
Embeddings package for protein sequence encoding.
Provides one-hot and ESM-2 based feature extraction.
"""

# embeddings/__init__.py
from .onehot import prepare_data_onehot
from .esm import prepare_data_esm_chunked
from .prott5 import prepare_data_prott5_chunked, get_prott5_model
from .constants import SS3_MAP

__all__ = [
    'prepare_data_onehot',
    'prepare_data_esm_chunked',
    'prepare_data_prott5_chunked',
    'get_prott5_model',
    'SS3_MAP'
]