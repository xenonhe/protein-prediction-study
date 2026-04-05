"""Shared data loading utilities."""
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

def load_sequences(filename: str):
    """Load sequences from a FASTA file in data/raw/."""
    path = DATA_DIR / "raw" / filename
    # TODO: implement with BioPython
    raise NotImplementedError
