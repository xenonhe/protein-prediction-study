# Data

## Source
Kaggle: https://www.kaggle.com/datasets/tamzidhasan/protein-secondary-structure-casp12-cb513-ts115

## Download
Set up your Kaggle API credentials first:
```bash
export KAGGLE_USERNAME=your_kaggle_username
export KAGGLE_KEY=your_kaggle_api_key
```

Then download:
```bash
kaggle datasets download -d tamzidhasan/protein-secondary-structure-casp12-cb513-ts115 -p data/raw --unzip
```

## Files (not tracked by git)
| File                                     | Split | Proteins |
|------------------------------------------|-------|----------|
| training_secondary_structure_train.csv   | Train | 8,678    |
| validation_secondary_structure_valid.csv | Valid | 2,170    |
| test_secondary_structure_cb513.csv       | Test  | 513      |
| test_secondary_structure_ts115.csv       | Test  | 115      |
| test_secondary_structure_casp12.csv      | Test  | 21       |

## Columns
- `seq`  : amino acid sequence (single-letter codes)
- `sst3` : 3-state secondary structure label per residue (H, E, C)
- `sst8` : 8-state secondary structure label per residue (H, G, I, E, B, T, S, C)

## Label Mapping
### Q3
H = Helix | E = Strand | C = Coil

### Q8
G = 3-helix | H = α-helix | I = π-helix
E = β-strand | B = β-bridge
T = Turn | S = Bend | C = Coil
