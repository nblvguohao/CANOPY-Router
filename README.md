# CANOPY-Router

Context-conditioned evidence routing for cancer driver gene prediction.

CANOPY-Router learns per-gene routing weights over six heterogeneous evidence channels, producing interpretable source attribution alongside state-of-the-art performance on the TREE benchmark.

## Quick Start

```bash
git clone https://github.com/nblvguohao/CANOPY-Router.git
cd CANOPY-Router

# Install dependencies
conda create -n canopy python=3.10 && conda activate canopy
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.7.0
pip install -r requirements.txt

# Run on BRCA data (download TREE dataset first, see Full Data Preparation below)
python src/run.py \
    --tree_h5    data/TREE/dataset/networks/BRCA_multiomics.h5 \
    --output_dir runs/quickstart \
    --run_name   BRCA_quickstart \
    --device     0 \
    --seed       1234 \
    --epochs     100 \
    --hidden_dim 64 \
    --enable_gske \
    --gske_npz       data/msigdb/c2_GenesetsMatrix.npz \
    --gske_c5_npz    data/msigdb/c5_GenesetsMatrix.npz \
    --gske_gene_list data/msigdb/geneList.csv \
    --enable_pgem \
    --alphamissense_path inputs/alphamissense_scores.pt \
    --enable_hyperconv --hyperconv_dim 32 \
    --enable_adaptive_patho
```

---

## Installation

**Recommended: Python 3.10, NumPy < 2.0**

```bash
# conda (recommended)
conda create -n canopy python=3.10
conda activate canopy
pip install torch==2.2.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.7.0
pip install -r requirements.txt
```

> **Note**: NumPy 2.x is incompatible with some dependencies. Use a fresh environment if you have NumPy 2.x installed.

---

## Repository Contents

```
CANOPY-Router/
├── src/
│   ├── model.py               # CANOPY-Router model (6-channel router + transformer)
│   ├── run.py                 # Training and evaluation pipeline
│   ├── utils.py               # Data loading and metrics
│   ├── supcl_module.py        # Supervised contrastive learning module
│   ├── graft_adapter.py       # GRAFT adapter for homogeneous networks
│   └── model_with_skipgate.py # Variant with skip-gate ablation
├── scripts/
│   ├── run_all_cancer_specific.py             # Cancer-specific batch runner
│   ├── run_all_pancancer.py                   # Pan-cancer batch runner
│   ├── run_disfusion_tree_splits.py           # DISFusion comparison with TREE splits
│   ├── external_validation_intogen.py         # IntOGen external validation
│   ├── stat_significance_and_druggability.py  # Statistical tests & druggability
│   └── analyze_routing_interpretability.py    # Routing weight analysis
├── data/
│   └── msigdb/
│       ├── c2_GenesetsMatrix.npz
│       ├── c5_GenesetsMatrix.npz
│       └── geneList.csv
├── inputs/
│   └── alphamissense_scores.pt  # AlphaMissense pathogenicity proxy
├── requirements.txt
└── LICENSE
```

---

## Full Data Preparation

For full benchmark experiments, download the following:

| Resource | Source | Path |
|----------|--------|------|
| TREE benchmark | [Schulte-Sasse et al.](https://github.com/schulter/TREE) | `data/TREE/dataset/networks/*.h5` |

The MSigDB gene sets and AlphaMissense scores are already included in this repository.

Full data directory layout:
```
data/TREE/dataset/networks/
    ├── BRCA_multiomics.h5
    ├── COAD_multiomics.h5
    ├── LIHC_multiomics.h5
    ├── LUAD_multiomics.h5
    ├── STAD_multiomics.h5
    └── UCEC_multiomics.h5
```

---

## Full Experiments

### Cancer-Specific (Heterogeneous Networks)

```bash
python scripts/run_all_cancer_specific.py \
    --tree_data_root data/TREE/dataset/networks \
    --out_root runs/cancer_specific \
    --devices 0 \
    --seeds 1234,2345,3456 \
    --epochs 100 --patience 10 \
    --enable_supcl --enable_ctam \
    --enable_gske \
    --gske_npz data/msigdb/c2_GenesetsMatrix.npz \
    --gske_c5_npz data/msigdb/c5_GenesetsMatrix.npz \
    --gske_gene_list data/msigdb/geneList.csv \
    --enable_mvga --enable_pgem \
    --alphamissense_path inputs/alphamissense_scores.pt \
    --enable_hyperconv --hyperconv_dim 32 \
    --enable_adaptive_patho --save_gate_stats
```

### Pan-Cancer (Homogeneous Networks)

```bash
python scripts/run_all_pancancer.py \
    --tree_data_root data/TREE/dataset/networks \
    --out_root runs/pan_cancer \
    --devices 0 \
    --seeds 1234,2345,3456 \
    --epochs 100 --patience 10 \
    --enable_supcl --enable_ctam \
    --enable_gske \
    --gske_npz data/msigdb/c2_GenesetsMatrix.npz \
    --gske_c5_npz data/msigdb/c5_GenesetsMatrix.npz \
    --gske_gene_list data/msigdb/geneList.csv \
    --enable_mvga --enable_pgem \
    --alphamissense_path inputs/alphamissense_scores.pt \
    --enable_hyperconv --hyperconv_dim 32 \
    --enable_adaptive_patho --save_gate_stats
```

---

## Model Architecture

Six evidence channels with dynamic gated routing:

| Channel | Source | Description |
|---------|--------|-------------|
| Ch.1 | PPI graph | GAT encoder over protein interaction network |
| Ch.2 | Multi-omics | Linear projection of mutation, methylation, expression |
| Ch.3 | Feature intensity | Feature-intensity-gated encoder (L2-norm gating) |
| Ch.4 | Cancer type | Cancer-type context embedding |
| Ch.5 | MSigDB | Gene-set membership embedding |
| Ch.6 | Pathway graph | Pre-computed pathway hypergraph convolution |

> **Note**: Source code uses `pathogenicity` as variable names for backward compatibility with trained checkpoints. In the paper, Ch.3 is referred to as "Feature-Intensity-Gate".

---

## Results (TREE Benchmark, Heterogeneous)

| Method | Macro AUPR | Macro AUC | Macro F1 |
|--------|-----------|-----------|----------|
| **CANOPY-Router** | **0.863** | **0.929** | **0.810** |
| DISHyper | 0.825 | 0.886 | 0.748 |
| MNGCL | 0.755 | 0.862 | 0.693 |
| CGMega | 0.742 | 0.851 | 0.678 |

---

## Citation

```bibtex
@article{lv2026canopy_router,
  title={CANOPY-Router: Context-Conditioned Evidence Routing for Interpretable Cancer Driver Gene Prioritization},
  author={Lv, Guohao and Xia, Yingchun and Li, Xiaowei and Liu, Huichao and Zhu, Xiaolei and Yang, Shuai and Wang, Qingyong and Gu, Lichuan},
  journal={CAAI Artificial Intelligence Research},
  year={2026},
  note={Under review}
}
```

## License

MIT License. See [LICENSE](LICENSE).
