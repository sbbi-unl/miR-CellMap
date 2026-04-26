# MicroRNA-gene Pipeline

This is the repository for MicroRNA-gene Pipeline **A Graph-Transformer Pipeline for Single-Cell microRNA–Gene Regulatory Analysis**.

If you have any questions or feedback, please contact **Jane Ohia** or **Juan Cui**.

---

## Dev environment

python: 3.9  
pytorch: 2.x  
torch-geometric: 2.x  
System: Linux / HPC environment recommended  
GPU: recommended for HGT training, but downstream inference and evaluation can run on CPU

---

## Preparations

### Main data used in this study

We used matched single-cell miRNA and mRNA data from the **PSCSR-seq / PSCSR-seq V2** framework and demonstrated the pipeline on the **A549 case study from GSE226714**.

Main data files used by the pipeline include:


- `ENCORI_hg38_miRNA_mRNA_all.txt`
- `ENCORI_hg38_CLIP-seq_miRNA-target_all_PDCD4.xls`
- `miRTarBase_MTI.csv`
- `GSM7082512_cells_chip1_miRNA.txt`	
- `GSM7082513_cells_chip1_mRNA.txt`

Place processed inputs under:

```bash
phase3_aligned_data/

Manual installation
Conda environment
conda create -n mirna_graph_env python=3.9 -y
conda activate mirna_graph_env
Install core scientific libraries
conda install -y numpy pandas scipy matplotlib seaborn scikit-learn jupyter
pip install scienceplots
Install Bayesian network package
pip install pgmpy
Install plotting / figure support
pip install matplotlib-venn
Install PyTorch

CPU version:

pip install torch torchvision torchaudio

GPU version example:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Use the PyTorch build that matches your CUDA version.

Install PyTorch Geometric
pip install torch-geometric

If needed, install the PyG dependency wheels matching your torch/CUDA version from the PyG wheel page.

Suggested package list

The scripts in this repository use the following Python libraries:

os
warnings
itertools
numpy
pandas
scipy
matplotlib
seaborn
scienceplots
scikit-learn
pgmpy
matplotlib-venn
torch
torch-geometric
Example installation block
conda create -n mirna_graph_env python=3.9 -y
conda activate mirna_graph_env

conda install -y numpy pandas scipy matplotlib seaborn scikit-learn
pip install scienceplots
pip install pgmpy
pip install matplotlib-venn

pip install torch torchvision torchaudio
pip install torch-geometric

Runing order

Run the scripts in this order:
python geo_alignment.py
python sparse_triplet_conversion.py
python filter_mirna_prior.py
python htg_construction.py
python hgt_autoencoder_training.py
python cell_type_aware.py
python mirna_gene_embedding_fig.py
python network_inference.py
python advance_modeling.py
python functional_validation.py
python benchmark_tables.py


Troubleshooting
1. ModuleNotFoundError: No module named pgmpy

Install:

pip install pgmpy
2. ModuleNotFoundError: No module named scienceplots

Install:

pip install scienceplots
3. ModuleNotFoundError: No module named matplotlib_venn

Install:

pip install matplotlib-venn
4. PyTorch Geometric installation issues

If torch-geometric fails to install:

check your Python version
check your torch version
check your CUDA version
install the matching wheels for PyG dependencies from the PyTorch Geometric wheel repository

Typical dependent wheels may include:

torch_scatter
torch_sparse
torch_cluster
torch_spline_conv

Then reinstall torch-geometric.

5. File not found errors

Make sure all required processed input files are present in:

phase3_aligned_data/

and that the ENCORI / PDCD4 validation files are in the expected working directory.

Environment export

To save your working conda environment:

conda env export > environment.yml

To recreate it later:

conda env create -f environment.yml
Citation

If you use this repository, please cite the associated manuscript:
Ohia J, Cui J. A Graph-Transformer Pipeline for Single-Cell microRNA–Gene Regulatory Analysis.
