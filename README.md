
# DynPerturb: A Dynamic Perturbation Model for Single-Cell Gene Expression Data Analysis

## Project Overview
DynPerturb is an advanced deep learning model designed to infer gene regulatory networks (GRNs) and analyze the effects of perturbations on cellular states using single-cell RNA-seq data. By incorporating both temporal and spatial information, DynPerturb enhances the understanding of gene interactions during cellular development, disease progression, and response to perturbations, making it an invaluable tool for biologists and researchers in drug discovery, genetic studies, and disease modeling.

## Data
### GRN Data
The data for Gene Regulatory Network (GRN) inference comes from the following dataset:
1. **Adult Human Kidney Single-Cell RNA-seq** (Version 1.5)
   - **Source**: [CellxGene Single-Cell Data](https://cellxgene.cziscience.com/e/dea717d4-7bc0-4e46-950f-fd7e1cc8df7d.cxg/)
   - **Data Location**: `/home/share/huadjyin/home/s_qinhua2/02code/tgn-master/kidney/data`
   - Data files: `ml_aPT-A.csv`, `ml_aPT-A.npy`, `ml_aPT-A_node.pkl`, `ml_aPT-B...`, etc.
   - This dataset includes single-cell gene expression profiles from different cell types of the human kidney.

### Cell Development Data
1. **Human Bone Marrow Hematopoietic Development** (Balanced Reference Map)
   - **Source**: [CellxGene Bone Marrow Data](https://cellxgene.cziscience.com/e/cd2f23c1-aef1-48ae-8eb4-0bcf124e567d.cxg/)
   - **Data Location**: `/home/share/huadjyin/home/s_qinhua2/02code/tgn-master/HumanBone/data`
   - Data files: `ml_HumanBone.csv`, `ml_HumanBone.npy`, `ml_HumanBone_node.pkl`
   - The dataset helps explore the differentiation process of blood cells from human bone marrow.

### Spatial Data
1. **Murine Cardiac Development Spatiotemporal Transcriptome Sequencing**
   - **Source**: [Gigascience Article](https://doi.org/10.1093/gigascience/giaf012)
   - **Data Location**: `/home/share/huadjyin/home/s_qinhua2/02code/tgn-master/spatiotemporal_mouse/data`
   - Data files: `ml_mouse_spatial.csv`, `ml_mouse_spatial.npy`, `ml_mouse_spatial_node.pkl`
   - Provides a detailed spatial transcriptomic map of murine heart development, useful for understanding heart tissue differentiation and development.

## Environment Setup
### Dependencies
The environment required for running the model can be easily set up using the provided `environment.yaml` file.
1. Download the `environment.yaml` file from this repository.
2. Modify the path in the last line of the file to match your system's conda path.
3. To install dependencies, run:
   ```bash
   conda env create -f environment.yaml
   ```
4. Activate the conda environment:
   ```bash
   conda activate DynPerturb
   ```

## Key Features
### Gene Regulatory Network (GRN) Inference
- The model uses advanced machine learning techniques to infer the relationships between genes in the form of a network. 
- It supports time-series and spatial-temporal data, allowing researchers to uncover gene interactions during development and disease progression.

### Perturbation Response Analysis
- DynPerturb can analyze the effect of different perturbations (e.g., drugs, genetic mutations) on gene expression. 
- It helps predict cellular responses to various conditions, a critical feature for drug discovery and genetic studies.

### Data Integration
- The model can handle data from multiple sources, including time-course, spatial, and developmental datasets, providing a holistic view of gene interactions.

### Embedding Generation and Saving
- DynPerturb computes node embeddings for every gene, which can then be saved and used for downstream analyses.
- This feature enables a more in-depth exploration of gene functions and interactions.

### Visualization and Interpretation
- The model supports visualization of gene networks and perturbation effects, helping researchers understand complex biological interactions.

## TASK1: Gene Regulatory Network (GRN) Reconstruction and Link Prediction

### STEP1: Input Data Preparation

#### Python script: `train_ChangeNodeFeat_SaveEmbeddings0529_link.py`

Example command line:

```bash
python train_ChangeNodeFeat_SaveEmbeddings0529_link.py --data HumanBone --bs 64 --n_epoch 100 --n_layer 1 --lr 1e-4
```

**Parameters**:
- `--data`: Dataset name, e.g., "HumanBone".
- `--bs`: Batch size for training.
- `--n_epoch`: Number of epochs.
- `--n_layer`: Number of network layers.
- `--lr`: Learning rate.

**Task**:
- This step involves loading the data using the `get_data_link2` function, which loads node features (gene expressions) and edge features (relationships between genes) for link prediction tasks. 
- Data preprocessing includes selecting and filtering specific gene pairs, normalizing gene expressions, and splitting data into training, validation, and test sets.
- The feature matrix is then prepared to be passed to the model for training.

### STEP2: Model Training and Evaluation

#### Python script: `train_main_kidney.py`

Example command line:

```bash
python train_main_kidney.py --data HumanBone --bs 64 --n_epoch 100 --n_layer 1 --lr 3e-4
```

**Parameters**:
- `--data`: Dataset name, e.g., "HumanBone".
- `--bs`: Batch size for training.
- `--n_epoch`: Number of epochs.
- `--n_layer`: Number of network layers.
- `--lr`: Learning rate.

**Task**:
- Initializes the model using the `NetModel` class, which defines a deep learning model for link prediction and node classification tasks.
- The model is trained using both the node features and edge features. The loss is calculated for link prediction and node classification tasks, and backpropagation is performed to optimize the model parameters.
- Model performance is evaluated using metrics like AUC (Area Under the Curve), precision, and recall.

### STEP3: Embedding Computation and Saving

**Task**:
- During training, the model computes node and edge embeddings, which are lower-dimensional representations of genes and gene interactions.
- The embeddings are saved to a specified directory for further analysis or downstream tasks, such as gene network visualization or perturbation analysis.

## TASK2: Node Classification

### STEP1: Model Setup and Hyperparameter Initialization

#### Python script: `train_ChangeNodeFeat_SaveEmbeddings0521.py`

Example command line:

```bash
python train_ChangeNodeFeat_SaveEmbeddings0521.py --data HumanBone --bs 64 --n_epoch 100 --n_layer 1 --lr 1e-4
```

**Parameters**:
- `--data`: The dataset name, for example, "HumanBone".
- `--bs`: Batch size used during training.
- `--n_epoch`: Number of epochs to train the model.
- `--n_layer`: Number of layers in the neural network.
- `--lr`: Learning rate for optimization.

**Task**:
- The first step in node classification is setting up the model and initializing its hyperparameters.
- The model uses random edge samplers to create negative samples, which helps train the model to distinguish real gene interactions from random pairs.
- The setup ensures that the model is properly initialized and ready for training on the classification task.

### STEP2: Model Training and Loss Computation

**Task**:
- The model is trained to classify genes (nodes) based on their interactions. The training involves optimizing the model’s parameters to minimize a loss function, which is a combination of binary cross-entropy loss for link prediction and cross-entropy loss for node classification.
- Early stopping is used to prevent overfitting and ensure that the model generalizes well to unseen data.

### STEP3: Model Evaluation

#### Python script: `ddpversion.py`

Example command line:

```bash
python ddpversion.py --data HumanBone --gpu 0 --n_epoch 100 --bs 64 --n_layer 2
```

**Parameters**:
- `--data`: Dataset name to evaluate.
- `--gpu`: The GPU index to use.
- `--n_epoch`: Number of epochs for evaluation.
- `--bs`: Batch size for evaluation.
- `--n_layer`: Number of layers in the model.

**Task**:
- After training, the model's performance is evaluated using the test set. Evaluation metrics such as AUC, precision, recall, and F1-score are computed for link prediction and node classification tasks.
- Confusion matrices are generated to show how well the model distinguishes between different classes of genes (nodes).

## TASK3: Temporal Modeling and Node Memory Augmentation

### STEP1: Memory Initialization

**Task**:
- If enabled, memory augmentation helps the model remember previous states across time steps. This memory mechanism improves the model’s ability to learn from temporal data, such as time-course RNA-seq data.
- This step initializes the memory network, which stores important gene interactions over time.

### STEP2: Temporal Encoding and Computation

**Task**:
- The model computes temporal embeddings for each gene, which helps capture time-dependent relationships between genes.
- Temporal embeddings are used to represent how gene interactions evolve over time, which is particularly useful for time-series data.

### STEP3: Final Model Evaluation and Embedding Saving

**Task**:
- After training, the model’s final performance is evaluated and the learned node and edge embeddings are saved.
- These embeddings are used to study gene interactions and can be further analyzed for tasks like gene prioritization, regulatory network analysis, or perturbation prediction.

## License


## Contact
For questions or support, feel free to reach out:
- **Your Name**: [your_email@example.com]
