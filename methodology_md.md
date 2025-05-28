# Methodology

## Experimental Design

### Research Questions
1. How do different fine-tuning approaches compare for document-level IE tasks?
2. Which model architecture (BERT vs GPT-Neo) performs better for NER and RE?
3. What is the effectiveness of parameter-efficient methods (LoRA) vs full fine-tuning?

### Experimental Setup

#### Models
- **BERT-base-uncased**: 110M parameters, bidirectional encoder
- **GPT-Neo-125M**: 125M parameters, autoregressive decoder

#### Fine-tuning Methods

##### 1. Baseline
- Standard training for 3 epochs
- Learning rate: 2e-5 (BERT), 3e-3 (GPT-Neo)
- Batch size: 16
- No hyperparameter optimization

##### 2. Full Fine-tuning
- All model parameters trainable
- Hyperparameter optimization with Optuna
- Search space:
  - Learning rate: [1e-5, 5e-5] (log scale)
  - Batch size: [4, 8, 16, 32]
- Budget: 100-200 steps

##### 3. LoRA (Low-Rank Adaptation)
- Parameter-efficient fine-tuning
- Only low-rank matrices trainable
- Hyperparameters:
  - Rank (r): [4, 8, 16]
  - Alpha: [16, 32]
  - Dropout: [0.0, 0.3]
  - Learning rate: [1e-5, 1e-3] (log scale)

##### 4. Partial Freezing
- Freeze lower transformer layers
- Hyperparameters:
  - Freeze percentage: [0.25, 0.75]
  - Learning rate: [1e-5, 5e-5] (log scale)
  - Batch size: [4, 8, 16, 32]

### Task-Specific Adaptations

#### Named Entity Recognition (NER)
- **Task Type**: Token classification
- **Input**: Tokenized document text
- **Output**: BIO tags for each token
- **Labels**: 39 classes (O + 19 entity types × 2)
- **Metrics**: Micro-averaged F1, Precision, Recall

#### Relation Extraction (RE)
- **Task Type**: Sentence classification
- **Input**: `[HEAD] [SEP] [TAIL] [SEP] [CONTEXT]`
- **Output**: Relation type (76 classes + no_relation)
- **Data Augmentation**: Negative sampling for no_relation
- **Metrics**: Micro-averaged F1, Precision, Recall

### Hyperparameter Optimization

#### Framework: Optuna
- **Objective**: Maximize F1-score on development set
- **Trials**: 8 per method (computational budget constraints)
- **Pruning**: Median pruning to stop unpromising trials early
- **Search Algorithm**: TPE (Tree-structured Parzen Estimator)

#### Evaluation Protocol
1. Train on training set with suggested hyperparameters
2. Evaluate on development set every 20 steps
3. Return final F1-score as objective value
4. Select best hyperparameters for final training

### Computational Resources
- **GPU**: NVIDIA RTX 3080 (12GB VRAM)
- **Training Time**: 
  - BERT: ~2-3 hours per method
  - GPT-Neo: ~3-4 hours per method
- **Total Compute**: ~40 GPU hours

### Reproducibility
- **Random Seeds**: Fixed at 42 for all experiments
- **Framework Versions**: 
  - PyTorch 1.12.0
  - Transformers 4.21.0
  - PEFT 0.4.0
- **Deterministic Operations**: Enabled where possible