# DocIE: Document-level Information Extraction

## Project Overview

This project implements and compares different fine-tuning approaches for Named Entity Recognition (NER) and Relation Extraction (RE) tasks using the DocIE dataset. We evaluate BERT and GPT-Neo models with three fine-tuning strategies: Full Fine-tuning, LoRA, and Partial Freezing.

### Key Results Summary
- **BERT NER**: Full fine-tuning achieved best F1-score of **89.16%**
- **BERT RE**: Baseline achieved best F1-score of **50.83%**  
- **GPT-Neo NER**: Full fine-tuning achieved best F1-score of **21.42%**
- **GPT-Neo RE**: Full fine-tuning achieved best F1-score of **33.54%**

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Experiments](#experiments)
- [Results](#results)
- [Usage](#usage)
- [Team Contributions](#team-contributions)

## Installation

### Requirements
```bash bash
pip install -r requirements.txt
```

### Dependencies
- torch>=1.9.0
- transformers>=4.21.0
- datasets>=2.0.0
- optuna>=3.0.0
- peft>=0.4.0
- scikit-learn>=1.0.0
- matplotlib>=3.5.0
- seaborn>=0.11.0

### Setup
```bash
git clone <repository-url>
cd NLP_Project_DocIE
pip install -e .
```

## Dataset

The DocIE dataset contains document-level information extraction annotations:
- **Training**: 51 documents
- **Development**: 23 documents  
- **Test**: 248 documents
- **Entity Types**: 19 (PERSON, ORG, GPE, DATE, etc.)
- **Relation Types**: 76 (HasPart, HasEffect, LocatedIn, etc.)

## Project Structure

### Notebooks
- `notebooks/BERT_NER.ipynb` - BERT Named Entity Recognition
- `notebooks/BERT_RE.ipynb` - BERT Relation Extraction
- `notebooks/GPT_NER.ipynb` - GPT-Neo Named Entity Recognition  
- `notebooks/GPT_RE.ipynb` - GPT-Neo Relation Extraction

### Source Code
- `src/data_processing.py` - Data loading and preprocessing utilities
- `src/models.py` - Model definitions and training functions
- `src/evaluation.py` - Evaluation metrics and visualization

## Methodology

### Models Evaluated
- **BERT-base-uncased** (110M parameters)
- **GPT-Neo-125M** (125M parameters)

### Fine-tuning Approaches
1. **Baseline**: Standard 3-epoch training
2. **Full Fine-tuning**: All parameters trainable with hyperparameter optimization
3. **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning
4. **Partial Freezing**: Freeze lower transformer layers

### Hyperparameter Optimization
- Framework: Optuna with 8 trials per method
- Budget: 100-200 training steps for fair comparison
- Metrics: F1-score for optimization target

## Experiments

### Named Entity Recognition (NER)
Token classification task with BIO tagging scheme.

**Input Format**: `[CLS] token1 token2 ... [SEP]`
**Output**: BIO tags for each token

### Relation Extraction (RE)  
Sentence-level classification between entity pairs.

**Input Format**: `[HEAD] [SEP] [TAIL] [SEP] [SENTENCE]`
**Output**: Relation type or "no_relation"

## Results

### BERT Performance

| Task | Method | F1-Score | Precision | Recall | Accuracy |
|------|--------|----------|-----------|--------|----------|
| NER | Baseline | 88.53% | 88.53% | 88.53% | 88.53% |
| NER | Full FT | **89.16%** | 89.16% | 89.16% | 89.16% |
| NER | LoRA | 5.60% | 5.60% | 5.60% | 5.60% |
| NER | Partial Freeze | 39.42% | 39.42% | 39.42% | 39.42% |
| RE | Baseline | **50.83%** | 50.83% | 50.83% | 50.83% |
| RE | Full FT | 50.00% | 50.00% | 50.00% | 50.00% |
| RE | LoRA | 50.00% | 50.00% | 50.00% | 50.00% |
| RE | Partial Freeze | 50.00% | 50.00% | 50.00% | 50.00% |

### GPT-Neo Performance

| Task | Method | F1-Score | Precision | Recall | Accuracy |
|------|--------|----------|-----------|--------|----------|
| NER | Baseline | 0.00% | 0.00% | 0.00% | 0.00% |
| NER | Full FT | **21.42%** | 21.42% | 21.42% | 21.42% |
| NER | LoRA | 12.07% | 12.07% | 12.07% | 12.07% |
| NER | Partial Freeze | 3.61% | 3.61% | 3.61% | 3.61% |
| RE | Baseline | 29.94% | 26.64% | 35.48% | 35.48% |
| RE | Full FT | **33.54%** | 26.60% | 45.71% | 45.71% |
| RE | LoRA | 29.23% | 25.46% | 34.32% | 34.32% |
| RE | Partial Freeze | 5.71% | 31.04% | 4.46% | 4.46% |

### Key Findings
1. **BERT significantly outperforms GPT-Neo** on both tasks
2. **Full fine-tuning** generally achieves best results
3. **LoRA struggles** with smaller datasets and complex tasks
4. **GPT-Neo baseline fails** for NER, requires fine-tuning

## Usage

### Running Individual Notebooks
```bash
jupyter notebook notebooks/BERT_NER.ipynb
```

### Training Custom Models
```python
from src.models import train_bert_ner
from src.data_processing import load_docie_data

# Load data
train_data, dev_data = load_docie_data()

# Train model
model, results = train_bert_ner(
    train_data=train_data,
    dev_data=dev_data,
    method="full_ft",
    hyperparams={"lr": 2e-5, "batch_size": 16}
)
```

### Reproducing Results
```bash
# Run all experiments
python scripts/run_all_experiments.py

# Generate visualizations
python scripts/create_visualizations.py
```

## Project Timeline

- **Week 1-2**: Data exploration and baseline implementation
- **Week 3-4**: Fine-tuning method implementation  
- **Week 5-6**: Hyperparameter optimization
- **Week 7**: Results analysis and documentation

## Team Contributions

| Team Member | Responsibilities |
|-------------|-----------------|
| [Your Name] | • BERT model implementation (NER & RE)<br>• GPT-Neo model implementation (NER & RE)<br>• Hyperparameter optimization with Optuna<br>• Results analysis and visualization<br>• Documentation and code cleanup<br>• Paper writing |

## References

1. DocIE Dataset: [https://xllms.github.io/DocIE/](https://xllms.github.io/DocIE/)
2. BERT: Devlin et al. (2018)
3. GPT-Neo: EleutherAI
4. LoRA: Hu et al. (2021)
5. Optuna: Akiba et al. (2019)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please contact [your-email@university.edu]
