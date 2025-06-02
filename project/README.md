# DocIE: Document-level Information Extraction

## Project Overview

This project implements and compares different fine-tuning approaches for Named Entity Recognition (NER) and Relation Extraction (RE) tasks using the DocIE dataset. We evaluate BERT and GPT-Neo models with three fine-tuning strategies: Full Fine-tuning, LoRA, and Partial Freezing.

### Key Results Summary
- **BERT NER**: Full fine-tuning achieved best F1-score of **89.16%**
- **BERT RE**: Baseline achieved best F1-score of **50.83%**  
- **GPT-Neo NER**: Full fine-tuning achieved best F1-score of **21.42%**
- **GPT-Neo RE**: Full fine-tuning achieved best F1-score of **33.54%**
- **Llama 2 3B v2 NER**: Baseline/Full FT achieved best F1-score of **88.86%**
- **Llama 2 3B v2 RE**: Baseline/Full FT achieved best F1-score of **50.00%**

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Experiments](#experiments)
- [Results](#results)
- [Data Augmentation](#Data-Augmentation)
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
git clone https://github.com/KypleeFearz/NLP_Project/tree/main
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
- `notebooks/Llama_NER.ipynb` - Llama 2 3B v2 Named Entity Recognition
- `notebooks/Llama_RE.ipynb` - Llama 2 3B v2 Relation Extraction

### Source Code
- `src/data_processing.py` - Data loading and preprocessing utilities
- `src/models.py` - Model definitions and training functions
- `src/evaluation.py` - Evaluation metrics and visualization

## Methodology

### Models Evaluated
- **BERT-base-uncased** (110M parameters)
- **GPT-Neo-125M** (125M parameters)
- **openlm-research/open_llama_3b_v2** (3B parameters)

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

### llama_3b_v2

| Task | Method         | F1 (micro) | F1 (macro) | Precision (micro) | Precision (macro) | Recall (micro) | Recall (macro) | Accuracy |
| :--- | :------------- | :----------- | :----------- | :---------------- | :------------------ | :--------------- | :--------------- | :--------- |
| NER  | Baseline       | 88.86%       | 25.64%       | 88.86%            | 31.63%              | 88.86%           | 27.62%           | 88.86%     |
| NER  | Full FT        | 88.86%       | 25.64%       | 88.86%            | 31.63%              | 88.86%           | 27.62%           | 88.86%     |
| NER  | LoRA           | 85.72%       | 12.10%       | 85.72%            | 13.32%              | 85.72%           | 11.87%           | 85.72%     |
| NER  | Partial Freeze | 84.41%       | 4.07%        | 84.41%            | 5.28%               | 84.41%           | 3.83%            | 84.41%     |
| RE   | Baseline       | 50.00%       | 2.15%        | 50.00%            | 1.61%               | 50.00%           | 3.23%            | 50.00%     |
| RE   | Full FT        | 50.00%       | 2.15%        | 50.00%            | 1.61%               | 50.00%           | 3.23%            | 50.00%     |
| RE   | LoRA           | 32.50%       | 1.61%        | 32.50%            | 1.34%               | 32.50%           | 2.65%            | 32.50%     |
| RE   | Partial Freeze | 45.54%       | 2.50%        | 45.54%            | 2.45%               | 45.54%           | 3.24%            | 45.54%     |

### Key Findings
1. **BERT significantly outperforms GPT-Neo** on both tasks
2. **Full fine-tuning** generally achieves best results
3. **LoRA struggles** with smaller datasets and complex tasks
4. **GPT-Neo baseline fails** for NER, requires fine-tuning

## Data Augmentation

To address the limited size of the original training dataset (51 documents) and improve model generalization, data augmentation techniques were applied [cite: `nlp_final_augmentation.ipynb`]. The following three methods were used:

1. **Entity Swapping**: Replaces 30% of entity mentions with other mentions of the same type and length within the document. This creates variations while maintaining entity type consistency.

2. **Mask and Fill**: Masks non-entity words (within the first 128 words of sentences) and uses BERT (`bert-base-uncased`) to predict replacements. This introduces lexical variations without affecting labeled entities.

3. **Paraphrasing**: Uses GPT-3.5-turbo to rewrite sentences while preserving tagged entities and their meaning. This creates syntactic and stylistic variations while maintaining semantic content.


### Dataset Expansion Results

The augmentation process significantly increased the number of training documents, with an overall increase factor of nearly 4x (from 74 original documents across categories to 292 augmented documents).

| Dataset Category | Original Documents | Augmented Documents | Increase Factor |
|------------------|-------------------|-------------------|----------------|
| Communication | 10 | 40 | 4x |
| Government | 9 | 36 | 4x |
| Entertainment | 12 | 48 | 4x |
| Energy | 10 | 40 | 4x |
| Education | 10 | 40 | 4x |
| Human Behavior | 13 | 51 | ~4x |
| Internet | 10 | 37 | ~4x |
| **Total** | **74** | **292** | **~3.95x** |

*Note: The original DocIE training set has 51 documents in total. The table above categorizes a subset of these for illustrating augmentation impact per category, summing to 74 for these categories and 292 after augmentation for these categories. The augmentation was applied to the full set of 51 training documents.*


## Performance Comparison: Augmented vs Non-Augmented Data

To evaluate the effectiveness of data augmentation, both GPT-Neo 125M and BERT were trained on the original dataset and the augmented dataset (~4x expansion).

### Named Entity Recognition (NER) Results

| Method | GPT-Neo Original | GPT-Neo Augmented | GPT-Neo Δ | BERT Original | BERT Augmented | BERT Δ |
|--------|------------------|-------------------|-----------|---------------|----------------|--------|
| Baseline | 0.00% | 5.77% | **+5.77%** | 88.53% | 89.60% | **+1.07%** |
| Full Fine-Tuning | 21.42% | 22.47% | **+1.05%** | 89.16% | 89.92% | **+0.76%** |
| LoRA | 12.07% | 23.22% | **+11.15%** | 5.60% | 17.80% | **+12.20%** |
| Partial Freeze | 3.61% | 1.73% | *-1.88%* | 39.42% | 17.01% | *-22.41%* |

### Relation Extraction (RE) Results

| Method | GPT-Neo Original | GPT-Neo Augmented | GPT-Neo Δ | BERT Original | BERT Augmented | BERT Δ |
|--------|------------------|-------------------|-----------|---------------|----------------|--------|
| Baseline | 29.94% | 35.28% | **+5.34%** | 50.83% | 51.54% | **+0.71%** |
| Full Fine-Tuning | 33.54% | 33.38% | *-0.16%* | 50.00% | 50.00% | *0.00%* |
| LoRA | 29.23% | 32.26% | **+3.03%** | 50.00% | 49.17% | *-0.83%* |
| Partial Freeze | 5.71% | 17.65% | **+11.94%** | 50.00% | 49.96% | *-0.04%* |

### Key Findings

**Augmentation Impact by Model Architecture:**
- **GPT-Neo**: Showed substantial improvements across most methods, particularly benefiting from increased data diversity
- **BERT**: Already high-performing on original data, showing more modest improvements with some diminishing returns

**Augmentation Impact by Fine-Tuning Method:**
- **LoRA (NER)**: Consistently showed largest improvements across both architectures (+11.15% GPT-Neo, +12.20% BERT)
- **Baseline Methods**: Reliable improvements, especially for GPT-Neo (+5.77% NER, +5.34% RE)
- **Partial Freeze**: Mixed results - significant GPT-Neo improvements but BERT degradation
- **Full Fine-Tuning**: Minimal changes, suggesting these methods already effectively utilized original data

**Overall Conclusions:**
- Data augmentation benefits are inversely related to baseline performance
- Parameter-efficient methods (LoRA) consistently benefit from augmented data across architectures
- Augmentation is most valuable when original model performance is limited by data scarcity


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

- **April and beginning of May**: Working on failed notebooks
- **Mid-May**:created the data augmentation notebook and augmented the dataset. This augmented data was then used for subsequent BERT and GPT-Neo runs.
- **Post-Presentation (late May)**: Following feedback, the Llama 2 3B v2 model was implemented.
- **Final of May**: Working on new notebooks and finalizing documentation and project paper


## Team Contributions

## Team Contributions

| Team Member          | Responsibilities                                                                                                                                                                                                                              |
| :------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Nikola Milosavljevic | • BERT model implementation (NER & RE)<br>• GPT-Neo model implementation (NER & RE)<br>• Hyperparameter optimization with Optuna for BERT and GPT-Neo<br>• Results analysis and visualization for BERT and GPT-Neo<br>• Initial failed notebooks exploration<br>• Documentation and code cleanup for BERT and GPT-Neo parts<br>• General paper writing and structuring |
| Youssef Riad         | • Llama 2 3B v2 model implementation (NER & RE)<br>• Hyperparameter optimization with Optuna for Llama<br>• Results analysis and visualization for Llama<br>• Implemented a T5-small model pipeline (not used in final results)<br>• Created and ran data augmentation notebook (with adjustments for GPT and BERT runs)<br>• Documentation for Llama part and data augmentation<br>• Report writing for Llama and data augmentation sections |

## References

## References

1.  **DocIE Dataset**: Sun, K., Lin, C. C., Liu, J., Zhang, Y., Zhao, T., & Lin, C. Y. (2024). *DocIE: A Large-Scale Dataset for Document-Level Information Extraction*. arXiv preprint arXiv:2402.01789. (Or the appropriate conference/publication if available - check the dataset's website: [https://xllms.github.io/DocIE/](https://xllms.github.io/DocIE/))
2.  **BERT**: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv preprint arXiv:1810.04805.
3.  **GPT-Neo**: Black, S., Biderman, S., Hallahan, E., Anthony, Q., Gao, L., Golding, L., ... & Leahy, C. (2021). *GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow*. EleutherAI.
4.  **Llama 2**: Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., ... & Lample, G. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models*. arXiv preprint arXiv:2307.09288.
5.  **LoRA**: Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv preprint arXiv:2106.09685.
6.  **Optuna**: Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.
7.  **Transformers (Hugging Face Library)**: Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2019). *HuggingFace's Transformers: State-of-the-art Natural Language Processing*. arXiv preprint arXiv:1910.03771.
8.  **PEFT (Hugging Face Library)**: Specific citation might vary, often cited via the Hugging Face documentation or the LoRA paper if PEFT is primarily used for LoRA. You can cite the GitHub repository: *PEFT: Parameter-Efficient Fine-Tuning of Billion-Scale Models on Low-Resource Hardware*. (2022). GitHub. Retrieved from [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
9.  **T5**: Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2019). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. arXiv preprint arXiv:1910.10683. (If you want to include the T5 model you experimented with).

## Contact

For questions or issues, please contact nikola.milosavljevic@student.unisg.ch or Youssef.Riad@student.unisg.ch
