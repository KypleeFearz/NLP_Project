# Exploratory Notebooks - Failed Experiments & Learning

This directory contains experimental notebooks that didn't make it to the final implementation. These failures were crucial learning experiences that informed our successful approaches.

## File Organization

### Failed Experiments
- `mixed_pipeline_failed.ipynb` - **CRITICAL FAILURE** - Mixed BERT/GPT pipeline
- `GPT_neo_original_better.ipynb` - **BETTER PERFORMANCE** - Original GPT approach (F1: 86%)
- `BERT_early_attempt.ipynb` - **TECHNICAL ISSUES** - Early BERT implementation
- `docie_re_incomplete.ipynb` - **SCOPE MISMATCH** - Incomplete RE demo
- `nlp_augmentation_v1_failed.ipynb` - **ITERATIVE LEARNING** - Initial augmentation attempts
- `Llama_NER.ipynb` / `Llama_RE.ipynb` - **RESOURCE CRISIS** - Memory limitations

### Status Legend
- **FAILED**: Abandoned due to critical issues
- **PARTIAL**: Some success but significant problems  
- **ITERATIVE**: Stepping stone to successful implementation
- **LEARNING**: Valuable lessons despite failure

---

## Mixed Pipeline Failed (`mixed_pipeline_failed.ipynb`)
**Status**: PARTIAL FAILURE  

### What We Tried
Unified notebook executing in sequence:
1. BERT NER implementation
2. GPT-Neo NER implementation  
3. BERT RE implementation
4. GPT-Neo RE implementation

### Why It Failed
- **Tokenization Chaos**: BERT (WordPiece) vs GPT-Neo (BPE) incompatibility
- **Memory Leaks**: Model switching caused CUDA OOM errors
- **Pipeline Instability**: Inconsistent results across runs
- **Code Complexity**: Unmaintainable conditional logic

### Key Learning
> **"Keep model pipelines completely separate"**  
> Different models need dedicated environments for stable results

### Evidence of Issues
```python
# Typical error pattern from this notebook:
RuntimeError: Token alignment mismatch between BERT and GPT tokenizers
IndexError: Label indices out of bounds after tokenization switch
```

---

## GPT-Neo Original Better (`GPT_neo_original_better.ipynb`)
**Status**: SUPERIOR PERFORMANCE  
**F1 Score**: **86.0%** (vs 21.4% in final version)  
**Why Better**: Simple approach, gradient accumulation, higher learning rate

### What Worked
- **Gradient Accumulation**: 4x effective batch size
- **Higher Learning Rate**: 3e-3 vs 2e-5 in final version
- **Simple Tokenization**: No complex offset mapping
- **Focused Training**: 100 steps with careful evaluation

### Why We Couldn't Reproduce
- **Time Constraints**: Discovered performance difference late in project
- **Hyperparameter Conflict**: Final version used Optuna-optimized params
- **Code Evolution**: Final version had different architecture

### Key Learning
> **"Sometimes simple beats sophisticated"**  
> Over-engineering can hurt performance

### Performance Comparison
| Version | F1 Score | Training Config | Tokenization |
|---------|----------|----------------|--------------|
| **Original** | **86.0%** | 100 steps + grad accumulation | Simple |
| **Final** | 21.4% | 200 steps, no grad accumulation | Complex |

---

## BERT Early Attempt (`BERT_early_attempt.ipynb`)
**Status**: ITERATIVE LEARNING  
**Role**: Stepping stone to successful BERT_NER.ipynb

### Problems Encountered
1. **Label Alignment Issues**: BIO tags didn't match subword tokens
2. **Memory Problems**: OOM with batch_size=32
3. **Evaluation Bugs**: Incorrect F1 computation
4. **Data Preprocessing**: Inconsistent entity mention handling

### How We Fixed It
- **Offset Mapping**: Implemented `return_offsets_mapping=True`
- **Batch Size Optimization**: Reduced to 16, then 8
- **Metric Debugging**: Fixed entity-level evaluation
- **Data Validation**: Added consistency checks

### Evolution Path
```
BERT_early_attempt.ipynb (failed)
    ↓ fix tokenization
BERT_v2.ipynb (better)
    ↓ fix evaluation
BERT_v3.ipynb (working)
    ↓ optimize hyperparameters
BERT_NER.ipynb (final: F1=89.16%)
```

### Key Learning
> **"Debug systematically, fix incrementally"**  
> Each iteration taught us something valuable

---

## DocIE RE Incomplete (`docie_re_incomplete.ipynb`)
**Status**: SCOPE MISMATCH  
**Issue**: Built NER demo instead of RE implementation

### What Was Actually Built
worked:
- Test data loading in Pandas
- NER pipeline demonstration
- Basic functionality verification

did not work:
- **NO relation extraction**
- **NO triple prediction**
- **NO RE training**

### Why It Went Wrong
1. **Unclear Objectives**: Started without clear RE requirements
2. **Tool Limitations**: No good pre-trained RE models available
3. **Time Pressure**: Easier to demo NER than build RE from scratch
4. **Scope Creep**: Pivoted to easier task unconsciously

### What We Learned
- **Define scope clearly** before starting implementation
- **Pre-trained pipelines are limited** for specialized tasks
- **Time management matters** - focus on core objectives
- **RE is much harder than NER** - requires custom implementation

### Key Learning
> **"Scope definition prevents scope creep"**  
> Clear objectives save development time

---

## Initial Data Augmentation Struggles (`nlp_augmentation_v1_failed.ipynb`)
**Status**: ITERATIVE LEARNING  
**Issue**: Early augmented data degraded model performance

### What We Tried
- Entity swapping without length matching
- Unrestricted mask-and-fill across entire documents
- Basic paraphrasing without entity protection
- Minimal validation of augmented outputs

### Why It Failed
- **Annotation Corruption**: Entity boundaries shifted after text modifications
- **Semantic Drift**: Paraphrasing changed factual relationships
- **Label Misalignment**: Relations pointed to non-existent entity IDs
- **Quality Control**: No systematic validation of augmented data

### How We Fixed It
1. **Pre-cleaning Pass**: Remove invalid entities BEFORE augmentation
2. **Wrapped Functions**: Skip individual failures, keep valid augmentations
3. **Entity Protection**: Placeholder tags during paraphrasing
4. **Strict Validation**: Assert all mentions exist, relations are valid

### Evolution Path
```
nlp_augmentation_v1_failed.ipynb (corrupted data)
    ↓ add validation
augmentation1.py (better but brittle)
    ↓ add error handling
augmentation.py (robust, self-healing)
    ↓ result: 4x data expansion that actually helps
```

### Key Learning
> **"Bad augmentation is worse than no augmentation"**  
> Quality beats quantity - always validate synthetic data

---

## Llama Model Resource Crisis (`Llama_NER.ipynb`, `Llama_RE.ipynb`)
**Status**: TECHNICAL LIMITATION  
**Issue**: 3B model exceeded GPU memory, limiting experimentation

### What We Tried
- Llama-3B full fine-tuning on A100 40GB
- Multiple Optuna trials for hyperparameter optimization
- 4-bit NF4 quantization to reduce memory
- Systematic evaluation across all fine-tuning strategies

### Why It Failed
- **Peak Memory**: 38GB usage, kernel crashes
- **HPO Impossible**: Only 1 Optuna trial before OOM
- **Quantization Issues**: 4-bit still unstable for training
- **Restart Hell**: Kernel restart needed between each variant

### What Actually Worked (Partially)
- Single trial per variant with bfloat16
- Manual hyperparameter selection
- Aggressive gradient accumulation settings
- Immediate evaluation after training

### Key Learning
> **"Model size ≠ Model feasibility"**  
> Consider compute requirements before choosing architecture

### Evidence of Pain
```python
# From actual notebook output:
"Resource constraints limited us to one Optuna trial per variant"
"Peak memory reached 38GB on a single A100 40GB GPU"
"Total wall-clock time for all LLaMA runs: ~12h"
```

---

### Most Valuable Failures
1. **Mixed Pipeline**: Taught us about model separation
2. **GPT-Neo Regression**: Showed importance of simple approaches  
3. **BERT Early**: Systematic debugging methodology
4. **Data Augmentation**: Quality control is paramount
5. **Llama Resource**: Feasibility analysis matters

### Success Factors Identified
- **Single-model focus**: Avoid mixing models
- **Simple tokenization**: Start basic, add complexity only if needed
- **Incremental development**: One feature at a time
- **Systematic debugging**: Log everything, validate assumptions
- **Resource planning**: Know your hardware limits
- **Data quality**: Validate augmentations rigorously

---

## How These Failures Led to Success

### Pipeline Separation (Mixed → Individual notebooks)
```
mixed_pipeline_failed.ipynb
    ↓ lesson: separate models
BERT_NER.ipynb + GPT_NER.ipynb + BERT_RE.ipynb + GPT_RE.ipynb
    ↓ result: stable, reproducible pipelines
```

### Debugging Methodology (BERT Early → BERT Final)
```
BERT_early_attempt.ipynb (many bugs)
    ↓ systematic debugging
BERT_NER.ipynb (F1: 89.16%)
    ↓ robust implementation
```

### Performance Optimization (GPT Regression Analysis)
```
GPT_neo_original_better.ipynb (F1: 86%)
    ↓ performance analysis
GPT_NER.ipynb (F1: 21.4%)
    ↓ understanding: gradient accumulation crucial
```

### Data Quality Evolution (Augmentation Failures → Success)
```
nlp_augmentation_v1_failed.ipynb (corrupted annotations)
    ↓ validation & error handling
augmentation.py (self-healing pipeline)
    ↓ result: valid 4x data expansion
```

### Resource Management (Llama Struggles → Adapted Strategy)
```
Llama initial attempts (OOM crashes)
    ↓ quantization & batch size tuning
Llama final runs (stable but limited)
    ↓ lesson: plan for resource constraints
```

---

## Key Takeaways

1. **Failures are valuable**: Most of our time was "failed" experiments
2. **Simple often wins**: GPT-neo original (simple) > GPT_NER (complex)
3. **Pipeline separation matters**: Mixed approaches create chaos
4. **Systematic debugging works**: Incremental fixes lead to success
5. **Data quality trumps quantity**: Bad augmentation hurts more than helps
6. **Know your limits**: Hardware constraints shape what's possible

---

## Final Thoughts

These "failed" notebooks represent 60% of our development time but provided 80% of our learning. They taught us:
- **Model-specific requirements** (BERT vs GPT-Neo vs Llama)
- **Debugging methodologies** (systematic problem-solving)
- **Performance optimization** (simple vs complex approaches)
- **Data quality control** (validation is non-negotiable)
- **Resource management** (feasibility before ambition)
- **Project management** (scope definition, time allocation)

**Most importantly**: They made our final successful implementations possible by teaching us what NOT to do.

> **"In machine learning, there are no failures—only expensive lessons."**
