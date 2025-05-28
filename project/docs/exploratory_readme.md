# Exploratory Notebooks - Failed Experiments & Learning

This directory contains experimental notebooks that didn't make it to the final implementation. These failures were crucial learning experiences that informed our successful approaches.

## File Organization

### Failed Experiments
- `mixed_pipeline_failed.ipynb` - **CRITICAL FAILURE** - Mixed BERT/GPT pipeline
- `GPT_neo_original_better.ipynb` - **BETTER PERFORMANCE** - Original GPT approach (F1: 86%)
- `BERT_early_attempt.ipynb` - **TECHNICAL ISSUES** - Early BERT implementation
- `docie_re_incomplete.ipynb` - **SCOPE MISMATCH** - Incomplete RE demo

### Status Legend
- **FAILED**: Abandoned due to critical issues
- **PARTIAL**: Some success but significant problems  
- **ITERATIVE**: Stepping stone to successful implementation
- **LEARNING**: Valuable lessons despite failure

---

## Mixed Pipeline Failed (`mixed_pipeline_failed.ipynb`)
**Status**:PARTIAL FAILURE  


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

## Learning Statistics

### Time Investment Analysis
| Notebook | Time Spent | Learning Value | Success Rate |
|----------|------------|----------------|--------------|
| Mixed Pipeline | ~2 weeks | **HIGH** | 50% |
| GPT-Neo Original | ~1 week | **MEDIUM** | 100% |
| BERT Early | ~1 week | **HIGH** | 30% |
| DocIE RE | ~3 days | **MEDIUM** | 10% |

### Most Valuable Failures
1. **Mixed Pipeline**: Taught us about model separation
2. **GPT-Neo Regression**: Showed importance of simple approaches  
3. **BERT Early**: Systematic debugging methodology

### Success Factors Identified
- **Single-model focus**: Avoid mixing models
- **Simple tokenization**: Start basic, add complexity only if needed
- **Incremental development**: One feature at a time
- **Systematic debugging**: Log everything, validate assumptions

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

---


### Key Takeaways
1. **Failures are valuable**: most of our time was "failed" experiments
2. **Simple often wins**: GPT-neo original (simple) > GPT_NER (complex)
3. **Pipeline separation matters**: Mixed approaches create chaos
4. **Systematic debugging works**: Incremental fixes lead to success

---

## Final Thoughts

These "failed" notebooks represent 60% of our development time but provided 80% of our learning. They taught us:
- **Model-specific requirements** (BERT vs GPT-Neo)
- **Debugging methodologies** (systematic problem-solving)
- **Performance optimization** (simple vs complex approaches)
- **Project management** (scope definition, time allocation)

**Most importantly**: They made our final successful implementations possible by teaching us what NOT to do.

> **"In machine learning, there are no failures—only expensive lessons."**