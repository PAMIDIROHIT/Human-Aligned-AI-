# RLHF Alignment Pipeline — Project Memory

## Model Choice Rationale
- **Base Model**: `meta-llama/Llama-3.2-1B`
- **Why**: 1B parameters is optimal for QLoRA fine-tuning on a single GPU. With 4-bit NF4
  quantization, the model footprint is ~0.6 GB, leaving ample headroom for optimizer states,
  activations, and gradient accumulation. A 7B model at 4-bit (~3.5 GB) would also fit but
  leaves less room for PPO's dual-model setup (policy + reference + reward + value head).
  The 1B model allows rapid iteration and full pipeline validation within 8–16 GB VRAM.
- **Fallback**: `mistralai/Mistral-7B-v0.1` if evaluation quality is too low with 1B.

## GPU Memory Budget Per Stage
| Stage | Models in VRAM | Estimated VRAM (fp16/4-bit) |
|-------|---------------|----------------------------|
| SFT (QLoRA) | Base (4-bit) + LoRA adapters | ~2 GB |
| Reward Model | Merged SFT model (4-bit) + reward head | ~3 GB |
| PPO | Policy (4-bit) + Ref (4-bit) + RM (4-bit) + Value head | ~8 GB |
| Evaluation | Single model inference (4-bit) | ~2 GB |

**Minimum GPU**: NVIDIA T4 (16 GB), Recommended: A100 (40 GB) for PPO stage.

## Dataset Preprocessing Decisions
### SFT (Stage 1): `tatsu-lab/alpaca`
- 52k instruction-following examples
- Prompt template: `### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}`
- Input field is optional; omitted when empty
- Train/Val split: 90/10 stratified
- Max sequence length: 512 tokens (packing enabled)

### RM + PPO (Stages 2 & 3): `Anthropic/hh-rlhf`
- ~170k chosen/rejected pairs for RM training
- Prompt-only subset for PPO rollouts
- Conversation format: `\n\nHuman:` / `\n\nAssistant:` delimiters
- Max sequence length: 512 tokens

### Evaluation: MT-Bench
- 80 multi-turn questions across 8 categories
- LangSmith LLM-as-judge scoring (GPT-4 judge)

## DVC Remote Configuration
```yaml
# .dvc/config
[core]
    remote = local_storage
['remote "local_storage"']
    url = /tmp/dvc_storage
```
For production, switch to S3: `s3://your-bucket/rlhf-pipeline/dvc-store`

## MLflow Tracking URI
- **Local**: `mlflow_tracking_uri: http://localhost:5000`
- **Experiment names**:
  - `rlhf-sft` (Stage 1)
  - `rlhf-reward-model` (Stage 2)
  - `rlhf-ppo` (Stage 3)
  - `rlhf-evaluation` (Final eval)

## Environment Variables
```bash
# Required environment variables
export MLFLOW_TRACKING_URI=http://localhost:5000
export HF_TOKEN=<your_huggingface_token>           # For gated model access
export LANGCHAIN_API_KEY=<your_langsmith_key>       # For LangSmith evaluation
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT=rlhf-pipeline-eval
export WANDB_DISABLED=true                          # Disable W&B, use MLflow
export TOKENIZERS_PARALLELISM=false                 # Avoid fork warnings
export CUDA_VISIBLE_DEVICES=0                       # Single GPU by default
```

## Hyperparameter Reference
All hyperparameters are defined in `params.yaml` and consumed by every training script.
Never hardcode hyperparameters in source code. Use `yaml.safe_load` to read them.

## Pipeline DAG
```
data/alpaca → [Stage 1: SFT] → models/sft_adapter/
                                      ↓
data/hh-rlhf → [Stage 2: RM] → models/reward_model/
                                      ↓
              [Stage 3: PPO] → models/ppo_policy/
                                      ↓
              [Evaluate: MT-Bench] → reports/mt_bench_scores.json
```

## Versions
- Python: 3.11+
- TRL: ≥0.9.0
- PEFT: ≥0.11.0
- Transformers: ≥4.41.0
- bitsandbytes: ≥0.43.0
- MLflow: ≥2.14.0
- DVC: ≥3.50.0
