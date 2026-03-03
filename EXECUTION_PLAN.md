# RLHF Pipeline Execution Plan

## Hardware & Environment

| Component | Specification |
|-----------|--------------|
| **GPUs** | 4x Tesla K80 (11,441 MB each, CC 3.7) |
| **Driver** | NVIDIA 470.256.02 |
| **CUDA** | 11.4 |
| **Python** | 3.10.19 (conda env: `rlhf`) |
| **PyTorch** | 1.12.1+cu113 (fp32 only, no fp16/bf16 on K80) |
| **Training Precision** | fp32 (K80 lacks tensor cores) |
| **Model** | TinyLlama-1.1B (~4.4 GB at fp32) |
| **Server IP** | 10.0.24.7 |

## Pre-Flight Checks

```bash
# 1. Activate environment
eval "$(conda shell.bash hook)" && conda activate rlhf

# 2. Verify GPUs are visible
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# 3. Verify datasets exist
ls -la data/sft/ data/reward/ data/ppo/

# 4. Start MLflow tracking server (background)
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns &
```

---

## Stage 1: Supervised Fine-Tuning (SFT)

**Datasets:** UltraChat-200k (207K samples) + OpenAssistant Guanaco (9.8K samples)
**Model:** TinyLlama-1.1B → LoRA fine-tuned

### Memory Estimate
| Component | Memory |
|-----------|--------|
| Model (fp32) | ~4.4 GB |
| LoRA adapters | ~0.05 GB |
| Optimizer states (AdamW) | ~0.1 GB |
| Gradients + activations (batch=2) | ~3-5 GB |
| **Total per GPU** | **~8-10 GB** (fits in 11 GB) |

### Launch Command (Multi-GPU)
```bash
accelerate launch \
    --config_file accelerate_config.yaml \
    -m src.stage1_sft.train \
    --config params.yaml
```

### Launch Command (Single GPU)
```bash
CUDA_VISIBLE_DEVICES=0 python -m src.stage1_sft.train --config params.yaml
```

### Expected Runtime
- **Samples:** ~217K combined
- **Batch size:** 2 per GPU × 4 GPUs = effective 8
- **Gradient accumulation:** 4 → effective 32
- **Steps:** ~217K / 32 × 3 epochs ≈ 20,344 steps
- **K80 speed:** ~1.5 steps/sec
- **Estimated time:** ~3.8 hours

### Outputs
- `models/sft_adapter/` — LoRA adapter weights
- MLflow experiment: `rlhf-sft`

### Evaluate SFT
```bash
python -m src.stage1_sft.evaluate --config params.yaml
```

---

## Stage 2: Reward Model Training

**Datasets:** Anthropic HH-RLHF (160K pairs) + UltraFeedback Binarized (61K pairs)
**Model:** TinyLlama-1.1B → Reward head

### Memory Estimate
| Component | Memory |
|-----------|--------|
| Model (fp32) | ~4.4 GB |
| Reward head | ~0.01 GB |
| Optimizer states | ~0.1 GB |
| Gradients + activations (batch=4) | ~4-6 GB |
| **Total per GPU** | **~9-11 GB** (tight fit) |

### Launch Command (Multi-GPU)
```bash
accelerate launch \
    --config_file accelerate_config.yaml \
    -m src.stage2_reward.train \
    --config params.yaml
```

### Launch Command (Single GPU — safer for memory)
```bash
CUDA_VISIBLE_DEVICES=0 python -m src.stage2_reward.train --config params.yaml
```

### Expected Runtime
- **Samples:** ~221K combined pairs
- **Batch size:** 4 per GPU × 4 GPUs = effective 16
- **Steps:** ~221K / 16 × 1 epoch ≈ 13,813 steps
- **Estimated time:** ~2.5 hours

### Outputs
- `models/reward_model/` — Full reward model
- MLflow experiment: `rlhf-reward`

### Evaluate Reward Model
```bash
python -m src.stage2_reward.evaluate --config params.yaml
```

---

## Stage 3: PPO Reinforcement Learning

**Datasets:** UltraFeedback-Binarized-Gen (61K prompts) + PKU-SafeRLHF (73K prompts)  
**Models loaded:** Actor (SFT), Reference (SFT copy), Reward model, Value head

### Memory Estimate (Multi-GPU Distribution)
| GPU | Model | Memory |
|-----|-------|--------|
| GPU 0-1 | Actor + Value head (auto sharded) | ~5 GB each |
| GPU 2 | Reward model | ~4.4 GB |
| GPU 3 | Reference model | ~4.4 GB |

### Launch Command
```bash
# PPO manages its own multi-GPU placement — DO NOT use accelerate launch
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.stage3_ppo.train --config params.yaml
```

### Expected Runtime
- **Prompts:** ~134K combined (deduplicated)
- **PPO batch size:** 16, mini-batch: 4
- **PPO epochs per batch:** 4
- **Steps:** ~134K / 16 × 1 pass ≈ 8,375 steps
- **K80 speed (PPO is slower):** ~0.3 steps/sec
- **Estimated time:** ~7.7 hours

### Outputs
- `models/ppo_policy/` — PPO-trained adapter
- `reports/ppo_metrics.json` — Training metrics
- MLflow experiment: `rlhf-ppo`

### Evaluate PPO
```bash
python -m src.stage3_ppo.evaluate --config params.yaml
```

---

## Stage 4: Evaluation & Reporting

### MT-Bench Evaluation
```bash
# Requires OpenAI API key for GPT-4 judge (optional)
export OPENAI_API_KEY="your-key"
python -m src.evaluation.mt_bench --config params.yaml
```

### KL-Reward Tradeoff Analysis
```bash
python -m src.evaluation.kl_reward_tradeoff --config params.yaml
```

### Outputs
- `reports/mt_bench_scores.json`
- `reports/kl_reward_tradeoff.png`
- `reports/ppo_metrics.json`

---

## Remote Prometheus Monitoring (from your laptop)

### On GPU Server (automatic — PPO train.py does this)
Metrics are exposed on `0.0.0.0:9091` during PPO training.

### On Your Laptop
1. Copy `monitoring/prometheus.yml` to your laptop
2. Edit: Replace `<GPU_SERVER_IP>` with `10.0.24.7`
3. Start Prometheus:
   ```bash
   prometheus --config.file=prometheus.yml
   ```
4. Open browser: `http://localhost:9090`
5. Available metrics:
   - `rlhf_ppo_reward` — reward per PPO step
   - `rlhf_ppo_kl_divergence` — KL divergence from reference
   - `rlhf_ppo_policy_loss` — PPO policy loss
   - `rlhf_ppo_value_loss` — value function loss
   - `rlhf_ppo_entropy` — policy entropy

---

## Complete Pipeline (Sequential Execution)

```bash
# Activate environment
eval "$(conda shell.bash hook)" && conda activate rlhf
cd /home/mohanganesh/ROHIT/GAI74/rlhf-pipeline

# Start MLflow
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns &

# Stage 1: SFT (~3.8 hours)
accelerate launch --config_file accelerate_config.yaml -m src.stage1_sft.train --config params.yaml
python -m src.stage1_sft.evaluate --config params.yaml

# Stage 2: Reward Model (~2.5 hours)
accelerate launch --config_file accelerate_config.yaml -m src.stage2_reward.train --config params.yaml
python -m src.stage2_reward.evaluate --config params.yaml

# Stage 3: PPO (~7.7 hours)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.stage3_ppo.train --config params.yaml
python -m src.stage3_ppo.evaluate --config params.yaml

# Stage 4: Final Evaluation
python -m src.evaluation.mt_bench --config params.yaml
python -m src.evaluation.kl_reward_tradeoff --config params.yaml
```

**Total estimated time: ~14 hours**

---

## Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size` in `params.yaml`
- SFT: try batch 1, gradient_accumulation 8
- Reward: try batch 2
- PPO: try batch_size 8, mini_batch_size 2

### Slow Training
- K80 GPUs are compute-limited; this is expected
- Gradient checkpointing is enabled to save memory
- Monitor GPU utilization: `watch -n1 nvidia-smi`

### Multi-GPU Issues
- Verify all GPUs: `python -c "import torch; print(torch.cuda.device_count())"`
- Check NCCL: `NCCL_DEBUG=INFO accelerate launch ...`
- Fall back to single GPU: `CUDA_VISIBLE_DEVICES=0`

### MLflow Not Connecting
- Ensure server is running: `curl http://localhost:5000`
- Check port not blocked: `ss -tlnp | grep 5000`

### Prometheus Not Scraping
- Verify from laptop: `curl http://10.0.24.7:9091/metrics`
- Check firewall allows port 9091
- Ensure PPO training is actively running (metrics only active during training)
