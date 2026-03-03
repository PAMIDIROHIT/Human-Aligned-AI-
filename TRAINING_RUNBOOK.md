# RLHF Pipeline — Training Runbook

> Everything you need to set up, fix, and run the full SFT → RM → PPO pipeline.
> Last updated: March 2026

---

## Table of Contents

1. [Current System Status](#1-current-system-status)
2. [Critical Blockers to Fix First](#2-critical-blockers-to-fix-first)
3. [Environment Variables & API Keys](#3-environment-variables--api-keys)
4. [Start Background Services](#4-start-background-services)
5. [HuggingFace Model Access](#5-huggingface-model-access)
6. [Stage 1 — SFT Training](#6-stage-1--sft-training)
7. [Stage 2 — Reward Model Training](#7-stage-2--reward-model-training)
8. [Stage 3 — PPO Training](#8-stage-3--ppo-training)
9. [Stage 4 — MT-Bench Evaluation](#9-stage-4--mt-bench-evaluation)
10. [Run Full Pipeline with DVC](#10-run-full-pipeline-with-dvc)
11. [Monitoring Setup](#11-monitoring-setup)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Current System Status

Checked on this machine — here is what we have right now:

| Component          | Status         | Details                                         |
|--------------------|----------------|-------------------------------------------------|
| GPU                | 4x Tesla K80   | 11 GB VRAM each, Compute Capability 3.7        |
| NVIDIA Driver      | 470.256.02     | Supports up to CUDA 11.4                        |
| PyTorch installed  | 2.10.0+cu128   | Built for CUDA 12.8 — MISMATCH with driver     |
| CUDA available     | NO             | PyTorch can't see GPU due to driver mismatch    |
| Python             | 3.12.3         | OK                                              |
| bitsandbytes       | 0.49.2         | Installed but needs GPU with CC >= 7.5 for 4-bit|
| Prometheus         | Installed       | User confirmed working                          |
| Grafana            | Not in PATH    | Needs install or path config                    |
| MLflow             | Installed       | Not running yet                                 |
| DVC                | Installed       | Not initialized yet                             |
| HF_TOKEN           | NOT SET        | Needed for Llama model download                 |
| LANGCHAIN_API_KEY  | NOT SET        | Needed for MT-Bench GPT-4 judge                 |
| MLFLOW_TRACKING_URI| NOT SET        | Needed for experiment logging                   |

---

## 2. Critical Blockers to Fix First

### BLOCKER 1: PyTorch CUDA Version Mismatch (Must Fix)

Your driver (470.x) supports **CUDA 11.4**, but the installed PyTorch was built for **CUDA 12.8**.
That's why `torch.cuda.is_available()` returns `False`.

**Fix — install PyTorch built for CUDA 11.8** (closest supported):

```bash
source /home/mohanganesh/ROHIT/GAI74/.venv/bin/activate

pip uninstall torch torchvision torchaudio -y

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

After installing, verify:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expected output: `CUDA: True | Device: Tesla K80`

> **Why cu118?** Driver 470.x provides CUDA 11.4 runtime compatibility.
> PyTorch cu118 wheels use 11.8 headers but are backward-compatible with 11.4 drivers
> as long as the driver supports the minimum required PTX. PyTorch 2.1.2+cu118 is the
> last stable version known to work well with K80 GPUs.

### BLOCKER 2: Tesla K80 Does NOT Support 4-bit Quantization

The K80 has compute capability 3.7. bitsandbytes 4-bit (NF4) quantization needs **CC >= 7.5** (Turing or newer like T4, A100, RTX 3090, etc.).

**What this means**: QLoRA with `load_in_4bit=True` will fail on K80.

**Fix — switch to float32 LoRA (no quantization)**:

We need to update `params.yaml` to disable quantization:

```yaml
sft:
  qlora:
    r: 64             # keep the same
    lora_alpha: 16
    target_modules: ["q_proj", "v_proj"]
    lora_dropout: 0.05
    bias: "none"
    task_type: "CAUSAL_LM"

  quantization:
    load_in_4bit: false      # <-- CHANGE: disable 4-bit
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_compute_dtype: "float16"
    bnb_4bit_use_double_quant: true

  training:
    fp16: false              # <-- CHANGE: K80 has poor fp16 support
    # everything else stays the same
```

**Memory impact without quantization** (Llama-3.2-1B in float32):

| Stage      | Model Size (fp32) | With LoRA Adapters | Fits in 11 GB K80? |
|------------|-------------------|--------------------|---------------------|
| SFT        | ~4 GB             | ~4.5 GB            | YES                 |
| Reward     | ~4 GB             | ~4.5 GB            | YES                 |
| PPO        | ~4 GB x 2 models  | ~9-10 GB           | TIGHT — see note    |

> **PPO Note**: With policy + reference model + value head + reward model, 4 copies
> won't fit on a single K80 in fp32. Options:
> - Use `device_map="auto"` with CPU offloading via accelerate
> - Use 2 K80s with `CUDA_VISIBLE_DEVICES=0,1`
> - Reduce `batch_size` and `mini_batch_size` in params.yaml

**Alternative (Recommended if possible)**: Use Google Colab (free T4 GPU, CC 7.5)
or any machine with T4/A100 — then QLoRA 4-bit works as-is, no changes needed.

---

## 3. Environment Variables & API Keys

Create a file at `~/.rlhf_env` and source it before every session:

```bash
cat > ~/.rlhf_env << 'EOF'
# Activate virtual environment
source /home/mohanganesh/ROHIT/GAI74/.venv/bin/activate

# HuggingFace — needed to download Llama model (gated)
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# MLflow — local experiment tracking server
export MLFLOW_TRACKING_URI="http://localhost:5000"

# LangSmith — needed for MT-Bench evaluation with GPT-4 judge
export LANGCHAIN_API_KEY="lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxx"
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_PROJECT="rlhf-pipeline-eval"

# Disable Weights & Biases (we use MLflow)
export WANDB_DISABLED="true"

# Avoid tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM="false"

# GPU selection (single GPU by default)
export CUDA_VISIBLE_DEVICES="0"
EOF
```

Then every time you open a terminal:
```bash
source ~/.rlhf_env
```

### Where to get these keys

| Key               | Where to Get It                                            |
|--------------------|------------------------------------------------------------|
| HF_TOKEN           | https://huggingface.co/settings/tokens → create "Read" token |
| LANGCHAIN_API_KEY  | https://smith.langchain.com → Settings → API Keys          |

### HuggingFace Token — Accept Model License

The Llama model is gated. After creating your HF token:

1. Go to https://huggingface.co/meta-llama/Llama-3.2-1B
2. Click "Accept License" (Meta community license)
3. Wait a few minutes for approval
4. Then the model will download when training starts

---

## 4. Start Background Services

### 4a. MLflow Tracking Server

```bash
source ~/.rlhf_env
cd /home/mohanganesh/ROHIT/GAI74/rlhf-pipeline

# Start MLflow (runs on http://localhost:5000)
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000 &

echo "MLflow running at http://localhost:5000"
```

Verify: Open browser → `http://localhost:5000` — you should see the MLflow UI.

### 4b. Prometheus (Already Working)

You mentioned Prometheus is already running. Make sure it uses our config:

```bash
# If not already running with our config:
prometheus --config.file=/home/mohanganesh/ROHIT/GAI74/rlhf-pipeline/monitoring/prometheus.yml &
```

Verify: `http://localhost:9090` should show Prometheus UI.

### 4c. Grafana (Install if not present)

```bash
# Check if installed
which grafana-server

# If not installed (Ubuntu/Debian):
sudo apt-get install -y grafana
sudo systemctl start grafana-server
sudo systemctl enable grafana-server
```

Setup after install:
1. Open `http://localhost:3000` (default login: admin/admin)
2. Add Prometheus data source → URL: `http://localhost:9090`
3. Import dashboard → Upload `monitoring/grafana_dashboard.json`

### 4d. Initialize DVC

```bash
cd /home/mohanganesh/ROHIT/GAI74/rlhf-pipeline
dvc init
mkdir -p /tmp/dvc_storage
dvc remote add -d local_storage /tmp/dvc_storage
```

---

## 5. HuggingFace Model Access

Test that your token works and the model is accessible:

```bash
source ~/.rlhf_env
python -c "
from huggingface_hub import login
login(token='$HF_TOKEN')
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
print('Tokenizer loaded! Vocab size:', tok.vocab_size)
"
```

If this fails with "gated repo" error → go accept the license (Step 3 above).

---

## 6. Stage 1 — SFT Training

### What it does
Fine-tunes Llama-3.2-1B on the Alpaca instruction dataset using LoRA adapters.

### Pre-checks
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] HF_TOKEN is set and model license accepted
- [ ] MLflow server is running
- [ ] `params.yaml` quantization is configured for your GPU (see Blocker 2)

### Run

```bash
source ~/.rlhf_env
cd /home/mohanganesh/ROHIT/GAI74/rlhf-pipeline

# Run SFT training
PYTHONPATH=. python src/stage1_sft/train.py
```

### What to expect
- Downloads ~2 GB model on first run
- Downloads ~52k Alpaca examples (~24 MB)
- Training: ~3 epochs, takes 2-6 hours on K80 (fp32) or ~30 min on T4 (4-bit)
- Outputs saved to: `models/sft_adapter/`
- Metrics logged to MLflow experiment: `rlhf-sft`

### Eval gate

```bash
PYTHONPATH=. python src/stage1_sft/evaluate.py
```

Pass criteria: perplexity ≤ 15.0

---

## 7. Stage 2 — Reward Model Training

### What it does
Trains a reward model on Anthropic HH-RLHF chosen/rejected pairs using Bradley-Terry loss.

### Pre-checks
- [ ] Stage 1 completed — `models/sft_adapter/` exists with adapter files
- [ ] MLflow still running

### Run

```bash
source ~/.rlhf_env
cd /home/mohanganesh/ROHIT/GAI74/rlhf-pipeline

# Run reward model training
PYTHONPATH=. python src/stage2_reward/train.py
```

### What to expect
- Downloads HH-RLHF dataset (~170k pairs, ~200 MB first time)
- Merges SFT LoRA into base model, then adds reward head
- Training: 1 epoch, takes 1-3 hours on K80
- Outputs saved to: `models/reward_model/`
- Metrics logged to MLflow experiment: `rlhf-reward-model`

### Eval gate

```bash
PYTHONPATH=. python src/stage2_reward/evaluate.py
```

Pass criteria: accuracy ≥ 60% AND positive margin ratio ≥ 60%

---

## 8. Stage 3 — PPO Training

### What it does
Optimizes the SFT policy against the reward model using PPO with adaptive KL control.

### Pre-checks
- [ ] Stage 1 completed — `models/sft_adapter/` exists
- [ ] Stage 2 completed — `models/reward_model/` exists
- [ ] MLflow still running
- [ ] Prometheus running (for live metrics)
- [ ] Enough VRAM — PPO needs ~9-10 GB (see Blocker 2 notes)

### Run

```bash
source ~/.rlhf_env
cd /home/mohanganesh/ROHIT/GAI74/rlhf-pipeline

# Run PPO training
PYTHONPATH=. python src/stage3_ppo/train.py
```

### What to expect
- Loads 4 models: policy (LoRA), reference, reward model, value head
- 2000 training steps, each step: generate → score → PPO update
- Takes 4-8 hours on a T4, longer on K80 (if it fits)
- Live metrics at `http://localhost:9091/metrics` (Prometheus scrapes this)
- Outputs saved to: `models/ppo_policy/`
- Metrics logged to MLflow experiment: `rlhf-ppo`

### Eval gate

```bash
PYTHONPATH=. python src/stage3_ppo/evaluate.py
```

Pass criteria: final KL ≤ 0.15 AND reward improvement ≥ 0.1

---

## 9. Stage 4 — MT-Bench Evaluation

### What it does
Evaluates base model vs SFT vs PPO on 80 MT-Bench questions using GPT-4 as judge.

### Pre-checks
- [ ] All 3 stages completed
- [ ] LANGCHAIN_API_KEY is set (needs OpenAI credits for GPT-4 calls)
- [ ] This costs money — ~80 GPT-4 API calls × 3 models = ~240 calls ≈ $2-5

### Run

```bash
source ~/.rlhf_env
cd /home/mohanganesh/ROHIT/GAI74/rlhf-pipeline

PYTHONPATH=. python src/evaluation/mt_bench.py
```

### Output
- Scores saved to: `reports/mt_bench_scores.json`
- Traces visible at: https://smith.langchain.com

---

## 10. Run Full Pipeline with DVC

Instead of running each stage manually, DVC runs everything in order:

```bash
source ~/.rlhf_env
cd /home/mohanganesh/ROHIT/GAI74/rlhf-pipeline

# Initialize DVC (one time only)
dvc init
dvc remote add -d local_storage /tmp/dvc_storage

# Run the full pipeline
dvc repro
```

DVC will run: `stage1_sft → stage1_eval → stage2_reward → stage2_eval → stage3_ppo → stage3_eval → evaluate`

If any eval gate fails, the pipeline stops at that stage.

---

## 11. Monitoring Setup

### Live Dashboard During PPO Training

| Service     | URL                     | Purpose                        |
|-------------|-------------------------|--------------------------------|
| MLflow      | http://localhost:5000   | Experiment tracking, metrics   |
| Prometheus  | http://localhost:9090   | PPO training metrics scraping  |
| Grafana     | http://localhost:3000   | Real-time PPO dashboard        |

### Grafana Dashboard Import
1. Open Grafana → Dashboards → Import
2. Upload: `monitoring/grafana_dashboard.json`
3. Select Prometheus as data source
4. You'll see: reward curves, KL divergence, entropy, clip fraction, etc.

---

## 12. Troubleshooting

### "CUDA out of memory" during PPO
```yaml
# In params.yaml, reduce batch sizes:
ppo:
  config:
    batch_size: 32       # was 128
    mini_batch_size: 8   # was 32
```

### "The NVIDIA driver on your system is too old"
You have the wrong PyTorch version. Fix with Blocker 1 above.

### "bitsandbytes: CUDA Setup failed"
Your GPU doesn't support 4-bit. Disable quantization (Blocker 2 above).

### MLflow connection refused
```bash
# Check if mlflow is running
curl http://localhost:5000/api/2.0/mlflow/experiments/search
# If not, start it (see Step 4a)
```

### Model download fails (401 / gated)
- Check HF_TOKEN is set: `echo $HF_TOKEN`
- Accept license at: https://huggingface.co/meta-llama/Llama-3.2-1B
- Try logging in: `huggingface-cli login`

### Training is extremely slow on K80
K80 is old hardware (2014). Expected approximate times:
- SFT: 4-8 hours
- RM: 2-4 hours
- PPO: 6-12 hours

To speed up: reduce `max_samples` in params.yaml for a test run, or use Colab/cloud GPU.

---

## Quick-Start Checklist (Do These In Order)

```
[ ] 1. Fix PyTorch CUDA version          → Section 2, Blocker 1
[ ] 2. Verify torch.cuda.is_available()  → should print True
[ ] 3. Decide: 4-bit (need T4+) or fp32  → Section 2, Blocker 2
[ ] 4. Update params.yaml if using fp32  → Section 2, Blocker 2
[ ] 5. Get HuggingFace token + accept license → Section 3 & 5
[ ] 6. Create ~/.rlhf_env with all keys  → Section 3
[ ] 7. Start MLflow server               → Section 4a
[ ] 8. Initialize DVC                    → Section 4d
[ ] 9. Run Stage 1: SFT                 → Section 6
[ ] 10. Check SFT eval gate passes       → Section 6
[ ] 11. Run Stage 2: Reward Model        → Section 7
[ ] 12. Check RM eval gate passes        → Section 7
[ ] 13. Run Stage 3: PPO                 → Section 8
[ ] 14. Check PPO eval gate passes       → Section 8
[ ] 15. Run MT-Bench evaluation          → Section 9
[ ] 16. Open Grafana dashboard           → Section 11
[ ] 17. Review results in MLflow UI      → Section 11
```

---

## File Reference

```
rlhf-pipeline/
├── params.yaml                          # All hyperparameters (edit this)
├── dvc.yaml                             # Pipeline DAG
├── requirements.txt                     # Python packages
├── CLAUDE.md                            # Architecture decisions
├── TRAINING_RUNBOOK.md                  # THIS FILE
│
├── src/
│   ├── stage1_sft/
│   │   ├── dataset.py                   # Alpaca data loader
│   │   ├── train.py                     # SFT training script
│   │   └── evaluate.py                  # Perplexity eval gate
│   ├── stage2_reward/
│   │   ├── dataset.py                   # HH-RLHF data loader
│   │   ├── model.py                     # Reward model architecture
│   │   ├── train.py                     # RM training script
│   │   └── evaluate.py                  # Accuracy eval gate
│   ├── stage3_ppo/
│   │   ├── dataset.py                   # PPO prompt loader
│   │   ├── kl_controller.py             # Adaptive KL controller
│   │   ├── train.py                     # PPO training loop
│   │   └── evaluate.py                  # KL + reward eval gate
│   └── evaluation/
│       ├── mt_bench.py                  # MT-Bench with GPT-4 judge
│       └── kl_reward_tradeoff.py        # Pareto analysis
│
├── models/                              # Created during training
│   ├── sft_adapter/                     # Stage 1 output
│   ├── reward_model/                    # Stage 2 output
│   └── ppo_policy/                      # Stage 3 output
│
├── monitoring/
│   ├── prometheus.yml                   # Prometheus scrape config
│   └── grafana_dashboard.json           # Import into Grafana
│
├── docker/                              # Dockerfiles per stage
├── notebooks/analysis.ipynb             # Post-training analysis
└── tests/                               # 55 unit tests
```
