# Monitoring Guide: Prometheus + DVC

## Server Details
- **GPU Server IP**: `10.0.24.7`
- **MLflow UI**: `http://10.0.24.7:5000`
- **Training Metrics Port**: `9091` (Prometheus endpoint from PPO stage)

---

## Part 1: Prometheus Monitoring

### What Prometheus Monitors
During the PPO training stage (Stage 3), the training script exposes live metrics at `http://10.0.24.7:9091/metrics`:
- `rlhf_mean_reward` — Average reward per batch
- `rlhf_kl_divergence` — KL divergence between policy and reference
- `rlhf_policy_loss` — PPO policy loss
- `rlhf_value_loss` — Value function loss
- `rlhf_entropy` — Policy entropy
- `rlhf_training_step` — Current training step counter

### Step-by-Step Setup (on your laptop)

#### 1. Install Prometheus
```bash
# macOS
brew install prometheus

# Linux (Ubuntu/Debian)
wget https://github.com/prometheus/prometheus/releases/download/v2.51.0/prometheus-2.51.0.linux-amd64.tar.gz
tar xzf prometheus-2.51.0.linux-amd64.tar.gz
cd prometheus-2.51.0.linux-amd64
```

#### 2. Copy the config file
Copy `monitoring/prometheus.yml` from this repo to your laptop:
```bash
scp mohanganesh@10.0.24.7:~/ROHIT/GAI74/rlhf-pipeline/monitoring/prometheus.yml ./prometheus.yml
```

#### 3. Start Prometheus
```bash
prometheus --config.file=prometheus.yml
```
Prometheus UI opens at: **http://localhost:9090**

#### 4. Verify scrape targets
1. Open http://localhost:9090/targets
2. You should see two targets:
   - `rlhf-training` → `10.0.24.7:9091` (will show UP during PPO training)
   - `mlflow` → `10.0.24.7:5000`

#### 5. Query metrics (PromQL)
In the Prometheus UI (http://localhost:9090/graph), enter these queries:

| Metric | PromQL Query |
|--------|-------------|
| Mean reward over time | `rlhf_mean_reward` |
| KL divergence | `rlhf_kl_divergence` |
| Reward rate of change | `rate(rlhf_mean_reward[5m])` |
| Policy loss | `rlhf_policy_loss` |
| Training step | `rlhf_training_step` |

#### 6. (Optional) Add Grafana Dashboard
```bash
# Install Grafana
brew install grafana  # macOS
# or: sudo apt install grafana  # Linux

# Start Grafana
grafana-server
```
1. Open http://localhost:3000 (default login: admin/admin)
2. Add Prometheus data source: http://localhost:9090
3. Import or create dashboard with the PromQL queries above

### Monitoring from the GPU Server Directly
If you don't have Prometheus on your laptop, you can still check metrics:
```bash
# Quick check — are metrics being exposed?
curl http://localhost:9091/metrics 2>/dev/null | grep rlhf_

# Watch metrics live (refresh every 10s)
watch -n 10 'curl -s http://localhost:9091/metrics | grep rlhf_'
```

> **Note**: The metrics endpoint (`port 9091`) is only active during PPO training (Stage 3). 
> For SFT and Reward Model stages, use MLflow and DVC instead.

---

## Part 2: MLflow Monitoring (All Stages)

MLflow tracks metrics for **all 3 training stages**. It's already running.

### Access MLflow UI
Open in browser: **http://10.0.24.7:5000**

Or via SSH tunnel from your laptop:
```bash
ssh -L 5000:localhost:5000 mohanganesh@10.0.24.7
# Then open: http://localhost:5000
```

### What's tracked in MLflow
| Stage | Experiment Name | Key Metrics |
|-------|----------------|-------------|
| SFT | `rlhf-sft` | loss, learning_rate, epoch, eval_loss |
| Reward Model | `rlhf-reward-model` | loss, accuracy, reward_margin |
| PPO | `rlhf-ppo` | mean_reward, kl_divergence, policy_loss |

### CLI queries
```bash
# List experiments
mlflow experiments search

# List runs for SFT experiment
mlflow runs list --experiment-id 1

# View specific run metrics
mlflow runs describe --run-id <RUN_ID>
```

---

## Part 3: DVC Pipeline Watching

### View the pipeline DAG
```bash
dvc dag
```
Output shows the dependency graph:
```
+------------+      +-------------+      +-----------+      +------------+
| stage1_sft | ---> | stage1_eval | ---> | stage2_rm | ---> | stage2_eval|
+------------+      +-------------+      +-----------+      +------------+
                                               |
                                               v
                                        +------------+      +------------+
                                        | stage3_ppo | ---> | stage3_eval|
                                        +------------+      +------------+
                                               |
                                               v
                                          +----------+
                                          | evaluate |
                                          +----------+
```

### Check pipeline status
```bash
# See which stages need to run / are outdated
dvc status

# Example output:
#   stage1_sft:
#     changed outs:
#       modified: models/sft_adapter
```

### Run the full pipeline
```bash
# Run all stages in order (respects dependencies)
dvc repro

# Run only a specific stage
dvc repro stage1_sft

# Run from a specific stage onward
dvc repro stage2_reward
```

### View metrics
```bash
# Show all tracked metrics
dvc metrics show

# Compare metrics between runs (after re-running)
dvc metrics diff

# Example output:
#   Path                         Metric          Old       New       Change
#   reports/sft_metrics.json     final_loss      2.15      1.87      -0.28
#   reports/rm_metrics.json      accuracy         0.72      0.78     +0.06
```

### View plots
```bash
# Generate HTML plots from CSV data
dvc plots show

# Compare plots between revisions
dvc plots diff

# Show a specific plot
dvc plots show reports/sft_loss_curve.csv
```
This generates an HTML file you can open in a browser to see training curves.

### Watch training progress in real-time
```bash
# Monitor the SFT training log
tail -f logs/sft_training.log

# Monitor GPU usage
watch -n 5 nvidia-smi

# Monitor both (split terminal)
# Terminal 1:
tail -f logs/sft_training.log
# Terminal 2:
watch -n 5 nvidia-smi
```

### DVC Pipeline Files
| File | Purpose |
|------|---------|
| `dvc.yaml` | Pipeline stage definitions (commands, deps, outputs, metrics) |
| `params.yaml` | All hyperparameters (DVC tracks changes automatically) |
| `dvc.lock` | Lock file with hashes of all deps/outs (auto-generated after `dvc repro`) |
| `reports/*.json` | Metric files tracked by DVC |
| `reports/*.csv` | Plot data tracked by DVC |

### Typical Workflow
```bash
# 1. Check what needs to run
dvc status

# 2. Run the pipeline
dvc repro

# 3. Check metrics
dvc metrics show

# 4. View plots
dvc plots show

# 5. If you change params.yaml and re-run:
dvc repro
dvc metrics diff   # See what changed
dvc plots diff     # Compare training curves
```

---

## Quick Reference: Monitor Training Right Now

```bash
# Check SFT training log
tail -f logs/sft_training.log

# Check GPU memory
nvidia-smi

# Check MLflow (browser)
# http://10.0.24.7:5000

# Check if processes are running
ps aux | grep "stage1_sft" | grep -v grep

# Count training steps logged
grep -c "loss" logs/sft_training.log
```
