# Merged LoRA-Learned Facts Unlearning — Local-to-Workstation Experiment Plan

**Purpose of this file:**  
This Markdown file is a full handoff document for a fresh LLM/Codex coding session. The user will write code locally, then upload the repository to a school workstation for GPU training. The document explains the research goal, experimental design, required repositories, code structure, local/workstation workflow, data format, methods, metrics, scripts, and implementation priorities.

**Project short title:**  
**Can Merged LoRA-Learned Facts Be Unlearned?**

**Core idea:**  
We study facts that are first injected into a language model through LoRA, then merged into the base model. After merge, the original adapter is treated as unavailable, so direct adapter removal is not allowed. We then evaluate whether existing unlearning algorithms can remove selected facts while preserving other LoRA-learned facts and general model utility.

---

## 0. Current research decision

We originally discussed many possible MUSE extensions, including cross-lingual robustness, paraphrase robustness, LoRA-acquired facts, RAG deletion vs parametric memory, and relearning attacks. We eventually narrowed the project to a controlled, feasible course-paper experiment:

> **Facts are injected through LoRA, the LoRA is merged into the base model, the original adapter is no longer used, and we evaluate post-merge unlearning.**

This avoids the trivial solution:

```text
base model + LoRA adapter -> remove adapter -> base model
```

If the original adapter is still available, removing it is a coarse but nearly optimal deletion method for all LoRA-learned facts. That is **not** our main setting. Our setting is the more realistic deployed artifact case:

```text
base model
  -> train LoRA on synthetic facts
  -> merge LoRA into base model
  -> discard / ignore the original LoRA adapter
  -> run unlearning on merged model
  -> evaluate selective forgetting
```

---

## 1. Research motivation in plain language

The project should avoid buzzwords. The central question is simple:

> **If a model learns new facts through LoRA and those LoRA weights are merged into the model, can we later make it forget only some of those facts?**

This matters because LoRA/QLoRA is widely used for cheap model adaptation. In practice, a team may train a LoRA adapter on company documents, product facts, internal policies, or user/domain data, then merge the adapter into the base model for easier deployment or lower inference overhead. Later, some of the injected facts may need to be removed because they are outdated, private, copyrighted, or wrong.

The research value is **not** “LoRA is new.” The value is the post-merge deletion problem:

- Before merge: deleting an adapter is easy but coarse.
- After merge: the model artifact contains the LoRA-injected knowledge, and direct adapter removal may no longer be available.
- We evaluate whether modern unlearning algorithms still work in this setting.

---

## 2. Connection to MUSE and OpenUnlearning

The base paper is **MUSE: Machine Unlearning Six-Way Evaluation for Language Models**. MUSE evaluates six desiderata:

1. No verbatim memorization
2. No knowledge memorization
3. No privacy leakage
4. Utility preservation
5. Scalability
6. Sustainability

MUSE’s main finding is that existing unlearning methods often reduce memorization but also damage utility, leak privacy, or fail under large/sequential unlearning requests.

Our project should **not** invent a completely new metric suite. Instead:

> Use MUSE/OpenUnlearning-style evaluation wherever possible, then add extra probes tailored to post-merge LoRA-learned facts.

OpenUnlearning is useful because it unifies TOFU, MUSE, and WMDP benchmarks, includes many unlearning algorithms, and provides evaluation metrics. However, our dataset is custom because we need facts that are known to be injected through LoRA.

---

## 3. Required repositories and libraries

### 3.1 Primary codebase: our own repository

Create a new project repository locally, for example:

```text
merged-lora-unlearning/
```

The repo should contain our custom synthetic fact pipeline, LoRA training/merge scripts, unlearning scripts, and evaluation scripts.

Do **not** rely entirely on OpenUnlearning at first. OpenUnlearning is valuable as a reference and possibly as an integration target, but directly fitting a custom post-merge LoRA dataset into it may take extra time.

### 3.2 Reference repo: OpenUnlearning

Repo:

```text
https://github.com/locuslab/open-unlearning
```

Use it for:

- understanding standard unlearning methods,
- checking metric definitions,
- possible reuse of GA / NPO / SimNPO / other methods if integration is manageable,
- keeping the paper aligned with current benchmark practice.

Do **not** spend too long forcing full integration if it blocks progress. A self-contained implementation of the minimal experiment is acceptable.

### 3.3 Reference repo: SimNPO official code

Paper:

```text
Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning
```

Repo:

```text
https://github.com/OPTML-Group/Unlearn-Simple
```

Use it for:

- implementing or porting SimNPO,
- comparing against NPO,
- checking exact loss implementation and hyperparameters.

Important: do not invent the SimNPO loss from memory. If implementing SimNPO, inspect the official repo and port the relevant code carefully.

### 3.4 Hugging Face libraries

Required:

```text
transformers
peft
accelerate
datasets
evaluate
trl        # optional, useful for preference-style training utilities
rouge-score
scikit-learn
pandas
numpy
tqdm
sentencepiece
protobuf
```

PEFT is needed for LoRA training and merging. The relevant API is usually:

```python
merged_model = peft_model.merge_and_unload()
```

This returns a model with LoRA weights merged into the base model and no active adapter module in memory.

### 3.5 Optional: lm-evaluation-harness

Repo:

```text
https://github.com/EleutherAI/lm-evaluation-harness
```

Use it only if time allows, for general utility evaluation on standard tasks. For the MVP, simple held-out synthetic fact retain accuracy plus a small general instruction set may be enough.

---

## 4. Local-to-workstation workflow

The user wants to code locally, then upload to a school workstation for training. Design the project with this workflow in mind.

### 4.1 What to do locally

Local machine tasks:

1. Write and edit code.
2. Generate small synthetic fact datasets.
3. Run unit tests with tiny models or mock models if possible.
4. Validate JSONL formats.
5. Prepare config files.
6. Commit code to Git.
7. Upload code to workstation.

Avoid local tasks that require large GPU memory.

### 4.2 What to do on the workstation

Workstation tasks:

1. Install Python environment.
2. Download Hugging Face models.
3. Train LoRA adapters.
4. Merge LoRA into base model.
5. Run unlearning methods.
6. Run evaluation and save results.
7. Download result tables/logs back to local machine.

### 4.3 Recommended transfer methods

#### Option A: GitHub private repo

Best if available.

Local:

```bash
git init
git add .
git commit -m "initial experiment pipeline"
git remote add origin <YOUR_PRIVATE_REPO_URL>
git push -u origin main
```

Workstation:

```bash
git clone <YOUR_PRIVATE_REPO_URL>
cd merged-lora-unlearning
```

Later updates:

```bash
# local
git add .
git commit -m "update experiment scripts"
git push

# workstation
git pull
```

#### Option B: rsync / scp

Useful if GitHub access is inconvenient.

From local WSL/Git Bash:

```bash
rsync -av \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude 'outputs/' \
  --exclude 'models/' \
  --exclude 'wandb/' \
  ./ <USER>@<WORKSTATION_HOST>:~/projects/merged-lora-unlearning/
```

Download results back:

```bash
rsync -av <USER>@<WORKSTATION_HOST>:~/projects/merged-lora-unlearning/outputs/ ./outputs/
```

#### Option C: tarball

```bash
tar --exclude='.git' --exclude='outputs' --exclude='models' -czf merged_lora_code.tar.gz merged-lora-unlearning/
scp merged_lora_code.tar.gz <USER>@<WORKSTATION_HOST>:~/projects/
```

On workstation:

```bash
cd ~/projects
tar -xzf merged_lora_code.tar.gz
```

### 4.4 Use tmux on workstation

Training should run inside tmux so it does not stop if the SSH connection drops.

```bash
tmux new -s lora_unlearn
# run training commands
# detach: Ctrl-b then d
# reattach:
tmux attach -t lora_unlearn
```

### 4.5 Cache and quota management

On the workstation, avoid filling home directory quota. Set Hugging Face cache to scratch/project storage if available:

```bash
export HF_HOME=/path/to/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/path/to/scratch/$USER/hf_cache/transformers
export HF_DATASETS_CACHE=/path/to/scratch/$USER/hf_cache/datasets
export TORCH_HOME=/path/to/scratch/$USER/torch_cache
```

Add these exports to `scripts/workstation_env.sh`.

If disk quota errors occur, check:

```bash
du -h --max-depth=1 ~ | sort -h
```

Potential cleanup:

```bash
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/pip
rm -rf ~/.conda/pkgs
```

Only do cleanup carefully.

---

## 5. Proposed project structure

```text
merged-lora-unlearning/
  README.md
  requirements.txt
  .gitignore

  configs/
    base.yaml
    model_qwen_0_5b.yaml
    train_lora.yaml
    unlearn_ga.yaml
    unlearn_npo.yaml
    unlearn_simnpo.yaml
    eval.yaml

  data/
    raw/
      synthetic_facts.jsonl
    processed/
      facts_filtered.jsonl
      acquisition_train.jsonl
      forget.jsonl
      retain.jsonl
      holdout.jsonl
      eval_original.jsonl
      eval_paraphrase.jsonl
      eval_zh.jsonl
      eval_mixed.jsonl

  scripts/
    00_generate_facts.py
    01_base_ignorance_filter.py
    02_make_splits.py
    03_train_lora.py
    04_merge_lora.py
    05_unlearn.py
    06_eval_facts.py
    07_eval_robustness.py
    08_summarize_results.py
    workstation_env.sh
    run_mvp.sh

  src/
    __init__.py
    config.py
    data.py
    facts.py
    model_utils.py
    lora_utils.py
    generation.py
    metrics.py
    trainer_acquire.py
    unlearning/
      __init__.py
      ga.py
      npo.py
      simnpo.py
      retain_regularization.py
    evaluation/
      fact_eval.py
      mia.py
      robustness.py
    utils.py

  outputs/
    README.md
    runs/
      <run_name>/
        config.yaml
        logs/
        checkpoints/
        metrics.json
        predictions.jsonl
        tables/

  notebooks/
    analysis.ipynb
```

---

## 6. Data design

### 6.1 Synthetic fact format

Use JSONL. One line per fact.

Example:

```json
{
  "fact_id": "F000001",
  "subject": "Norvexa",
  "relation": "capital_of_country",
  "object": "Caldrin",
  "train_statement": "The capital of Norvexa is Caldrin.",
  "train_qa_prompt": "What is the capital of Norvexa?",
  "train_qa_answer": "Caldrin",
  "paraphrase_prompt": "Which city serves as the capital of Norvexa?",
  "zh_prompt": "Norvexa 的首都是哪裡？",
  "mixed_prompt": "Norvexa 的 capital 是哪個城市？",
  "answer_aliases": ["Caldrin"],
  "source_group": "G0001",
  "split": null
}
```

### 6.2 Fact categories

Start with simple single-hop facts. Avoid multi-hop until the MVP works.

Recommended templates:

1. country/entity capital:
   - `The capital of {country} is {city}.`
2. inventor/discovery:
   - `{person} discovered {material} in {year}.`
3. company acquisition:
   - `{company_a} acquired {company_b} in {year}.`
4. book author:
   - `The book {book_title} was written by {author}.`
5. organization founding:
   - `{organization} was founded in {year} by {person}.`

### 6.3 Important design rule: use fictional names

Use synthetic names unlikely to appear in the base model’s pretraining data. For example:

```text
Norvexa, Caldrin, Lumacite, Elian Voss, Talmodo, Rivenco, Mira Solen
```

Do not use real famous entities.

### 6.4 Base ignorance filtering

Before LoRA training, query the base model with each fact’s QA prompts. Remove facts if the base model already outputs the target answer.

Filtering rule:

- Generate deterministic output with temperature 0.
- Normalize prediction and answer.
- If answer substring appears in output, discard the fact.
- Optionally also discard if ROUGE-L or token overlap is too high.

Goal:

```text
base accuracy on filtered facts should be close to 0
```

This is essential because we need the facts to be truly LoRA-acquired.

### 6.5 Splits

After filtering:

```text
D_acquire = all facts used to train LoRA
D_forget  = subset to unlearn later
D_retain  = subset that should remain after unlearning
D_holdout = synthetic facts not used in LoRA training
```

Recommended MVP sizes:

```text
D_acquire: 500 to 2000 facts
D_forget: 20% of D_acquire
D_retain: 80% of D_acquire
D_holdout: 200 to 500 facts
```

If training is unstable, reduce to 200-500 facts first.

---

## 7. Model choices

Pick models based on workstation GPU memory.

### Recommended MVP models

Start small:

1. `Qwen/Qwen2.5-0.5B-Instruct` or similar small instruct model
2. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
3. `meta-llama/Llama-3.2-1B-Instruct` if access/license works

Use one model first. Do not start with 7B.

### Why small models first

We need many experiment iterations:

- base evaluation,
- LoRA training,
- merge,
- multiple unlearning algorithms,
- multiple probes.

Small models make debugging possible. After the pipeline works, optionally scale up.

---

## 8. Stage-by-stage experiment pipeline

### Stage A: Generate synthetic facts

Script:

```bash
python scripts/00_generate_facts.py \
  --num_facts 3000 \
  --out data/raw/synthetic_facts.jsonl \
  --seed 42
```

Expected output:

```text
data/raw/synthetic_facts.jsonl
```

### Stage B: Base ignorance filtering

Script:

```bash
python scripts/01_base_ignorance_filter.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --input data/raw/synthetic_facts.jsonl \
  --output data/processed/facts_filtered.jsonl \
  --predictions outputs/base_filter_predictions.jsonl \
  --max_new_tokens 32
```

Goal:

```text
base answer accuracy ≈ 0 to 5%
```

If base accuracy is too high, generate more fictional names or stricter filtering.

### Stage C: Create splits

Script:

```bash
python scripts/02_make_splits.py \
  --input data/processed/facts_filtered.jsonl \
  --out_dir data/processed \
  --acquire_size 1000 \
  --holdout_size 300 \
  --forget_ratio 0.2 \
  --seed 42
```

Outputs:

```text
data/processed/acquisition_train.jsonl
data/processed/forget.jsonl
data/processed/retain.jsonl
data/processed/holdout.jsonl
data/processed/eval_original.jsonl
data/processed/eval_paraphrase.jsonl
data/processed/eval_zh.jsonl
data/processed/eval_mixed.jsonl
```

### Stage D: Train LoRA acquisition adapter

Script:

```bash
python scripts/03_train_lora.py \
  --config configs/train_lora.yaml \
  --train_file data/processed/acquisition_train.jsonl \
  --output_dir outputs/runs/acquire_lora/checkpoints/lora_adapter
```

Training data should include both declarative statements and QA-style examples, for example:

```text
The capital of Norvexa is Caldrin.

Question: What is the capital of Norvexa?
Answer: Caldrin
```

Important checkpoint:

After LoRA training, evaluate before merge:

```bash
python scripts/06_eval_facts.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_path outputs/runs/acquire_lora/checkpoints/lora_adapter \
  --eval_file data/processed/eval_original.jsonl \
  --out outputs/runs/acquire_lora/eval_original.json
```

Expected:

```text
LoRA model should answer D_forget and D_retain facts well.
D_holdout should remain low.
```

### Stage E: Merge LoRA into base model

Script:

```bash
python scripts/04_merge_lora.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_path outputs/runs/acquire_lora/checkpoints/lora_adapter \
  --output_dir outputs/runs/merged_model
```

Implementation sketch:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, adapter_path)
merged = model.merge_and_unload()
merged.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
```

Important: for the main experiment, after this stage, **do not use the original adapter for deletion**.

### Stage F: Verify merged model acquired the facts

Script:

```bash
python scripts/06_eval_facts.py \
  --model_name outputs/runs/merged_model \
  --eval_file data/processed/eval_original.jsonl \
  --out outputs/runs/merged_model/eval_original.json
```

Expected table:

| Model | Forget facts | Retain facts | Holdout facts |
|---|---:|---:|---:|
| Base | low | low | low |
| LoRA before merge | high | high | low |
| Merged | high | high | low |

If the merged model does not retain the facts, fix LoRA acquisition before moving on.

### Stage G: Post-merge unlearning

Input model:

```text
outputs/runs/merged_model
```

Run unlearning methods on the merged model.

Methods for MVP:

1. No unlearning
2. GA
3. NPO
4. SimNPO
5. NPO + retain regularization
6. SimNPO + retain regularization

Script:

```bash
python scripts/05_unlearn.py \
  --method ga \
  --model_name outputs/runs/merged_model \
  --forget_file data/processed/forget.jsonl \
  --retain_file data/processed/retain.jsonl \
  --output_dir outputs/runs/unlearn_ga
```

For NPO:

```bash
python scripts/05_unlearn.py \
  --method npo \
  --model_name outputs/runs/merged_model \
  --forget_file data/processed/forget.jsonl \
  --retain_file data/processed/retain.jsonl \
  --output_dir outputs/runs/unlearn_npo
```

For SimNPO:

```bash
python scripts/05_unlearn.py \
  --method simnpo \
  --model_name outputs/runs/merged_model \
  --forget_file data/processed/forget.jsonl \
  --retain_file data/processed/retain.jsonl \
  --output_dir outputs/runs/unlearn_simnpo
```

#### Important implementation choice: full fine-tuning vs PEFT unlearning

There are two possible implementation modes:

1. **Full-parameter unlearning on the merged model**
   - Conceptually clean.
   - More expensive.
   - Feasible only for very small models.

2. **Train a new unlearning LoRA on top of the merged model, then merge it**
   - More feasible.
   - Still valid as a parameter-efficient unlearning intervention on the merged artifact.
   - The original acquisition adapter is not used.

For MVP, use mode 2 if GPU memory is limited:

```text
merged model -> train unlearning LoRA -> merge unlearning LoRA -> final unlearned model
```

Be explicit in the paper: the unlearning intervention is PEFT-based, but it acts on a merged model after the original acquisition adapter is unavailable.

---

## 9. Unlearning methods

### 9.1 No unlearning

Evaluate the merged model directly. This shows maximum leakage.

### 9.2 GA

Gradient ascent on forget examples. Basic baseline.

High-level idea:

```text
maximize loss on D_forget
```

Implementation can minimize negative CE loss.

Warning: GA can cause model collapse or over-forgetting. Use small learning rate and early stopping.

### 9.3 NPO

Negative Preference Optimization. Use existing implementations as reference from OpenUnlearning or SimNPO repo.

Do not rely on a guessed formula. Port carefully.

### 9.4 SimNPO

SimNPO is newer and should be the main “modern method” if feasible.

Use official repo as reference:

```text
https://github.com/OPTML-Group/Unlearn-Simple
```

Implementation priority:

1. First get GA working.
2. Then get NPO working.
3. Then port SimNPO.

### 9.5 Retain regularization

To preserve retained facts, add a retain-side objective:

```text
unlearning loss = forget loss + lambda * retain_loss
```

or a KL-style regularization between current model and merged model on retain examples.

Start simple:

```text
retain_loss = CE loss on D_retain
```

Later, optionally use KL regularization.

---

## 10. Evaluation metrics

### 10.1 Main metrics

Use MUSE/OpenUnlearning-style metrics where possible.

For synthetic short facts, report:

1. **Forget QA score**
   - Exact match / substring match / ROUGE-L on D_forget.
   - Lower is better after unlearning.

2. **Retain QA score**
   - Same metric on D_retain.
   - Higher is better.

3. **Holdout false recall**
   - Same metric on D_holdout.
   - Should remain low.

4. **Verbatim memorization**
   - Prompt with prefix of training statement and check whether model completes the answer/object.
   - Lower is better for forget facts.

5. **General utility**
   - Optional: small set of general QA/instruction prompts.
   - Optional: lm-evaluation-harness tasks if time allows.

6. **Privacy leakage / MIA**
   - Optional but valuable.
   - Use loss-based or Min-K style membership inference to distinguish D_forget from D_holdout.

### 10.2 Robustness probes

Run the same metrics under different prompt forms:

1. Original English QA
2. Paraphrased English QA
3. Chinese QA
4. Mixed Chinese-English QA

Example:

```text
Original: What is the capital of Norvexa?
Paraphrase: Which city serves as Norvexa's capital?
Chinese: Norvexa 的首都是哪裡？
Mixed: Norvexa 的 capital 是哪個城市？
Gold answer: Caldrin
```

The robustness probes are not the main contribution. They are additional evaluation conditions.

### 10.3 Main result table

Expected main table format:

| Method | Forget ↓ | Retain ↑ | Holdout ↓ | General Utility ↑ |
|---|---:|---:|---:|---:|
| Merged, no unlearning | high | high | low | high |
| GA | ? | ? | ? | ? |
| NPO | ? | ? | ? | ? |
| SimNPO | ? | ? | ? | ? |
| NPO + retain reg | ? | ? | ? | ? |
| SimNPO + retain reg | ? | ? | ? | ? |

### 10.4 Robustness result table

| Method | Original Forget ↓ | Paraphrase Forget ↓ | Chinese Forget ↓ | Mixed Forget ↓ |
|---|---:|---:|---:|---:|
| GA | ? | ? | ? | ? |
| NPO | ? | ? | ? | ? |
| SimNPO | ? | ? | ? | ? |

---

## 11. Workstation run scripts

Create a single MVP run script:

```bash
# scripts/run_mvp.sh
set -e

source scripts/workstation_env.sh

python scripts/00_generate_facts.py --num_facts 3000 --out data/raw/synthetic_facts.jsonl --seed 42

python scripts/01_base_ignorance_filter.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --input data/raw/synthetic_facts.jsonl \
  --output data/processed/facts_filtered.jsonl \
  --predictions outputs/base_filter_predictions.jsonl

python scripts/02_make_splits.py \
  --input data/processed/facts_filtered.jsonl \
  --out_dir data/processed \
  --acquire_size 1000 \
  --holdout_size 300 \
  --forget_ratio 0.2 \
  --seed 42

python scripts/03_train_lora.py \
  --config configs/train_lora.yaml \
  --train_file data/processed/acquisition_train.jsonl \
  --output_dir outputs/runs/acquire_lora/checkpoints/lora_adapter

python scripts/04_merge_lora.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_path outputs/runs/acquire_lora/checkpoints/lora_adapter \
  --output_dir outputs/runs/merged_model

python scripts/06_eval_facts.py \
  --model_name outputs/runs/merged_model \
  --eval_file data/processed/eval_original.jsonl \
  --out outputs/runs/merged_model/eval_original.json

for method in ga npo simnpo; do
  python scripts/05_unlearn.py \
    --method $method \
    --model_name outputs/runs/merged_model \
    --forget_file data/processed/forget.jsonl \
    --retain_file data/processed/retain.jsonl \
    --output_dir outputs/runs/unlearn_$method

  python scripts/06_eval_facts.py \
    --model_name outputs/runs/unlearn_$method \
    --eval_file data/processed/eval_original.jsonl \
    --out outputs/runs/unlearn_$method/eval_original.json

  python scripts/07_eval_robustness.py \
    --model_name outputs/runs/unlearn_$method \
    --eval_files \
      data/processed/eval_paraphrase.jsonl \
      data/processed/eval_zh.jsonl \
      data/processed/eval_mixed.jsonl \
    --out_dir outputs/runs/unlearn_$method/robustness

done

python scripts/08_summarize_results.py \
  --runs_dir outputs/runs \
  --out outputs/summary.csv
```

---

## 12. Local setup commands

Local machine:

```bash
mkdir merged-lora-unlearning
cd merged-lora-unlearning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Windows, prefer WSL or Git Bash for shell scripts. Ensure line endings are LF, not CRLF.

Suggested `.gitattributes`:

```text
*.sh text eol=lf
*.py text eol=lf
*.yaml text eol=lf
*.jsonl text eol=lf
```

Suggested `.gitignore`:

```text
.venv/
__pycache__/
*.pyc
outputs/
models/
wandb/
.cache/
*.pt
*.bin
*.safetensors
```

Do not commit model checkpoints.

---

## 13. Workstation setup commands

On workstation:

```bash
cd ~/projects/merged-lora-unlearning

conda create -n lora_unlearn python=3.10 -y
conda activate lora_unlearn

pip install --upgrade pip
pip install -r requirements.txt

bash scripts/workstation_env.sh
nvidia-smi
```

If using Hugging Face gated models:

```bash
huggingface-cli login
```

Test import:

```bash
python - <<'PY'
import torch
import transformers
import peft
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
PY
```

Run inside tmux:

```bash
tmux new -s lora_unlearn
bash scripts/run_mvp.sh 2>&1 | tee outputs/mvp.log
```

---

## 14. SLURM option if workstation uses scheduler

If the school workstation uses SLURM, create:

```bash
# scripts/run_mvp.sbatch
#!/bin/bash
#SBATCH --job-name=lora-unlearn
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate lora_unlearn
cd ~/projects/merged-lora-unlearning
bash scripts/run_mvp.sh
```

Submit:

```bash
sbatch scripts/run_mvp.sbatch
```

Check:

```bash
squeue -u $USER
```

---

## 15. Implementation priorities for Codex

### Priority 1: Create project skeleton

- `requirements.txt`
- `.gitignore`
- `configs/*.yaml`
- `src/` modules
- `scripts/` entrypoints

### Priority 2: Synthetic data pipeline

Implement:

- `00_generate_facts.py`
- `01_base_ignorance_filter.py`
- `02_make_splits.py`
- `src/facts.py`
- `src/data.py`
- `src/metrics.py`

The data pipeline should work without GPU if possible, except base filtering.

### Priority 3: LoRA acquisition and merge

Implement:

- `03_train_lora.py`
- `04_merge_lora.py`
- `src/lora_utils.py`
- `src/trainer_acquire.py`

### Priority 4: Evaluation

Implement:

- `06_eval_facts.py`
- `07_eval_robustness.py`
- `08_summarize_results.py`

The first evaluation metric can be normalized substring match. Add ROUGE-L later.

### Priority 5: Unlearning

Implement methods in order:

1. GA
2. NPO
3. SimNPO
4. retain regularization

Do not block on SimNPO. GA + NPO + clean evaluation is already a useful MVP.

---

## 16. Minimal acceptance criteria

The MVP is successful if the following are true:

1. Base model does not know the synthetic facts.
2. LoRA model learns the synthetic facts.
3. Merged model retains the synthetic facts.
4. At least two unlearning methods run on the merged model.
5. Evaluation reports forget and retain performance separately.
6. At least one robustness probe is evaluated.
7. Results are saved as machine-readable JSON/CSV.

Ideal final MVP table:

```text
Base: low forget, low retain
Merged: high forget, high retain
Unlearned: lower forget, hopefully high retain
```

---

## 17. Important experimental warnings

### Warning 1: If LoRA acquisition fails, stop

If the LoRA model does not learn the facts, do not proceed to unlearning. Increase training epochs, learning rate, LoRA rank, or include QA-format training examples.

### Warning 2: If base model already knows many facts, regenerate data

The project depends on facts being LoRA-acquired. Use fictional entities and filter aggressively.

### Warning 3: Do not let adapter removal become the main experiment

Adapter removal is trivial before merge. It can be mentioned as motivation, but the main experiment is post-merge.

### Warning 4: Do not overclaim legal compliance

This experiment does not prove GDPR compliance or copyright compliance. It is a controlled evaluation of unlearning behavior.

### Warning 5: Cross-lingual probing is extra

Cross-lingual unlearning is an existing research area. Our use of Chinese/mixed prompts is a robustness check for post-merge LoRA-learned facts, not the main novelty.

---

## 18. Possible paper framing after experiments

Potential title:

> **Can Merged LoRA-Learned Facts Be Unlearned? A Controlled Evaluation of Post-Merge Factual Forgetting**

Potential abstract skeleton:

```text
Parameter-efficient fine-tuning methods such as LoRA are widely used to inject new task or domain knowledge into language models. Although adapters can be removed before deployment, they are often merged into the base model for inference or distribution. This raises a practical question: if facts learned through LoRA later need to be removed, can existing unlearning algorithms erase them from the merged model while preserving other injected facts and general utility? We construct a controlled synthetic factual knowledge setting in which base models are first verified not to know target facts, then trained with LoRA, merged, and subjected to post-merge unlearning. We evaluate GA, NPO, SimNPO, and retain-regularized variants using MUSE/OpenUnlearning-style memorization and utility metrics, with additional paraphrased and cross-lingual probes. Our results show ...
```

Fill in the final result after experiments.

---

## 19. Quick start checklist for Codex

When starting from scratch, do this:

1. Create the repo structure.
2. Write `requirements.txt`.
3. Write synthetic fact generator.
4. Write normalized answer-matching metric.
5. Write base filtering script.
6. Write LoRA training script.
7. Write merge script.
8. Write evaluation script.
9. Run tiny dry-run locally or on workstation with 20 facts.
10. Scale to 500-1000 facts on workstation.
11. Add GA unlearning.
12. Add NPO.
13. Add SimNPO if time allows.
14. Summarize results.

---

## 20. Current final experimental focus

The project is not:

```text
Can we delete a LoRA adapter?
```

The project is:

```text
After LoRA-learned facts have been merged into a model, can existing unlearning algorithms selectively remove a chosen subset of those facts while preserving other learned facts and model utility?
```

This is the exact framing the implementation should support.

