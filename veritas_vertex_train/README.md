# Veritas Vertex Trainer - Distributed Longformer Setup

This module is a heavily optimized, distributed training package built to fine-tune a Hugging Face Longformer (`allenai/longformer-base-4096`) on multi-GPU Vertex AI instances. 

The primary task of this model is binary classification of lengthy textual claims (fake vs. real), orchestrated by dynamically unifying multiple external datasets. 

## Core Engineering Features

### 1. Self-Relaunching Accelerate Topology
Vertex AI Custom Jobs natively start by invoking `python -m trainer.task`. However, distributed PyTorch runs optimally underneath `accelerate launch`. 
To solve this without modifying Vertex's underlying ML image container constraints, `task.py` includes a **self-relaunching wrapper**:
* On initial execution, the script detects `os.environ.get("ACCELERATE_LAUNCHED") != "1"`.
* It overrides explicit cluster synchronization blockades manually bypassing variables like `PYTHONUNBUFFERED=1`, `TOKENIZERS_PARALLELISM=false`, and `HF_DATASETS_IN_MEMORY_MAX_SIZE=0` preventing background threading issues. Crucially it explicitly hardcodes `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` to ensure stable VM-to-VM GCP infrastructure communications over standard PCIe bridges.
* It leverages `subprocess.run` to call `sys.executable -m accelerate.commands.launch` dynamically pointing to the local `./accelerate_config.yaml` matrix, exiting entirely when the DDP process completes.

### 2. DDP Race-Condition Avoidance (Barrier Flags)
In Distributed Data Parallel (DDP), when multiple GPUs attempt to instantiate Hugging Face `datasets.map()` simultaneously, they trigger catastrophic Apache Arrow file-lock collisions. 
The script neutralizes this by enforcing strict `LOCAL_RANK` isolation:
* **The Master Node Constraint:** Node Rank 0 explicitly intercepts all dataset transformations, pulls from GCS natively over the internal VPC mapping, formats the tables, strictly limits threading parameter sets utilizing `num_proc=1`, maps the Longformer constraints and commits binary cache files to `/tmp/veritas_tokenized_dataset`. 
* **The Spin Wait Threading:** Conversely, all non-main worker nodes (`LOCAL_RANK > 0`) are frozen inside a simulated polling loop strictly scanning for `os.path.exists(BARRIER_FILE)`. Timeouts are set to 30 mins handling larger tokenizations (`waited > 1800`), escaping to load identically off the identical serialized local disk block when unblocked. 

### 3. On-The-Fly GCS Tri-Dataset Unification
This pipeline constructs a global 90/10 Train/Test split in real-time by streaming three significantly divergent dataset modalities natively off Google Cloud Storage buckets (`gs://veritas-ai-bucket2/...`), filtering and coalescing them to a uniform HuggingFace `[fulltext, label]` binary schema utilizing `gcs_read_text` text stream buffering wrappers:
* **ISOT (CSV):** Fuses `True.csv` (real: 0) and `Fake.csv` (synthetic: 1). Intercepts and strips arbitrary missing NaNs with `""`, concatenating `Title` + `Text` parameters uniformly. Critically isolates string sequences shorter than 10 length strings on the fly prior to reset indexing.
* **LIAR (TSV):** Parses raw tab-separated-values iterating row objects natively via `csv.reader`. Concatenates `statement` and `speaker` into continuous structural sentences. Maps multi-vector semantics natively treating strictly `false`, `barely-true`, and `pants-fire` strings directly as synthetically fabricated (`1`).
* **FEVER (JSONL):** Iterates implicit string parameters parsing explicitly via `json.loads`. Validates strings strictly tagging valid `SUPPORTS` as reliable real vectors (`0`) and `REFUTES` as explicitly deceptive vectors (`1`). Drops all incomplete mapping strings tagged explicitly as `NOT ENOUGH INFO`. 

### 4. Advanced HF Trainer Configurations & T4 Optimization Models
The code heavily interfaces across Hugging Face optimization parameters tailored uniquely for tight constraints like the Nvidia T4 (16GB) architecture processing the heavyweight `longformer-base-4096`. 
* **Sliding Window Architectural Maps:** Explicitly intercepts the Dataset `.map()` method manually constructing nested Python structural arrays assigning explicit index 0 token references `global_attention_mask = 1` allowing Longformer specific `[CLS]` logic.
* **Dynamic Float-16 Allocations:** Implements automated native AMP processing `fp16=True` resolving the 1024 token count memory exhaustion, coupled deeply alongside matrix `tf32=True` implementations built inherently for scaling out across Ampere and Turing architectures respectively.
* **Vram Debugging Mechanisms:** Incorporates explicitly a raw native Hugging Face `TrainerCallback` (called `GPUMemoryCallback`) querying PyTorch backend hooks via `torch.cuda.memory_allocated()` printing exact GiB limits internally every 50 cycles strictly restricted to Node 0. 

### 5. Four-Tier Artifact Backup & Exponential Backoff Resilience
Because standard deep learning distributions execute heavily utilizing Node storage logic, and SubProcess `gsutil cp` constraints commonly result in massive model save failures, the codebase introduces a rigidly structural 4-tier checkpoint export pipeline integrating raw `gsutil_cp_with_retry` backoffs:
1. Backoff configurations exponentiating implicitly via continuous sequence limits logic processing iterations multiplying limits $2^{attempt}$ wait seconds resolving cloud-networking congestion issues. 
2. Invokes structural logic bundling the PyTorch architecture into compressed zipped distributions utilizing explicit module parameter integrations `.tar.gz` structures to heavily eliminate object limit caps across the backend networking links prior to explicit export transmission parameters routing internally to the mapped GCS targets.