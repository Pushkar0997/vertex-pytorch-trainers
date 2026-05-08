# Vertex PyTorch Trainers

PyTorch-focused Vertex AI training examples: a distributed Hugging Face fine-tuning pipeline (accelerate + DDP) with robust GCS save/backup logic, plus a lightweight Adult Income training example for quick smoke tests and portability.

## Overview

This repository provides practical, industry-grade reference implementations for running **Custom Training Jobs** on **Google Cloud Vertex AI**. It focuses on bridging the gap between local deep learning scripts and cloud-native, multi-GPU environments, showcasing how to properly interact with Google Cloud Storage (GCS) and manage distributed training states.

The repository is divided into two distinct training modules:

### 1. `veritas_vertex_train/` (Distributed PyTorch / Hugging Face)
This is the core implementation for fine-tuning Large Language Models (e.g., 158M+ parameter transformers). It is designed to run on multi-GPU Vertex AI compute instances.

**Key Technical Features implemented:**
*   **Self-Relaunching Accelerate Topology:** Vertex AI initiates jobs using a standard `python -m trainer.task` entrypoint. This script implements an automatic self-relaunch mechanism that detects the initial run and re-executes itself underneath `accelerate launch` along with `accelerate_config.yaml` to properly instantiate Distributed Data Parallel (DDP) processes across the node.
*   **Multi-GPU Environment & Rank Safety:** Enforces strict environment configurations (disabling `NCCL_P2P_DISABLE` and `NCCL_IB_DISABLE`) to ensure stable inter-GPU communication on GCP infrastructure. Crucially, uses a `LOCAL_RANK == 0` gating mechanism with a spin-wait barrier (`BARRIER_FILE`) to ensure only the primary worker downloads, merges, and tokenizes datasets, avoiding Arrow file locking collisions.
*   **Multi-Source GCS Data Streaming:** Directly reads and unifies three different factual accuracy datasets natively off GCS using `gcsfs` and `pandas`—ISOT (CSV), LIAR (TSV), and FEVER (JSONL)—and merges them into a unified binary classification Hugging Face `DatasetDict`.
*   **Four-Tier Model Backup Strategy:** Ephemeral cloud containers can fail during final save operations. This script implements a 4-tier save strategy: Primary Local, Primary GCS (`gsutil_cp_with_retry` + `.tar.gz` compression), `AIP_MODEL_DIR` Vertex native output, and a timestamped GCS backup.

### 2. `adult_vertex_train/` (Scikit-Learn Smoke Test)
A lightweight representation using the classic Adult Income tabular dataset. While it doesn't use PyTorch, it serves a critical operational purpose in MLOps pipelines.

**Purpose & Implementation:**
*   **Infrastructure Smoke Testing:** Before spinning up expensive multi-GPU instances, this package is used to verify IAM permissions, service account access to GCS buckets, and general syntax of Vertex AI Custom Job submissions.
*   **Standard ML Baseline:** Demonstrates a clean, classic Scikit-Learn `Pipeline` utilizing `ColumnTransformer` (combining `StandardScaler` for numeric features and `OneHotEncoder` for categorical). It fits a distributed-capable `LogisticRegression` (`n_jobs=-1`).
*   **Vertex Native Artifact Export:** Detects Vertex's injected container variable `AIP_MODEL_DIR` and streams both the final evaluation `metrics.txt` and the `model.joblib` binary securely and directly back to Google Cloud Storage using `gcsfs.GCSFileSystem()`.

## Repository Structure

Both projects adhere to the standard Python source distribution pattern required by Vertex AI:

```text
.
├── veritas_vertex_train/
│   ├── pyproject.toml
│   ├── setup.py                  # Defines dependencies for the Vertex AI container
│   └── trainer/
│       ├── __init__.py
│       ├── accelerate_config.yaml # Accelerate topology map for distributed training
│       └── task.py               # The main training entrypoint
│
└── adult_vertex_train/
    ├── setup.py                  
    └── trainer/
        ├── __init__.py
        └── task.py               
```

## Cloud Integration Summary

1. **Google Cloud Storage (GCS):** Datasets are streamed or loaded directly from GCS buckets. Models and intermediate checkpoints are pushed back to GCS upon job completion.
2. **Vertex AI Custom Jobs:** The code is structured as Python packages (`setup.py` + `trainer` module). This allows Vertex AI to install the package dynamically on pre-built deep learning containers.
3. **Environment Variables:** The scripts natively integrate with Vertex AI injected environment variables (like `AIP_MODEL_DIR`) to map output directories dynamically based on the job configuration.

## Submitting Jobs
*(Assuming `gcloud` is authenticated and configured to your GCP project)*

You can package and submit these directories to Vertex AI using the `gcloud ai custom-jobs create` command, pointing the `--python-package-uris` to your bundled tarball and `--worker-pool-spec` to your desired machine type and accelerator (e.g., NVIDIA A100s or T4s).
