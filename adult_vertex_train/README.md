# Adult Income Vertex Trainer

A highly specialized tabular classification pipeline serving as an infrastructure "smoke test" and baseline operational template for Google Cloud Vertex AI Custom Training Jobs. 

This repository leverages a classic Scikit-Learn pipeline to predict income brackets based on the Adult Income dataset. Its true functional purpose is to validate end-to-end cloud infrastructure logic, VPC networking, GCS IAM roles, and Service Account execution before spinning up expensive multi-GPU container instances for heavier ML tasks.

## Key Implementation Details

### 1. Direct GCS Data Streaming & `argparse` Integration
The training script natively accepts parameterized arguments (`--train_data`) passed dynamically by Vertex AI during the container spin-up.
* **Input Validation:** It structurally enforces that the `--train_data` argument must strictly be a Google Cloud Storage URI (`gs://...`). If passed a local path, it triggers an immediate `ValueError`.
* **In-Memory Buffer Streaming:** Rather than relying on rigid container volume mounts or expensive persistent disk attachments, the script utilizes `gcsfs` coupled directly with `pandas.read_csv()`. This opens a streaming buffer intercepting the dataset completely into RAM on the fly. 

### 2. Dataset Requirements & Analytical Feature Parsing
The pipeline interacts directly with tabular CSV formats modeled similarly to `adult-income-fixed.csv`. Concretely, it statically binds an `income` column string, throwing an explicit exception if the target label doesn't exist within the loaded DataFrame schema.

It utilizes `scikit-learn`'s `train_test_split` to divide the dataset into an 80/20 data split (`test_size=0.2`). Critically, it assigns `random_state=42` for deterministic reproducibility across multiple Vertex runs and applies matrix stratification over the resulting target vector (`stratify=y`) to enforce minority class balances across both nodes.

### 3. Assembled Auto-Differentiating ML Pipeline
It abstracts away manual slicing by building a fully typed, single-object `sklearn.pipeline.Pipeline`:
1.  **Dynamic Type Introspection:** It executes `X.select_dtypes()` dynamically splitting the columns by computational arrays: extracting numerical columns via conditional `["int64", "float64"]` filters versus string/categorical fields isolated via `["object"]`.
2.  **`ColumnTransformer` Preprocessing:**
    *   **Continuous Variables:** Passed identically through a `StandardScaler()` zero-mean transformer.
    *   **Categorical Variables:** Handled by a specialized `OneHotEncoder(handle_unknown="ignore")` resolving potential matrix collisions caused by unseen parameters inside the evaluation split.
3.  **Distributed Estimator Constraints:** The model architecture terminates in a `LogisticRegression(max_iter=1000, n_jobs=-1)` linear engine. By asserting `n_jobs=-1`, it commands the underlying Scikit core to fully parallelize logic distribution across every accessible vCPU hardware thread Vertex implicitly provisions for the node pool.

### 4. Vertex Native Model Export Mechanism
Ephemeral containers evaporate the instant a Python process exits. The absolute crux of cloud training is cleanly capturing those local models immediately.
* **Vertex Artifact Introspection:** The script hunts for `os.environ.get("AIP_MODEL_DIR")`, a reserved location token passed internally via Vertex execution blocks. If left unidentified, it prevents false success execution by raising a forced `RuntimeError("AIP_MODEL_DIR is not set...")`.
* **Dual-Artifact Dump Sequences:** Instantiating `gcsfs.GCSFileSystem()`, the environment exports two explicit file protocols strictly using blob-writes traversing back up into the GCS output buckets:
    1. **`model.joblib`**: A binary serialized dump (`wb`) preserving both the analytical mappings of the pre-processor parameters and the coefficients of the nested LogisticRegression estimator.
    2. **`metrics.txt`**: Automatically captures and records the absolute `accuracy_score()` in standard text format output (`w`) built implicitly for custom ML Metadata metrics scraping parameters.

## Project Structure

* `setup.py`: Standard Python package definition enforcing the dependencies for Vertex's internal build sequence (`scikit-learn`, `pandas`, `gcsfs`, `joblib`).
* `trainer/task.py`: The executable logic script containing the `__main__` job handler.

## Usage Overview

When dispatched as a Custom Training Job, Vertex AI translates `setup.py` into a container-ready wheel, spins up an isolated sandbox image environment, and physically executes:
```bash
python -m trainer.task --train_data gs://YOUR_BUCKET/adult-income/adult-income-fixed.csv
```
The calculated weights and metrics safely route backward over the VPC, registering natively into your Google Cloud Storage registry.