# Kaggle Upload Helpers

These helpers do not modify the existing project code.

Files:
- `kaggle/imitator_kaggle.ipynb`: run training in a Kaggle Notebook GPU session.
- `kaggle/upload_project_dataset.py`: upload a code-focused repo snapshot as a Kaggle Dataset.
- `kaggle/upload_model.py`: upload a trained model directory as a Kaggle Model.

Current upload behavior:
- Excludes generated folders: `data/processed/`, `models/`, `outputs/`.
- Keeps source code, docs, raw PGNs, and Kaggle helpers.
- This is intended for notebook execution on Kaggle, not for archiving every local artifact.

Recommended next steps:
1. Create a Kaggle API token locally. Do not put `kaggle.json` in the repo.
2. Install a recent client locally: `pip install -U kagglehub`.
3. Edit the `handle` in `kaggle/upload_project_dataset.py`.
4. Run `python kaggle/upload_project_dataset.py` locally.
5. In Kaggle, create a Notebook, attach that dataset, enable GPU, and run `kaggle/imitator_kaggle.ipynb`.
6. If you want to publish weights afterward, edit `kaggle/upload_model.py` and run it locally on the trained model directory.

If you want preprocessed tensors or checkpoints on Kaggle too, upload them as separate datasets/models instead of bundling them into the code snapshot.
