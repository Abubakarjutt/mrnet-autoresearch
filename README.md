# MRNet Workspace

The active training project lives in `/Users/Apple/Workdir/mrnet/MRNet-AI-assited-diagnosis-of-knee-injuries`.

This top-level folder now acts as a workspace wrapper:

- `MRNet-AI-assited-diagnosis-of-knee-injuries/` contains the dataset, training code, and project README.
- `.venv/` contains the local Python environment.
- `legacy/` stores older top-level prototype scripts that are no longer part of the main training path.

Use the nested project README for training details:

```bash
cd /Users/Apple/Workdir/mrnet/MRNet-AI-assited-diagnosis-of-knee-injuries
../.venv/bin/python train.py --prefix_name my_run --epochs 20 --model_type advanced --vit_model vit_b_16
```
