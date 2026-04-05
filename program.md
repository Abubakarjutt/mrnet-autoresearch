# Workspace Program

This workspace is set up to mimic the spirit of Karpathy's `autoresearch`, but for MRNet.

## Active Loop

The active autonomous improvement loop lives in:

[`MRNet-AI-assited-diagnosis-of-knee-injuries/autoresearch_loop.py`](/Users/Apple/Workdir/mrnet/MRNet-AI-assited-diagnosis-of-knee-injuries/autoresearch_loop.py)

and is documented in:

[`MRNet-AI-assited-diagnosis-of-knee-injuries/program.md`](/Users/Apple/Workdir/mrnet/MRNet-AI-assited-diagnosis-of-knee-injuries/program.md)

## Intent

The system should continuously improve the model architecture by:

1. reading the current best saved architecture
2. mutating a few architecture choices
3. running a short experiment
4. advancing only if validation AUC improves

This is the MRNet equivalent of the experiment-advance loop in the original `autoresearch` folder.
