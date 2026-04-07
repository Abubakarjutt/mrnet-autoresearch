# Workspace Program

This workspace is set up to follow the same self-improvement principles as Karpathy's `autoresearch`, but for MRNet.

## Active Loop

The active autonomous improvement loop lives in:

[`MRNet-AI-assited-diagnosis-of-knee-injuries/autoresearch_loop.py`](/Users/Apple/Workdir/mrnet/MRNet-AI-assited-diagnosis-of-knee-injuries/autoresearch_loop.py)

and is documented in:

[`MRNet-AI-assited-diagnosis-of-knee-injuries/program.md`](/Users/Apple/Workdir/mrnet/MRNet-AI-assited-diagnosis-of-knee-injuries/program.md)

## Intent

The system should continuously improve the model architecture by:

1. running a baseline first on a fresh state directory
2. reading the current best saved architecture
3. mutating a constrained research surface while keeping the evaluation harness fixed
4. running a fixed-budget experiment
5. advancing only if validation AUC improves, or if the AUC is effectively identical and the architecture is simpler

This is the MRNet equivalent of the experiment-advance-reset loop in the original `autoresearch` folder.
