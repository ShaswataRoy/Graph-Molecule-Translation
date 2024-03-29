# Multi-Objective Molecule Generation using Interpretable Substructures

This is the implementation of our ICML 2020 paper: https://arxiv.org/abs/2002.03244

## Environment Setup
An environment for rationale can be easily setup via Anaconda:
```
git clone https://github.com/wengong-jin/multiobj-rationale.git
cd multiobj-rationale
conda env create -f environment.yml
conda activate rationale
```

## Generate molecules with specific substructures
To generate molecules that contain specific substructures (e.g. benzene), first specify a rationale file named `rationale.txt`. Here is one example file with one line for a benzene.
```
c1ccc[c:1]c1
```
where atoms marked with 1 means the model should grow this fragment from these atoms. Then run
```
python decode.py --rationale rationale.txt --model ckpt/gsk3_jnk3_qed_sa/model.final --num_decode 100
```
This will generate 100 molecules with at least one benzene ring starting at the highlighted position.
