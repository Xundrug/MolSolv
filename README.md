Machine Learning Method for Learning SMD Solvation Model

![image](https://github.com/lexsaints/powershell/blob/master/IMG/ps2.png)

## Requirements

* Python 3.6
* openbabel >= 3.0
* numpy 1.18.1
* scipy
* pandas 0.25.3
* freesasa
* pytorch
* pytorch geometric

You also can create the python environment by conda configure file:
```
conda env create -f environment.yaml
```
If you run torch-sparse with error, please uninstall the package `torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric`:
```
pip uninstall torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
```
and then reinstall them:
```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
```

## Usage
```
python predict_smd_solv.py -h

usage: predict_smd_solv.py [-h] [--smi SMI] [--fmax FMAX] [--cores CORES]
                           [--num_confs NUM_CONFS] [--output OUTPUT]

calculate solvation energy for small molecules

optional arguments:
  -h, --help            show this help message and exit
  --smi SMI             the molecular smiles
  --fmax FMAX           The convergence criterion is that the force on all
                        individual atoms should be less than fmax
  --cores CORES         the number of cpu for calculatuon
  --num_confs NUM_CONFS
                        the number of conformation for solvation energy
                        prediction
  --output OUTPUT       the output file name
```
