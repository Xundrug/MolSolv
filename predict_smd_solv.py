import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 

from openbabel import pybel
import numpy as np
import pandas as pd

from src.utils import filter_mol
import torch
from src.descriptor import mol2vec
from src.models import load_model
from src.optimize_mol import optimize
from src.gen_confs import gen_confs_set
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from multiprocessing import Pool
import argparse
import warnings
warnings.filterwarnings("ignore")


def predict_single(pmol, model, fmax):
    device = "cpu"
    obmol = pmol.OBMol
    if fmax:
        obmol, dE = optimize(obmol, fmax)
    else:
        dE = 0.0
    data = mol2vec(obmol)
    with torch.no_grad():
        data = data.to(device)
        solv = model(data).cpu().numpy()[0][0]
    return solv, dE


def predict_multicore_wrapper(param):
    block, fmax, charge = param
    nmodel, imodel = load_model()
    pmol = pybel.readstring("mol", block)
    if charge > 0.0:
        solv, dE = predict_single(pmol, imodel, fmax)
    else:
        solv, dE = predict_single(pmol, nmodel, fmax)
    dE = dE / 27.2114 * 627.5094
    return [solv, dE, solv+dE]


def predict_by_smi(smi, fmax, charge, cores, num_confs):
    blocks = gen_confs_set(smi, num_confs)
    params = []
    for block in blocks:
        params.append([block, fmax, charge])

    pool = Pool(cores)
    score = pool.map(predict_multicore_wrapper, params)
    pool.close()

    df_score = pd.DataFrame(score)
    dfsg_sorted = df_score.sort_values(2)
    lower_solv = dfsg_sorted.iloc[0, 0]
    lower_dE = dfsg_sorted.iloc[0, 1]
    return lower_solv, lower_dE

 
def predict(smi, fmax, cores, num_confs):
    pmol = pybel.readstring("smi", smi)
    if (not filter_mol( pmol )):
        print("#### Warning filter molecule")
        return 0.0
    charge = abs(pmol.charge)
    solv, dE = predict_by_smi(smi, fmax, charge, cores, num_confs)
    return solv, dE


def get_solv_data(smi, fmax, cores, num_confs):
    solv, dE =  predict(smi, fmax, cores, num_confs)
    data  = {}
    data['smi'] = smi
    data['solv'] = solv
    data['dE'] = dE
    return data

def run():
    parser = argparse.ArgumentParser(
        description='calculate solvation energy for small molecules')
    parser.add_argument('--smi', type=str, default='CCNCc1cnccc1', help='the molecular smiles')
    parser.add_argument('--fmax', type=float, default=0.01, help='The convergence criterion is that the force on all individual atoms should be less than fmax')
    parser.add_argument('--cores', type=int, default=None, help='the number of cpu for calculatuon')
    parser.add_argument('--num_confs', type=int, default=6, help='the number of conformation for solvation energy prediction')
    parser.add_argument('--output', type=str, default="molsolv_output.dat", help='the output file name')
    
    args = parser.parse_args()

    smi = args.smi
    fmax = args.fmax
    cores = args.cores
    num_confs = args.num_confs
    output = args.output

    data = get_solv_data(smi, fmax, cores, num_confs)
    
    with open(output, "a") as f:
        f.write(data['smi'] + "\t" + str(data["solv"]) + "\t" + str(data["dE"]) + "\n")
    
    print("\n\n")
    print("-----------------------------------------------------------------")
    print("smiles\tsolv (kcal/mol)\tdE (kcal/mol)\n{}\t{}\t{}".format(data['smi'], round(data["solv"], 2), round(data["dE"], 2)))
    print("-----------------------------------------------------------------")
    return 
    


if __name__=='__main__':
    run()
   
