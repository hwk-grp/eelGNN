import numpy
import pandas
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle

dataset_name = 'chromophore'

two_data = numpy.array(pandas.read_excel('../res/data/' + dataset_name + '.xlsx'))
cs = list(set(two_data[:, 1]))
ss = list(set(two_data[:, 2]))
ss.remove('gas')

alg = 'espaloma'

def charge_generator(chromophores, solvents, algorithm):
    cd = {}
    sd = {}

    # The installation of espaloma_charge is avaiable at https://github.com/choderalab/espaloma_charge
    if algorithm == 'espaloma':
        from espaloma_charge import charge

    for lst in [chromophores, solvents]:
        for sm in tqdm(lst):
            mol = Chem.AddHs(Chem.MolFromSmiles(sm))
            n_atoms = mol.GetNumAtoms()
            if algorithm == 'gasteiger':
                AllChem.ComputeGasteigerCharges(mol)
                oric = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())]
            else:
                oric = charge(mol)
            if all(abs(x) < 5 for x in oric):
                pass
            else:
                print(f'Calculation of {sm} is not available!')
                oric = [0] * len(oric)
            if Chem.MolFromSmiles(sm).GetNumAtoms() > 1:
                for i in range(n_atoms):
                    atom = mol.GetAtoms()[i]
                    nei = atom.GetNeighbors()
                    for adj_atom in nei:
                        if adj_atom.GetAtomicNum() == 1:
                            hidx = adj_atom.GetIdx()
                            oric[i] = oric[i] + oric[hidx]
                oric = oric[:Chem.RemoveHs(mol).GetNumAtoms()]
            if lst == cs:
                cd[sm] = oric
            elif lst == ss:
                sd[sm] = oric

    return cd, sd

chromophore_dict, solvent_dict = charge_generator(cs, ss, algorithm=alg)

# Open the file in binary write mode
with open(f'./charge/{dataset_name}_solute_{alg}.pkl', 'wb') as file:
    # Serialize the dictionary and save it to the file
    pickle.dump(chromophore_dict, file)

with open(f'./charge/{dataset_name}_solvent_{alg}.pkl', 'wb') as file:
    # Serialize the dictionary and save it to the file
    pickle.dump(solvent_dict, file)
