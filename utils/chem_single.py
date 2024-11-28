import numpy
import pandas
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm
from torch_geometric.data import Data
from utils.feature import get_onehot_features

def load_single_dataset(path_user_dataset, h_setup):
    list_mols = list()
    df = pandas.read_excel(path_user_dataset)
    id_target = numpy.array(df)

    n = id_target.shape[0]
    for i in tqdm(range(n), desc='Load molecular structures...'):
        mol = smiles_to_single_graph(id_target[i, 0], h_setup, target=id_target[i, 1])
        if mol is not None:
            list_mols.append((id_target[i, 0], mol, id_target[i, 1]))

    return list_mols


def smiles_to_single_graph(smiles, addh, target):
    # chromophore smiles to a RDKit object
    mol = Chem.MolFromSmiles(smiles)
    if addh or mol.GetNumAtoms() == 1: mol = Chem.AddHs(mol)
    features = rdMolDescriptors.GetFeatureInvariants(mol)
    stereo = Chem.FindMolChiralCenters(mol)
    chiral_centers = [0] * mol.GetNumAtoms()
    for i in stereo:
        chiral_centers[i[0]] = i[1]
    bonds = []

    af = get_onehot_features(smiles, chiral_centers, features, addh)
    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            bond_ij = mol.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                bonds.append([i, j])
            if i == j:
                bonds.append([i, j])

    atom_feats = torch.tensor(numpy.array(af), dtype=torch.float)
    bonds = torch.tensor(bonds, dtype=torch.long).T
    y = torch.tensor(target, dtype=torch.float).view(-1, 1)

    return Data(x=atom_feats, edge_index=bonds, y=y)



