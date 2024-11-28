'''
These functions are imported from https://github.com/Namkyeong/CGIB/tree/main/MolecularInteraction
Set of atom features same with CGIB, CIGIN is used.
'''

import numpy
from rdkit import Chem

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


def get_onehot_features(smiles, stereo, features, explicit_H=False):
    af = []
    mol = Chem.MolFromSmiles(smiles)
    tmp_an = mol.GetNumAtoms()
    if explicit_H or tmp_an == 1: mol = Chem.AddHs(mol)
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        possible_atoms = ['H', 'B', 'C', 'N', 'O', 'F', 'Na', 'Si', 'P', 'S', 'Cl', 'Ge', 'Se', 'Br', 'Sn', 'Te', 'I']
        atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atoms) # 17
        atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3]) # 4
        atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1]) # 2
        atom_features += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) # 7
        atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1]) # 3
        atom_features += one_of_k_encoding_unk(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D]) # 5
        atom_features += [int(i) for i in list("{0:06b}".format(features[i]))]

        if not explicit_H:
            atom_features += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

        try:
            atom_features += one_of_k_encoding_unk(stereo[i], ['R', 'S'])
            atom_features += [atom.HasProp('_ChiralityPossible')]
        except Exception as e:

            atom_features += [False, False
                              ] + [atom.HasProp('_ChiralityPossible')]
        af.append(numpy.array(atom_features))

    return af


