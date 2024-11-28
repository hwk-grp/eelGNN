import numpy
import pandas
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pickle
import copy
from utils.feature import get_onehot_features


class TriadicData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.x.size(0)
        if key == 'slv_edge_index':
            return self.slv_x.size(0)
        if key == 'chr_edge_index':
            return self.chr_x.size(0)
        return super().__inc__(key, value, *args, **kwargs)

target_list = ['Absorption max (nm)', 'Emission max (nm)', 'Lifetime (ns)', 'Quantum yield', 'log(e/mol-1 dm3 cm-1)',
               'abs FWHM (cm-1)', 'emi FWHM (cm-1)', 'abs FWHM (nm)', 'emi FWHM (nm)', 'esol', 'molar mass']


def load_multi_dataset(path_user_dataset, target_idx, alg_setup, hetero_setup, h_setup, gamma):
    list_mols = list()
    df = pandas.read_excel(path_user_dataset)
    id_target = numpy.array(df.dropna(subset=[target_list[target_idx]]))
    id_target = id_target[id_target[:, 2] != 'gas']
    id_target = id_target[id_target[:, 1] != id_target[:, 2]]

    if target_idx == 2:
        id_target[:, 5] = numpy.log10(id_target[:, 5].astype(float))
    elif target_idx == 3:
        id_target[:, 6] = numpy.where(id_target[:, 6] == 0, 1e-5, id_target[:, 6])
        id_target[:, 6] = numpy.log10(id_target[:, 6].astype(float))
    elif target_idx == 5:
        inv_target = numpy.array(df.dropna(subset=['abs FWHM (nm)']))
        inv_target = inv_target[inv_target[:, 2] != 'gas']
        inv_target = inv_target[inv_target[:, 1] != inv_target[:, 2]]
        ld, nm = inv_target[:, 3], inv_target[:, 10]
        inv_target[:, 8] = 1e7 * (1/(ld - nm/2) - 1/(ld + nm/2))
        id_target = numpy.vstack((id_target, inv_target))
    elif target_idx == 6:
        inv_target = numpy.array(df.dropna(subset=['emi FWHM (nm)']))
        inv_target = inv_target[inv_target[:, 2] != 'gas']
        inv_target = inv_target[inv_target[:, 1] != inv_target[:, 2]]
        ld, nm = inv_target[:, 4], inv_target[:, 11]
        inv_target[:, 9] = 1e7 * (1 / (ld - nm / 2) - 1 / (ld + nm / 2))
        id_target = numpy.vstack((id_target, inv_target))
    elif target_idx == 7:
        inv_target = numpy.array(df.dropna(subset=['abs FWHM (cm-1)']))
        ld, cm = inv_target[:, 3], inv_target[:, 8]
        inv_target[:, 10] = 2 * (-1e7/cm + (1e14/(cm**2) + ld**2)**(1/2))
        id_target = numpy.vstack((id_target, inv_target))
    elif target_idx == 8:
        inv_target = numpy.array(df.dropna(subset=['emi FWHM (cm-1)']))
        ld, cm = inv_target[:, 4], inv_target[:, 9]
        inv_target[:, 11] = 2 * (-1e7/cm + (1e14/(cm**2) + ld**2)**(1/2))
        id_target = numpy.vstack((id_target, inv_target))

    for i in tqdm(range(id_target.shape[0]), desc='Load molecular structures...'):
        mol = smiles_to_graph(id_target[i, 1], id_target[i, 2], alg_setup, hetero_setup,
                                            h_setup, gamma, target=id_target[i, target_idx + 3])
        if mol is not None:
            list_mols.append((id_target[i, 1], id_target[i, 2], mol, id_target[i, target_idx + 3]))

    return list_mols


def smiles_to_graph(csmiles, ssmiles, alg_setup, hetero_setup, addh, gamma, target):
    mol = Chem.MolFromSmiles(csmiles)
    if addh or mol.GetNumAtoms() == 1: mol = Chem.AddHs(mol)
    features = rdMolDescriptors.GetFeatureInvariants(mol)
    stereo = Chem.FindMolChiralCenters(mol)
    n_atoms = mol.GetNumAtoms()
    chiral_centers = [0] * mol.GetNumAtoms()
    for i in stereo:
        chiral_centers[i[0]] = i[1]

    bonds = []
    chr_bonds = []
    rel = []

    chr_af = get_onehot_features(csmiles, chiral_centers, features, addh)
    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            bond_ij = mol.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                chr_bonds.append([i, j])
                bonds.append([i, j])
                rel.append(0)
            if i == j:
                chr_bonds.append([i, j])
                bonds.append([i, j])
                rel.append(0)

    subg_af = copy.deepcopy(chr_af)

    mol2 = Chem.MolFromSmiles(ssmiles)
    if addh or mol2.GetNumAtoms() == 1: mol2 = Chem.AddHs(mol2)
    features2 = rdMolDescriptors.GetFeatureInvariants(mol2)
    stereo2 = Chem.FindMolChiralCenters(mol2)
    n_atoms2 = mol2.GetNumAtoms()
    chiral_centers2 = [0] * mol2.GetNumAtoms()
    for i in stereo2:
        chiral_centers2[i[0]] = i[1]
    slv_af = get_onehot_features(ssmiles, chiral_centers2, features2, addh)

    slv_bonds = []
    for i in range(mol2.GetNumAtoms()):
        for j in range(mol2.GetNumAtoms()):
            bond_ij = mol2.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                slv_bonds.append([i, j])
                bonds.append([n_atoms + i, n_atoms + j])
                rel.append(1)
            if i == j:
                slv_bonds.append([i, j])
                bonds.append([n_atoms + i, n_atoms + j])
                rel.append(1)

    subg_af = copy.deepcopy(subg_af + slv_af)

    if alg_setup == 'pauling':
        lb_chr = 2.5
        ub_chr = 2.9
        lb_slv = 2.5
        ub_slv = 2.9

        # Elements without electronegativity information are set to zero
        pauling_en = numpy.array([2.2, 0.0, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 0.0,
                  0.93, 1.31, 1.61, 1.9, 2.19, 2.58, 3.16, 0.0, 0.82, 1.0,
                  1.36, 1.54, 1.63, 1.66, 1.55, 1.83, 1.88, 1.91, 1.9, 1.65,
                  1.81, 2.01, 2.18, 2.55, 2.96, 0.0, 0.82, 0.95, 1.22, 1.33,
                  1.6, 2.16, 2.1, 2.2, 2.28, 2.2, 1.93, 1.69, 1.78, 1.96,
                  2.05, 2.1, 2.66, 2.6, 0.79, 0.89, 1.1, 1.12, 1.13, 1.14,
                  0.0, 1.17, 0.0, 1.2, 0.0, 1.22, 1.23, 1.24, 1.25, 0.0,
                  1.0, 1.3, 1.5, 1.7, 1.9, 2.2, 2.2, 2.2, 2.4, 1.9, 1.8,
                  1.8, 1.9, 2.0, 2.2, 0.0, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7,
                  1.3, 1.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        c_atom_charge = [pauling_en[mol.GetAtomWithIdx(i).GetAtomicNum() - 1] for i in range(mol.GetNumAtoms())]
        s_atom_charge = [pauling_en[mol2.GetAtomWithIdx(i).GetAtomicNum() - 1] for i in range(mol2.GetNumAtoms())]
    elif alg_setup == 'gasteiger':
        lb_chr = -gamma
        ub_chr = gamma
        lb_slv = -0.15
        ub_slv = 0.15

        with open('utils/charge/chromophore_solute_gasteiger.pkl', 'rb') as pickle_file:
            cd_dict = pickle.load(pickle_file)

        with open('utils/charge/chromophore_solvent_gasteiger.pkl', 'rb') as pickle_file2:
            sd_dict = pickle.load(pickle_file2)

        c_atom_charge = cd_dict[csmiles]
        s_atom_charge = sd_dict[ssmiles]
    else:
        lb_chr = -gamma
        ub_chr = gamma
        lb_slv = -0.235
        ub_slv = 0.235

        with open('utils/charge/chromophore_solute_espaloma.pkl', 'rb') as pickle_file:
            cd_dict = pickle.load(pickle_file)

        with open('utils/charge/chromophore_solvent_espaloma.pkl', 'rb') as pickle_file2:
            sd_dict = pickle.load(pickle_file2)

        c_atom_charge = cd_dict[csmiles]
        s_atom_charge = sd_dict[ssmiles]

    if alg_setup == 'pauling':
        nc = [i for i, ac in enumerate(c_atom_charge) if ac > ub_chr and mol.GetAtomWithIdx(i).GetAtomicNum() != 1]
        pc = [i for i, ac in enumerate(c_atom_charge) if ac < lb_chr and mol.GetAtomWithIdx(i).GetAtomicNum() != 1]
        ns = [i for i, ac in enumerate(s_atom_charge) if ac > ub_chr and mol2.GetAtomWithIdx(i).GetAtomicNum() != 1]
        ps = [i for i, ac in enumerate(s_atom_charge) if ac < lb_chr and mol2.GetAtomWithIdx(i).GetAtomicNum() != 1]
    else:
        pc = [i for i, ac in enumerate(c_atom_charge) if ac > ub_chr and mol.GetAtomWithIdx(i).GetAtomicNum() != 1]
        nc = [i for i, ac in enumerate(c_atom_charge) if ac < lb_chr and mol.GetAtomWithIdx(i).GetAtomicNum() != 1]
        ps = [i for i, ac in enumerate(s_atom_charge) if ac > ub_slv and mol2.GetAtomWithIdx(i).GetAtomicNum() != 1]
        ns = [i for i, ac in enumerate(s_atom_charge) if ac < lb_slv and mol2.GetAtomWithIdx(i).GetAtomicNum() != 1]

    type_idx = 2
    for npc in [nc, pc]:
        for nps in [ns, ps]:
            for chr_atom in npc:
                for slv_atom in nps:
                    bonds.append([n_atoms + slv_atom, chr_atom])
                    bonds.append([chr_atom, n_atoms + slv_atom])
                    rel.append(type_idx)
                    rel.append(type_idx)
            type_idx += 1

    # inter means whether intermolecular edge was formed
    if set(rel) == {0, 1} or set(rel) == {1, 0}:
        inter = False
    else:
        inter = True

    if hetero_setup == 'one_rel':
        rel = [min(x, 2) for x in rel]
    elif hetero_setup == 'attr_repl':
        rel = [2 if x == 5 else 3 if x == 4 else x for x in rel]

    # If edge is not formed, make a arbitrary graph which is not used in model
    if alg_setup == 'no_edge':
        subg_af = numpy.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
       0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
       0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 1, 0]])
        bonds = [[0, 0], [0, 1], [1, 0], [1, 1]]

    atom_feats = torch.tensor(numpy.array(subg_af), dtype=torch.float)
    bonds = torch.tensor(bonds, dtype=torch.long).T
    edge_type = torch.tensor(rel, dtype=torch.long).t().contiguous()
    slv_x = torch.tensor(numpy.array(slv_af), dtype=torch.float)
    slv_edge_index = torch.tensor(slv_bonds, dtype=torch.long).T
    chr_x = torch.tensor(numpy.array(chr_af), dtype=torch.float)
    chr_edge_index = torch.tensor(chr_bonds, dtype=torch.long).T
    y = torch.tensor(target, dtype=torch.float).view(-1, 1)

    return TriadicData(x=atom_feats, y=y, edge_index=bonds, edge_type=edge_type, n_atoms=n_atoms, inter=inter,
                       csmls=csmiles, ssmls=ssmiles, slv_x=slv_x, slv_edge_index=
                       slv_edge_index, chr_x=chr_x, chr_edge_index=chr_edge_index)
