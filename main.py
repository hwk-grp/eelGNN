import os
# os.environ['OMP_NUM_THREADS'] = "2"
import numpy
import pandas
import torch
import random
import sys
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import utils.ml_multiple
import utils.ml_single
from utils.chem_single import load_single_dataset
from utils.chem_multiple import load_multi_dataset
from utils.model import model_builder

# torch.set_num_threads(2)

# experimental setting
dataset_name = 'chromophore'
n_epochs = 500
train_ratio = 0.8
n_splits = 5
n_repeats = 5
criterion = torch.nn.MSELoss()
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

h_setup = False# If True, Chem.AddHs() will operate to every molecule

# Set target and algorithm.
try:
    target_idx, alg_idx, hetero_idx, aggr_idx, pool_idx = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
except:
    target_idx, alg_idx, hetero_idx, aggr_idx, pool_idx = 5, 1, 0, 1, 0

target_list = ['Absorption max (nm)', 'Emission max (nm)', 'Lifetime (ns)', 'Quantum yield', 'log(e/mol-1 dm3 cm-1)',
               'abs FWHM (cm-1)', 'emi FWHM (cm-1)', 'esol', 'molar_mass']
algorithm_list = ['espaloma', 'gasteiger', 'pauling', 'no_edge']
hetero_list = ['one_rel', 'attr_repl', 'four_rel']
func_gnn = ['add', 'mean']

target_setup, alg_setup, hetero_setup, aggr_setup, pool_setup = target_list[target_idx - 1], algorithm_list[alg_idx - 1], hetero_list[hetero_idx - 1], func_gnn[aggr_idx - 1], func_gnn[pool_idx - 1]
multi_component = (target_setup != 'esol' and target_setup != 'molar_mass')

if not multi_component:
    alg_setup == 'no_edge'

if alg_setup == 'no_edge':
    if multi_component:
        gnn = 'gcn_multi'
    else:
        gnn = 'gcn_single'
else:
    gnn = 'eelgcn'
    gamma = 0.5 if alg_setup == 0 else 0.25

if hetero_setup == 'one_rel':
    num_rel = 3
elif hetero_setup == 'attr_repl':
    num_rel = 4
else:
    num_rel = 6

# Hyperparameters
if target_idx == 1:
    batch_size, init_lr, l2_coeff, dims = 128, 5e-3, 5e-6, 104
elif target_idx == 2:
    batch_size, init_lr, l2_coeff, dims = 256, 5e-3, 5e-6, 52
elif target_idx == 3:
    batch_size, init_lr, l2_coeff, dims = 128, 5e-3, 5e-6, 52
elif target_idx == 4:
    batch_size, init_lr, l2_coeff, dims = 128, 1e-3, 1e-6, 26
elif target_idx == 5:
    batch_size, init_lr, l2_coeff, dims = 128, 1e-3, 1e-6, 104
elif target_idx == 6:
    batch_size, init_lr, l2_coeff, dims = 64, 5e-3, 5e-6, 104
elif target_idx == 7:
    batch_size, init_lr, l2_coeff, dims = 256, 1e-3, 5e-6, 104
elif target_idx == 8:
    batch_size, init_lr, l2_coeff, dims = 64, 5e-3, 1e-6, 52
else:# target_idx == 10 or something
    batch_size, init_lr, l2_coeff, dims = 64, 1e-3, 5e-6, 104

n_layers = [3, 3, 3]
ensemble = [2, 2]
follow_batch = ['x', 'slv_x', 'chr_x'] if multi_component else None

# Customize result file path and name
fdir = 'res/preds'
target_name = target_setup.replace(" ", "_") if target_idx != 5 else 'loge'
file_name = f'{gnn}_{target_name}_{alg_setup}_{aggr_setup}_{pool_setup}'


def seed_everything(data, seed=0):
    # To fix the random seed
    random.seed(seed)
    random.shuffle(data)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # backends
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except:
        pass
    kfd = KFold(n_splits=n_splits, random_state=seed, shuffle=True)

    return kfd


# Machine learning start!
with open(f'{fdir}/{file_name}.txt', 'w') as f:
    sys.stdout = f
    try:
        # Load dataset
        if multi_component:
            wdata = load_multi_dataset('res/data/' + dataset_name + '.xlsx', target_idx - 1, alg_setup, hetero_setup, h_setup, gamma)
        else:
            wdata = load_single_dataset('res/data/' + target_setup + '.xlsx', h_setup)

        # carry out experiment
        tot_test_rmse = []
        tot_test_r2 = []

        for repeat in range(n_repeats):
            kf = seed_everything(wdata)
            data = wdata
            n_train = int(train_ratio * len(data))
            n_test = len(data) - n_train
            if multi_component:
                chromophore_smiles = [x[0] for x in data]
                solvent_smiles = [x[1] for x in data]
                mols = [x[2] for x in data]
                targets = [x[3] for x in data]
            else:
                molecule_smiles = [x[0] for x in data]
                mols = [x[1] for x in data]
                targets = [x[2] for x in data]
            n_atom_feats = mols[0].x.size(1)

            # Generate training and test datasets
            train_data = mols[:n_train]
            test_data = mols[n_train:]
            if multi_component:
                test_chromophore = chromophore_smiles[n_train:]
                test_solvent = solvent_smiles[n_train:]
            else:
                test_molecule = molecule_smiles[n_train:]
            test_targets = numpy.array(targets[n_train:]).reshape(-1, 1)
            train_targets = numpy.array(targets[:n_train])
            ttavg, ttstd = numpy.average(train_targets), numpy.std(train_targets)
            for i in range(len(train_data)):
                train_data[i].y = torch.tensor([(train_targets[i] - ttavg) / ttstd], dtype=torch.float).view(-1, 1)
            for i in range(len(test_data)):
                test_data[i].y = torch.tensor([(test_targets[i] - ttavg) / ttstd], dtype=torch.float).view(-1, 1)
            test_loader = DataLoader(test_data, batch_size=batch_size, follow_batch=follow_batch)

            # Train graph neural network (GNN)
            print(f'Train {repeat}/{n_repeats} GNN-based predictor...')
            models = list()
            for fold, (train, valid) in enumerate(kf.split(train_data)):# make it function(def)
                train_data_cv = torch.utils.data.dataset.Subset(train_data, train)
                valid_data_cv = torch.utils.data.dataset.Subset(train_data, valid)
                valid_targets = numpy.array([x.y.item() for x in valid_data_cv]).reshape(-1, 1)

                train_loader = DataLoader(train_data_cv, batch_size=batch_size, shuffle=True, follow_batch=follow_batch)
                valid_loader = DataLoader(valid_data_cv, batch_size=batch_size, follow_batch=follow_batch)
                models = model_builder(models, gnn, n_atom_feats, num_rel, dims, n_layers, ensemble, aggr_setup, pool_setup)
                models[fold].to(device)
                optimizer = torch.optim.Adam(models[fold].parameters(), lr=init_lr, weight_decay=l2_coeff)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=20, verbose=True)

                for i in range(n_epochs):
                    if multi_component:
                        train_loss = utils.ml_multiple.train(models[fold], optimizer, train_loader, criterion)
                        valid_loss = utils.ml_multiple.valid(models[fold], valid_loader, criterion)
                    else:
                        train_loss = utils.ml_single.train(models[fold], optimizer, train_loader, criterion)
                        valid_loss = utils.ml_single.valid(models[fold], valid_loader, criterion)

                    if i % 20 == 19:
                        print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(i + 1, n_epochs, train_loss ** (0.5)))
                    scheduler.step(valid_loss)

                    if i == n_epochs - 1:
                        print(f'{fold}-fold train complete, RMSE of validation data: {round(valid_loss ** (1/2), 4)}')

                torch.save(models[fold], f'res/state_dict/seed{repeat}_model{fold}_sd_{file_name}.pt')

            # Test the trained GNN
            if multi_component:
                preds = utils.ml_multiple.test(models[0], test_loader)
                for i in range(1, n_splits):
                    preds += utils.ml_multiple.test(models[i], test_loader)
            else:
                preds = utils.ml_single.test(models[0], test_loader)
                for i in range(1, n_splits):
                    preds += utils.ml_single.test(models[i], test_loader)
            preds /= n_splits
            preds = preds * ttstd + ttavg
            test_mae = numpy.mean(numpy.abs(test_targets - preds))
            test_rmse = numpy.sqrt(numpy.mean((test_targets - preds) ** 2))
            r2 = r2_score(test_targets, preds)
            print(f'Average MAE: {test_mae}\tAverage RMSE: {test_rmse}\tAverage R2 score: {r2}')
            tot_test_rmse.append(test_rmse)
            tot_test_r2.append(r2)

            # save prediction results
            tot = []
            yes_edge = []
            no_edge = []

            if multi_component:
                for i in range(preds.shape[0]):
                    tot.append([test_chromophore[i], test_solvent[i], test_targets[i].item(), preds[i].item()])
                df = pandas.DataFrame(tot)
                df.columns = ['chromophore', 'solvent', 'true_y', 'pred_y']
                df.to_excel(f'{fdir}/preds_{str(repeat)}exp_{file_name}.xlsx', index=False)
            else:
                for i in range(preds.shape[0]):
                    tot.append([test_molecule[i], test_targets[i].item(), preds[i].item()])
                df = pandas.DataFrame(tot)
                df.columns = ['molecule', 'true_y', 'pred_y']
                df.to_excel(f'{fdir}/preds_{str(repeat)}exp_{file_name}.xlsx', index=False)

        final_test_rmse = numpy.mean(numpy.array(tot_test_rmse))
        final_test_rmstd = numpy.std(numpy.array(tot_test_rmse))
        print(f'{n_repeats}-experiment averaged result: test RMSE, std: {final_test_rmse}, {final_test_rmstd}')

        final_test_r2 = numpy.mean(numpy.array(tot_test_r2))
        final_test_r2std = numpy.std(numpy.array(tot_test_r2))
        print(f'{n_repeats}-experiment averaged result: test R2, std: {final_test_r2}, {final_test_r2std}')
    finally:
        sys.stdout = sys.__stdout__

print('-------Final line-------')
