import torch

def train(model, optimizer, data_loader, criterion):
    model.train()
    train_loss = 0

    for i, (batch) in enumerate(data_loader):
        if torch.cuda.is_available(): batch.cuda()
        # Trick to realize eelGNN
        pseudo_batch = list()
        for i in set(batch.x_batch.tolist()):
            num_c_atoms = batch.n_atoms[i]
            num_entire_nodes = torch.bincount(batch.x_batch)[i]
            [pseudo_batch.append(2 * i) for _ in range(0, num_c_atoms)]
            [pseudo_batch.append(2 * i + 1) for _ in range(num_c_atoms, num_entire_nodes)]
        tensor_pseudo_batch = torch.tensor(pseudo_batch, requires_grad=False)
        # optimizing step
        pred = model(batch, tensor_pseudo_batch.cuda()) if torch.cuda.is_available() else model(batch, tensor_pseudo_batch)
        loss = criterion(pred, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.detach().item()

    return train_loss / len(data_loader)


def test(model, data_loader):
    model.eval()
    list_preds = list()

    with torch.no_grad():
        for batch in data_loader:
            batch.cuda()
            pseudo_batch = list()
            for i in set(batch.x_batch.tolist()):
                num_c_atoms = batch.n_atoms[i]
                num_entire_nodes = torch.bincount(batch.x_batch)[i]
                [pseudo_batch.append(2 * i) for _ in range(0, num_c_atoms)]
                [pseudo_batch.append(2 * i + 1) for _ in range(num_c_atoms, num_entire_nodes)]
            tensor_pseudo_batch = torch.tensor(pseudo_batch, requires_grad=False)
            preds = model(batch, tensor_pseudo_batch.cuda())
            list_preds.append(preds)

    return torch.cat(list_preds, dim=0).cpu().numpy()

def valid(model, data_loader, criterion):
    model.eval()
    valid_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            batch.cuda()
            pseudo_batch = list()
            for i in set(batch.x_batch.tolist()):
                num_c_atoms = batch.n_atoms[i]
                num_entire_nodes = torch.bincount(batch.x_batch)[i]
                [pseudo_batch.append(2 * i) for _ in range(0, num_c_atoms)]
                [pseudo_batch.append(2 * i + 1) for _ in range(num_c_atoms, num_entire_nodes)]
            tensor_pseudo_batch = torch.tensor(pseudo_batch, requires_grad=False)
            preds = model(batch, tensor_pseudo_batch.cuda())
            loss = criterion(preds, batch.y)
            valid_loss += loss.detach().item()

    return valid_loss / len(data_loader)
