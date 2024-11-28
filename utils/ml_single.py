import torch

def train(model, optimizer, data_loader, criterion):
    model.train()
    train_loss = 0
    for i, (batch) in enumerate(data_loader):
        if torch.cuda.is_available():
            batch.cuda()
        pred = model(batch)
        loss = criterion(pred, batch.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().item()

    return train_loss / len(data_loader)

def valid(model, data_loader, criterion):
    model.eval()
    valid_loss = 0

    with torch.no_grad():
        for i, (batch) in enumerate(data_loader):
            if torch.cuda.is_available():
                batch.cuda()

            pred = model(batch)

            loss = criterion(pred, batch.y)

            valid_loss += loss.detach().item()

        return valid_loss / len(data_loader)

def test(model, data_loader):
    model.eval()
    list_preds = list()

    with torch.no_grad():
        for batch in data_loader:
            if torch.cuda.is_available():
                batch.cuda()

            preds = model(batch)
            list_preds.append(preds)

    return torch.cat(list_preds, dim=0).cpu().numpy()
