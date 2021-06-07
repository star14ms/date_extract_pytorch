import torch
from torch.utils.data import Dataset
from dataset import sequence
from tqdm import tqdm
import numpy as np


class CustomImageDataset(Dataset):
    def __init__(self, x, t, transform=None, target_transform=None):
        self.x = x
        self.t = t
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.t[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = (image, label,)
        return sample


def load_addition_torch_dataset():
    
    # 데이터 로드 (45000, 7) (45000, 5) (5000, 7) (5000, 5)
    (x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
    char_to_id, id_to_char = sequence.get_vocab()

    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
    t_train, t_test = torch.from_numpy(t_train), torch.from_numpy(t_test)

    training_data = CustomImageDataset(x_train, t_train)
    test_data = CustomImageDataset(x_test, t_test)

    return (training_data, test_data), (char_to_id, id_to_char)

################################################################################################################################

def train(dataloader, model, device, loss_fn, optimizer, graph_datas=None, verbose=True):
    size = len(dataloader.dataset)
    train_loss, correct = 0, 0
    for batch, (X, t) in enumerate(dataloader):
        X, t = X.to(device), t.to(device)

        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, t)
        if graph_datas!=None: graph_datas['train_losses_all'].append(loss)
        train_loss += loss_fn(pred, t).item()
        correct += (pred.argmax(1) == t).type(torch.float).sum().item()

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            if verbose: print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss /= size
    correct /= size
    return correct, train_loss


def test(dataloader, model, device, loss_fn, return_wrongs_info=False, verbose=True, use_tqdm=False):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    if return_wrongs_info: wrong_idxs, ys, wrong_ts = [], [], []
    with torch.no_grad():
        for X, t in tqdm(dataloader) if use_tqdm else dataloader:
            X, t = X.to(device), t.to(device)
            decoder_x, t = t[:, :-1], t[:, 1:]
            pred = model(X, decoder_x)
            test_loss += loss_fn(pred, t).item()
            correct += (pred.argmax(1) == t).type(torch.float).sum().item()
            if return_wrongs_info: 
                wrong_idxs.extend(pred.argmax(1) != t)
                ys.extend(pred.cpu().numpy())
                wrong_ts.extend(t)

    test_loss /= size
    correct /= size
    if verbose: print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")

    if return_wrongs_info:
        wrong_idxs = np.where(np.array(wrong_idxs) == True)[0]
        ys = np.array(ys)[wrong_idxs]
        wrong_ts = np.array(wrong_ts)[wrong_idxs]
        return wrong_idxs, ys, wrong_ts
    else:
        return correct, test_loss

################################################################################################################################

def ids_to_str(ids, id_to_char):
    text = ''
    for i in range(len(ids)):
        id = ids[i].item()
        text = text + id_to_char[id]
    
    return text