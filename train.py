import torch
from torch import nn
from torch.utils.data import DataLoader
from networks import NeuralNetwork
from modules import load_addition_torch_dataset, ids_to_str, train, test
import time as t
import pickle
import os
from util import time, alerm
from plot import plot


batch_size = 100
(training_data, test_data), (char_to_id, id_to_char) = load_addition_torch_dataset()
train_dataloader = DataLoader(training_data, batch_size)
test_dataloader = DataLoader(test_data, batch_size)

# Q = ids_to_str(training_data[0][0], id_to_char)
# A = ids_to_str(training_data[0][1], id_to_char)
# print(Q, A)

# 입력 데이터 반전
# x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

################################################################################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device\n".format(device))

model = NeuralNetwork(num_embeddings=len(char_to_id), embedding_dim=16, hidden_size=128, x_num_layers=7, t_num_layers=5).to(device)
network = 'ER_ERL(LSTM)'

for X, t in train_dataloader:
    pred = model(X, t[:, :-1])
    print(pred.shape)
exit()

################################################################################################################################

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
graph_datas = {'train_losses':[], 'test_losses':[], 'train_losses_all':[], 'train_accs':[], 'test_accs':[]}
# with open(model_path.replace('model','graph_datas')+'.pkl', 'rb') as f:
#     graph_datas = pickle.load(f)
    # losses = {'train_losses':graph_datas['train_losses'], 'test_losses':graph_datas['test_losses']}
    # accs = {'train_accs':graph_datas['train_accs'], 'test_accs':graph_datas['test_accs']}
    # plot.loss_graphs(losses, smooth=False, ylim=0.001)
    # plot.accuracy_graphs(accs, ylim_min=95)
    # plot.loss_graph(graph_datas['train_losses_all'], smooth=False, ylim=0.1)
    # exit()

# with open('graph_datas acc_99.53 loss_0.000361 CCPCPCC_LL(+N,D)(f128, h100) 100ep'+'.pkl', 'rb') as f:
    # graph_datas2 = pickle.load(f)
# with open('graph_datas acc_99.44 loss_0.000318 CCPCPCCP_LL(+N,D)(f128, h100) 100ep'+'.pkl', 'rb') as f:
    # graph_datas3 = pickle.load(f)

# test_losses = {'CCPCPC+C loss':graph_datas['test_losses'], 'CCPCPCC loss':graph_datas2['test_losses'], 'CCPCPCCP loss':graph_datas3['test_losses']}
# test_accs = {'CCPCPC+C test_accs':graph_datas['test_accs'], 'CCPCPCC test_accs':graph_datas2['test_accs'], 'CCPCPCCP loss':graph_datas3['test_accs']}
# plot.loss_graphs(test_losses, smooth=False, ylim=0.001)
# plot.accuracy_graphs(test_accs)
# exit()

start_time = t.time()
max_acc = 0
epochs = 100
file_path1, file_path2 = None, None # file_path1_
save_min_acc = 0.99

for i in range(epochs):
    print(f"Epoch {i+1} ({time.str_hms_delta(start_time)})\n-------------------------------")
    train_acc_avg, train_loss_avg = train(train_dataloader, model, device, loss_fn, optimizer, graph_datas)
    acc, test_loss_avg = test(test_dataloader, model, device, loss_fn)
    
    graph_datas['train_losses'].append(train_loss_avg)
    graph_datas['test_losses'].append(test_loss_avg)
    graph_datas['train_accs'].append(train_acc_avg)
    graph_datas['test_accs'].append(acc)

    if acc > max_acc:
        max_acc = acc

        if acc > save_min_acc:
            info = f'acc_{(100*acc):>0.2f} loss_{test_loss_avg:>0.6f}'
            if file_path1 != None and os.path.isfile(file_path1): os.remove(file_path1)
            if file_path2 != None and os.path.isfile(file_path2): os.remove(file_path2)
            
            file_path1 = f"model {info} {network} {i+1}ep.pth"
            file_path2 = file_path1.replace('model','graph_datas').replace('pth','pkl')
            
            torch.save(model, file_path1)
            with open(file_path2, 'wb') as f:
                pickle.dump(graph_datas, f)
    
print("Done!")

if epochs > 0:
    with open(f'graph_datas acc_{(100*acc):>0.2f} loss_{test_loss_avg:>0.6f} {network} {epochs}ep.pkl', 'wb') as f:
        pickle.dump(graph_datas, f)

    # 모델 저장하기
    torch.save(model, f"model acc_{(100*acc):>0.2f} loss_{test_loss_avg:>0.6f} {network} {epochs}ep.pth")
    print("Saved PyTorch Model State to model.pth")

    try:
        alerm()
    except Exception as e:
        print('Alerm Error!')
        print(e)

    losses = {'train_losses':graph_datas['train_losses'], 'test_losses':graph_datas['test_losses']}
    accs = {'train_accs':graph_datas['train_accs'], 'test_accs':graph_datas['test_accs']}
    plot.loss_graphs(losses, smooth=False, ylim=0.001)
    plot.accuracy_graphs(accs, ylim_min=95)
    plot.loss_graph(graph_datas['train_losses_all'], ylim=0.1)