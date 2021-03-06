import numpy as np
import matplotlib.pyplot as plt
import math
import pickle, os
# %matplotlib inline 이게 뭐고, 어떻게 사용하는거지???

def save_graph_datas(graph_datas, network, optimizer, lr, str_data_info, save_inside_dir=True, pkl_dir="saved_pkls", verbose=True):
    
    if "test_accs" not in graph_datas:
        if network.saved_network_pkl == None:
            acc = "None"
        else:
            acc = network.saved_network_pkl.split(" ")[-5].lstrip("acc_")
    else:
        acc = math.floor(graph_datas["test_accs"][-1]*100)/100
    file_name = f"{str_data_info} ln_{network.learning_num} acc_{acc} " + \
        f"{optimizer.__class__.__name__} lr_{lr} {network.network_dir} learning_info.pkl"
    file_path = file_name
    if save_inside_dir:
        file_path = f"{pkl_dir}/{network.network_dir}/" + file_path

    if not os.path.exists(f"{pkl_dir}/{network.network_dir}"):
        os.makedirs(f"{pkl_dir}/{network.network_dir}")
        print(f"{pkl_dir}/{network.network_dir} 폴더 생성")

    with open(file_path, 'wb') as f:
        pickle.dump(graph_datas, f)

    if verbose: print(f"\n그래프 데이터 저장 성공!\n({file_name})")

def load_graph_datas(file_name="graph_data.pkl", pkl_dir="saved_pkls"): ### class 밖에선 self 지워야 함

    if (".pkl" not in file_name) or (file_name[-4:] != ".pkl"):
        file_name = f"{file_name}.pkl"
    file_path = file_name

    # network_dir = file_path.split(" [")[0].split(" ")[-1] + " [" + file_path.split(" [")[1].rstrip(" params.pkl")
    network_dir = file_path.split(" ")[-2] # " " -> "_" 같이 수정해주어야 함
    file_path = f"{network_dir}/{file_path}"

    if (file_path.split("/")[:-1] != f"{pkl_dir}"):
        file_path = f"{pkl_dir}/{file_path}"

    if not os.path.exists(f"{pkl_dir}/{network_dir}"):
        os.makedirs(f"{pkl_dir}/{network_dir}")
        print(f"Error: {pkl_dir}/{network_dir} 폴더가 없어서 생성")

    if not os.path.exists(file_path):
        print(f"Error: {file_path} 파일이 없음")
        file_path = file_name
        if not os.path.exists(file_path):
            print(f"Error: {file_path} 파일이 없음")
            file_path = f"{pkl_dir}/{file_name}"
            if not os.path.exists(file_path):
                print(f"Error: {file_name} 파일이 없음")
                return

    # main process
    with open(file_path, 'rb') as f:
        graph_datas = pickle.load(f)
        
    print(f"\n그래프 데이터 불러오기 성공!\n({file_name})")
    return graph_datas

def smooth_curve(x):
    """損失関数のグラフを滑らかにするために用いる

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]

class plot:

################################################################################################################################

    def loss_graph(losses, smooth=True, ylim=None):
        x = np.arange(len(losses))
        y = smooth_curve(losses) if smooth and len(losses) > 10 else losses
        plt.plot(x, y, f"-", label="loss")
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.ylim(0, ylim)
        plt.legend(loc='upper right')
        plt.show()
    

    def loss_graphs(losses_dic, smooth=True, ylim=None, colors=('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')):
        keys = list(losses_dic.keys())
        x = np.arange(len(losses_dic[keys[0]]))
        for key, color in zip(keys, colors):
            y = smooth_curve(losses_dic[key]) if smooth else losses_dic[key]
            plt.plot(x, y, label=key) # f"-{color}"
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.ylim(0, ylim)
        plt.legend(loc='upper right')
        plt.show()
   

    def many_loss_graphs(results_losses, graph_draw_num=9, col_num=3, sort=True, smooth=True, ylim=0.01, verbose=True):
        row_num = int(np.ceil(graph_draw_num / col_num))
        i = 0
        print()
        
        if sort:
            losses_train_items = sorted(results_losses.items(), key=lambda x:x[1][-1], reverse=False)
        else:
            losses_train_items = results_losses.items()
        
        for key, one_losses in losses_train_items:
            one_loss = ("%.2f" % one_losses[-1]).rjust(5) ### str().rjust(5)
            if verbose: print(f"{'B' if sort else 'T'}est-{str(i+1).ljust(2)} (val acc:{one_loss}%) | " + key)
            
            plt.subplot(row_num, col_num, i+1)
            plt.title(key)
            plt.ylim(0, ylim)
            # if i % col_num: plt.yticks([])
            if i < graph_draw_num-col_num: plt.xticks([])
            x = np.arange(len(one_losses))
            y = smooth_curve(one_losses) if smooth else one_losses
            plt.plot(x, y, label="loss")
            
            i += 1
            if i >= graph_draw_num:
                break
        
        plt.show()

################################################################################################################################

    def accuracy_graph(accs, ylim_min=None):
        markers = {'train': 'o', 'test': 's'}
        x = np.arange(len(accs))
        if max(accs) <= 1: 
            accs = np.array(accs)*100
        plt.plot(x, accs, f"-", label='test acc')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(ylim_min, 100)
        plt.legend(loc='lower right') # 그래프 이름 표시
        plt.show()


    def accuracy_graphs(accs_dic, ylim_min=None, colors=('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')):
        markers = {'train': 'o', 'test': 's'}
        keys = list(accs_dic.keys())
        x = np.arange(len(accs_dic[keys[0]]))
        for key, color in zip(keys, colors):
            if max(accs_dic[key]) <= 1: 
                accs_dic[key] = np.array(accs_dic[key])*100
            plt.plot(x, accs_dic[key], f"-", label=key)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(ylim_min, 100)
        plt.legend(loc='lower right') # 그래프 이름 표시
        plt.show()
    

    def many_accuracy_graphs(results_train, results_val, graph_draw_num=9, col_num=3, sort=True, verbose=True):
        row_num = int(np.ceil(graph_draw_num / col_num))
        i = 0
        print()
        
        if sort:
            results_val_items = sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True)
        else:
            results_val_items = results_val.items()
        
        for key, val_accs in results_val_items:
            val_acc = ("%.2f" % val_accs[-1]).rjust(5) ### str().rjust(5)
            if verbose: print(f"{'B' if sort else 'T'}est-{str(i+1).ljust(2)} (val acc:{val_acc}%) | " + key)
            if max(val_accs) <= 1: 
                val_accs = np.array(val_accs)*100
            plt.subplot(row_num, col_num, i+1)
            plt.title(key)
            plt.ylim(0, 100)
            # if i % col_num: plt.yticks([])
            if i < graph_draw_num-col_num: plt.xticks([])
            x = np.arange(len(val_accs))
            plt.plot(x, val_accs)
            plt.plot(x, results_train[key], "--")
            i += 1
            
            if i >= graph_draw_num:
                break
        
        plt.show()
    
################################################################################################################################

    def imgs_show(img, nx='auto', filter_show=False, margin=3, scale=10, title_info=[], text_info=[], dark_mode=True, 
        adjust={'l':0, 'r':1, 'b':0.02, 't':0.98, 'hs':0.05, 'ws':0.02}):
        """
        c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
        """
        FN, C, _, _ = img.shape
        imgs_num = FN if not filter_show else C
        if imgs_num == 1:
            nx = 1
        elif nx == 'auto': 
            nx = math.ceil(math.sqrt(imgs_num*108/192)/108*192)
            # print(nx)
        ny = int(np.ceil(imgs_num / nx))
        l, r, b, t, hs, ws = adjust['l'], adjust['r'], adjust['b'], adjust['t'], adjust['hs'], adjust['ws']
    
        fig = plt.figure()
        fig.subplots_adjust(left=l, right=r, bottom=b, top=t, hspace=hs, wspace=ws)
        cmap='gray' if dark_mode else plt.cm.gray_r

        for i in range(imgs_num):
            ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
            if title_info!=[]:
                plt.title(f'\n{title_info[i]}' if title_info!='' else f'{i+1}')
            if text_info!=[]:
                info = text_info[i].split(' | ')
                ax.text(0, 0, info[0], ha="left", va="top", fontsize=16, color='white' if dark_mode else 'black')
                fig.canvas.draw()
                ax.text(18, 25, info[-1], ha="left", va="top", fontsize=11, color='white' if dark_mode else 'black')
                fig.canvas.draw()
                
            ax.imshow(img[i, 0] if not filter_show else img[0, i], cmap=cmap, interpolation='nearest')

        plt.show()


    def all_filters_show(network, nx='auto'):
        for key in network.params.keys():
            if network.params[key].ndim == 4:
                plot.imgs_show(network.params[key], filters_name=key, nx=nx)

################################################################################################################################

    def compare_filter(filters1, filters2, filters_name, nx=10):
        FN, C, FH, FW = filters1.shape
        ny = int(np.ceil(FN / nx)) * 2
    
        fig = plt.figure()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95, hspace=0.3, wspace=0.3)
    
        for i in range(FN):
            line = i // nx
            ax = fig.add_subplot(ny, nx, nx*line+i+1, xticks=[], yticks=[])
            plt.title(f"{filters_name} ({i+1})")
            ax.imshow(filters1[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
        
            ax = fig.add_subplot(ny, nx, nx*line+i+1+nx, xticks=[], yticks=[])
            plt.title(f"↓")
            ax.imshow(filters2[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.show()
    

    def all_filters_compare(network, pkl_name, nx=5):
        params0 = {} ### 복사 안하고 가리키면 가리킨 곳이 바뀔때 같이 변해버림
        for key, value in network.params.items():
            params0[key] = value
        network.load_params(pkl_name)
        
        for key in params0.keys():
            if params0[key].ndim == 4:
                plot.compare_filter(params0[key], network.params[key], key, nx=nx)

################################################################################################################################

    def activation_value_distribution(activation_values, ylim=None):
        plt.figure(figsize=(3*len(activation_values),5))
        plt.subplots_adjust(left=0.05, right=0.98, wspace=0.3)
        
        for idx, (key, a) in enumerate(activation_values.items()):
            plt.subplot(1, len(activation_values), idx+1)
            plt.title(key+'\nactivation')
            # if i != 0: plt.yticks([], [])
            plt.xlim(0, 1)
            plt.ylim(0, ylim) if ylim!=None else ()
            plt.hist(a.flatten(), 30, range=(0,1))
            
        plt.show()


    def compare_activation_value_distribution(activation_values_list, pkls_name, ylim=None):
        # plt.figure(figsize=(3*len(activation_values),5))
        plt.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.95, wspace=0.2)

        for i_list, activation_values in enumerate(activation_values_list):
            title = " ".join(pkls_name[i_list].split(" ")[-7:-5])
            for i, a in activation_values.items():
                plt.subplot(len(activation_values_list), len(activation_values), i_list*len(activation_values)+i+1)
                if i_list==0:
                    plt.title(str(i+1) + f"-activation\n{title}")
                else:
                    plt.title(f"{title}")
                if i_list != len(activation_values_list)-1: plt.xticks([], [])
                plt.xlim(0, 1)
                plt.ylim(0, ylim) if ylim!=None else ()
                plt.hist(a.flatten(), 30, range=(0,1))
        
        plt.show()

################################################################################################################################

    def set_font(font_path):
        import matplotlib.font_manager as fm

        font_name = fm.FontProperties(fname=font_path, size=50).get_name()
        plt.rc('font', family=font_name)
