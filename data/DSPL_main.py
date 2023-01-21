import warnings
warnings.filterwarnings("ignore")
import torch
from model import config
import torch.optim as optim
import pickle as pkl
print('torch.cuda.is_available()',torch.cuda.is_available())
args = config.Arguments()
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
print('device',device)
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
from model import data_laoder,Separate_ASD, LSTM_Model,test_method,Source_model_train_IJCNN
from model import Mutual_train as Mutual_train_3
import numpy as np
import random
from sklearn.model_selection import train_test_split
from model import supcontra
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
print('device',device)
soft =True
from model import Supplement_Abide as Supplement


def obtain_AbideII_source_target(ADHD_DATA, ADHD_LABEL, ADHD_PHENO, site):
    feature = ADHD_DATA
    a = np.where(site == 1)[0]
    b = np.where(site == 3)[0]
    c = np.where(site == 6)[0]
    d = np.where(site == 8)[0]
    e = np.where(site == 9)[0]
    f = np.where(site == 13)[0]
    g = np.where(site == 14)[0]
    client_1 = np.concatenate([a, b,c,d,e,f,g], axis=0)
    all_index = [i for i in range(len(np.unique(site)))]
    for i in [1,3,6,8,9,13,14]:
        all_index.remove(i)
    www=[]
    for i in all_index:
        www.append(np.where(site == i)[0])
    client_2 =www[0]
    for k in range(len(www)-1):
        client_2=np.concatenate([client_2,www[k+1]])
    data_index = [client_1, client_2]
    site_feature = []
    site_label = []
    site_pheno = []
    #     print(data_index)
    for i in data_index:
        site_feature.append(feature[i])
        site_label.append(ADHD_LABEL[i])
        site_pheno.append(ADHD_PHENO[i])
    return site_feature, site_label, site_pheno

def obtain_ADHD_data(ADHD_DATA, ADHD_LABEL, ADHD_PHENO, ADHD_SITE):
    #     ADHD_SITE = np.load('Data/ADHD_site_whole.npy')
    #     ADHD_DATA = np.load('Data/ADHD_data_whole.npy', allow_pickle=True)
    #     ADHD_LABEL = np.load('Data/ADHD_label_whole.npy', allow_pickle=True)
    feature = ADHD_DATA
    site_index = np.unique(ADHD_SITE).tolist()
    a = np.where(ADHD_SITE == 4)[0]
    b = np.where(ADHD_SITE == 5)[0]
    c = np.where(ADHD_SITE == 1)[0]
    d = np.where(ADHD_SITE == 2)[0]
    e = np.where(ADHD_SITE == 6)[0]
    site_2 = np.concatenate([c, d, e], axis=0)
    site_3 = np.concatenate([a, b], axis=0)
    site_1 = np.where(ADHD_SITE == 3)[0]
    site_t = np.where(ADHD_SITE == 0)[0]
    #     print('ADHD',site_1.shape,site_2.shape,site_3.shape,site_t.shape)
    client_index = [site_1, site_2, site_3, site_t]
    sequence_1 = client_index[0]  ###263
    sequence_2 = client_index[1]  ###228
    sequence_3 = client_index[2]  ###203
    sequence_t = client_index[3]  ###245
    print('sequence_1', len(sequence_1))
    print('sequence_2', len(sequence_2))
    print('sequence_3', len(sequence_3))
    print('sequence_t', len(sequence_t))
    tar_index = np.concatenate([sequence_1, sequence_2], axis=0)
    sor_index = np.concatenate([sequence_t, sequence_3], axis=0)

    data_index = [tar_index, sor_index]
    site_feature = []
    site_label = []
    site_pheno = []
    #     print(data_index)
    for i in data_index:
        site_feature.append(feature[i])
        site_label.append(ADHD_LABEL[i])
        site_pheno.append(ADHD_PHENO[i])
    return site_feature, site_label, site_pheno


class client_data(Dataset):
    def  __init__(self,data, aug_data,label, pheno, fname, transforms=None):
        self.D_client0 = data
        self.D_client1 = aug_data
        self.D_label = label
        self.pheno = pheno
        self.fname = fname
        self.transforms = transforms
    def __getitem__(self, index):
        img_0 = self.D_client0[index]
        img_1 = self.D_client1[index]
        label = self.D_label[index]
        pheno_0 = self.pheno[index]
        fname = self.fname[index]
        # if self.transforms is not None:
        #     data = self.transforms(data)
        return (img_0, img_1, label,pheno_0,fname)
    def __len__(self):
        return len(self.D_client1)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def generate_augdata(data, Cond):
    if Cond==True:
        a = np.random.randn(data.shape[0],data.shape[1],data.shape[2],data.shape[3])/5
    else:
        a = np.random.randn(data.shape[0], data.shape[1], data.shape[2]) / 5
    aug_data=data+a
    aug_data=torch.tensor(aug_data)
    return aug_data

def generate_agu_tensor(X_train_ASD_s1,y_train_ASD_s1,pheno_ASD_s1,file_name,Cond):
    X_train_ASD_s1 = X_train_ASD_s1.type(torch.FloatTensor)
    y_train_ASD_s1 = y_train_ASD_s1.type(torch.FloatTensor)
    pheno_ASD_s1 = pheno_ASD_s1.type(torch.FloatTensor)
    data_tensor_s1 = generate_augdata(X_train_ASD_s1,Cond)
    feature_tensor_s1 = torch.Tensor(X_train_ASD_s1)
    label_tensor_s1 = torch.Tensor(y_train_ASD_s1)
    dataset_tensor = client_data(feature_tensor_s1, data_tensor_s1, label_tensor_s1,pheno_ASD_s1, file_name)
    #combined_loader = DataLoader(dataset_tensor,batch_size=64,shuffle=False)
    return dataset_tensor

def global_test(model, device, test_loader,num_test_samples):
    model.to(device)
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data,_,target,pheno,__) in enumerate(test_loader):
            data = data.type(torch.FloatTensor)
            target = target.type(torch.long)
            pheno = pheno.type(torch.FloatTensor)
            data, target, pheno = data.to(device), target.to(device),pheno.to(device)
            pre_label,_= model(data, pheno)
            #loss_2 = F.nll_loss(pre_label, target)
            pred = pre_label.argmax(1, keepdim=True) # get the index of the max log-probability
            # print('pred',pred)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print('correct_num:',correct)
    correct_rate = correct / num_test_samples
    return correct_rate

def one_hot(labels,num_classes):
    one_hot_codes = np.eye(num_classes)
    one_hot_labels = []
    for label in labels:
        # 将连续的整型值映射为one_hot编码
        one_hot_label = one_hot_codes[label]
        one_hot_labels.append(one_hot_label)

    one_hot_labels = np.array(one_hot_labels)
    return one_hot_labels
    # print(one_hot_labels)

def load_data(Con2D, data_name):
    if data_name == 'ASD':
        site = np.load('data/ASD_site.npy')
        label = np.load('data/ASD_labels.npy', allow_pickle=True)
        pheno_temp = np.load('data/ASD_pheno.npy', allow_pickle=True)
        pheno = pheno_temp.squeeze()
        print('pheno', pheno.shape)
        if Con2D == False:
            data = np.load('data/ASD_feats.npy', allow_pickle=True)
        else:
            data = np.load('data/ASD_feats.npy', allow_pickle=True)
            data_ = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))
            data_ = np.concatenate([data_, data_, data_], 3)
            data = data_.transpose(0, 3, 1, 2).astype(np.float32)
        print('data_shape',data.shape)
        a, b, c = Separate_ASD.obtain_ASD_data(data, label, pheno, site)
        source_data = np.array(a[0])
        target_data = np.array(a[1])
        source_label = np.array(b[0])
        target_label = np.array(b[1])
        source_pheno = np.array(c[0])
        target_pheno = np.array(c[1])
    elif data_name == 'Abide_II_global':
        site = np.load('data/ASD_site_abide_ii_global.npy')
        label = np.load('data/ASD_labels_abide_ii_global.npy', allow_pickle=True)
        pheno_temp = np.load('data/ASD_Abide_II_pheno_global.npy', allow_pickle=True)
        pheno = pheno_temp.squeeze()
        # pheno = np.transpose(pheno)
        #     print('pheno', pheno.shape)
        if Con2D == False:
            data = np.load('data/ASD_feats_abide_ii_global.npy', allow_pickle=True)
        else:
            data = np.load('data/ASD_feats_abide_ii_global.npy', allow_pickle=True)
            data_ = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))
            data_ = np.concatenate([data_, data_, data_], 3)
            data = data_.transpose(0, 3, 1, 2).astype(np.float32)
        #     print('data_shape',data.shape)
        a, b, c = obtain_AbideII_source_target(data, label, pheno, site)
        source_data = np.array(a[0])
        target_data = np.array(a[1])
        source_label = np.array(b[0])
        target_label = np.array(b[1])
        source_pheno = np.array(c[0])
        target_pheno = np.array(c[1])
    elif data_name == 'Abide_II_non_global':
        site = np.load('data/ASD_site_abide_ii_non_global.npy')
        label = np.load('data/ASD_labels_abide_ii_non_global.npy', allow_pickle=True)
        pheno_temp = np.load('data/ASD_Abide_II_pheno_non_global.npy', allow_pickle=True)
        pheno = pheno_temp.squeeze()
        # pheno = np.transpose(pheno)
        #     print('pheno', pheno.shape)
        if Con2D == False:
            data = np.load('data/ASD_feats_abide_ii_non_global.npy', allow_pickle=True)
        else:
            data = np.load('data/ASD_feats_abide_ii_non_global.npy', allow_pickle=True)
            data_ = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))
            data_ = np.concatenate([data_, data_, data_], 3)
            data = data_.transpose(0, 3, 1, 2).astype(np.float32)
        #     print('data_shape',data.shape)
        a, b, c = obtain_AbideII_source_target(data, label, pheno, site)
        source_data = np.array(a[0])
        target_data = np.array(a[1])
        source_label = np.array(b[0])
        target_label = np.array(b[1])
        source_pheno = np.array(c[0])
        target_pheno = np.array(c[1])
    else:
        site = np.load('data/ADHD_site_whole.npy')
        label = np.load('data/ADHD_labels.npy', allow_pickle=True)
        pheno_temp = np.load('data/ADHD_pheno.npy', allow_pickle=True)
        pheno = pheno_temp.squeeze()
        # pheno = np.transpose(pheno)
        #     print('pheno', pheno.shape)
        if Con2D == False:
            data = np.load('data/ADHD_feats.npy', allow_pickle=True)
        else:
            data = np.load('data/ADHD_feats.npy', allow_pickle=True)
            data_ = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))
            data_ = np.concatenate([data_, data_, data_], 3)
            data = data_.transpose(0, 3, 1, 2).astype(np.float32)
        #     print('data_shape',data.shape)
        a, b, c = obtain_ADHD_data(data, label, pheno, site)
        source_data = np.array(a[0])
        target_data = np.array(a[1])
        source_label = np.array(b[0])
        target_label = np.array(b[1])
        source_pheno = np.array(c[0])
        target_pheno = np.array(c[1])
    return source_data,target_data,source_label,target_label,source_pheno,target_pheno



def data_augment(data,label,pheno,crops):
    augment_label = []
    augment_data = []
    augmented_pheno = []
    sk_data = []
    sk_label = []
    for i in range(len(data)):
        max = data[i].shape[0]
        if max>=100:
            sk_data.append(data[i])
            sk_label.append(label[i])

            range_list = range(90+1 , int(max))
            random_index = random.sample(range_list, crops)
            for j in range(crops):
                r = random_index[j]
                augment_data.append(data[i][r - 90:r])
                augment_label.append(label[i])
                augmented_pheno.append(pheno[i])



    return np.array(augment_data),np.array(augment_label),np.array(augmented_pheno), np.array(sk_data),np.array(sk_label)

if __name__ =='__main__':
    Cond = False
    crops = 10
    data_name = 'ASD'
    source_data, target_data, source_label, target_label, source_pheno, target_pheno = load_data(Cond,data_name)


    data=np.concatenate([source_data,target_data],axis=0)
    label = np.concatenate([source_label, target_label], axis=0)
    pheno = np.concatenate([source_pheno, target_pheno], axis=0)
    print('data',data.shape)
    print('label',label.shape)
    print('pheno',pheno.shape)

    import scipy.io as scio

    if data_name == 'Abide_II_non_global':
        roi_index = scio.loadmat('data/ADHD_cc200_label_index.mat')
        roi_index = roi_index['index'][0] - 1
        data = data[:, :, roi_index]

    Accuracy = []
    mutual_generated_avg = []
    dir_generated_avg = []
    mutual_gene_cor_avg_nnum=[]
    dir_gene_cor_avg_nnum = []

    # criterion = CE_Smooth.SoftEntropy()
    clu_criterion = supcontra.SupConLoss()
    accuracy_mean = []
    accuracy_sub_mean = []
    accuracy_sub2_mean = []
    criterion = torch.nn.BCELoss()
    CV_count = 5
    ww2l_avg_test, ww2l_avg_test_bes, final_avg_test, unlabel_accu_avg, direct_train_avg_0 = [], [], [], [],[]
    direct_train_avg_0 = []
    direct_train_avg_1 = []
    direct_train_avg_2 = []
    direct_train_avg_3 = []
    direct_train_avg_4 = []
    direct_train_avg_5 = []
    direct_train_avg_6 = []
    direct_train_avg_7 = []
    direct_train_avg_8 = []
    direct_train_avg_9 = []
    direct_train_avg_10 = []
    direct_train_avg_11 = []
    direct_train_avg_12 = []
    direct_train_avg_13 = []
    direct_train_avg_14 = []
    direct_train_avg_15 = []
    direct_train_avg_16 = []
    direct_train_avg_17 = []
    direct_train_avg_18 = []
    direct_train_avg_19 = []



    pseudo_avg_0 = []
    pseudo_avg_1 = []
    pseudo_avg_2 = []
    pseudo_avg_3 = []
    pseudo_avg_4 = []
    pseudo_avg_5 = []
    pseudo_avg_6 = []
    pseudo_avg_7 = []
    pseudo_avg_8 = []
    pseudo_avg_9 = []
    pseudo_avg_10 = []
    pseudo_avg_11 = []
    pseudo_avg_12 = []
    pseudo_avg_13 = []
    pseudo_avg_14 = []
    pseudo_avg_15 = []
    pseudo_avg_16 = []
    pseudo_avg_17 = []
    pseudo_avg_18 = []
    pseudo_avg_19 = []


    for i_count in range(CV_count):
        mutual_generated_num = []
        dir_generated_num = []
        mutual_gene_cor_nnum = []
        dir_gene_cor_nnum = []
        ADHD_bert = []
        ADHD_sub_bert = []
        ADHD_sub_bert2 = []
        j_count = 0
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2 ** (i_count + 4))
        test_accu_avg = []
        test_accu_ww2l = []
        test_accu_ww2l_bes = []
        unlabel_accu_list = []
        direct_train_0,direct_train_1,direct_train_2,direct_train_3,direct_train_4 = [],[],[],[],[]
        direct_train_6, direct_train_7, direct_train_8, direct_train_9, direct_train_10 = [], [], [], [], []
        direct_train_13, direct_train_14, direct_train_15, direct_train_16, direct_train_11 = [], [], [], [], []
        direct_train_17, direct_train_18, direct_train_19, direct_train_5, direct_train_12 = [], [], [], [], []


        pseudo_0 = []
        pseudo_1 = []
        pseudo_2 = []
        pseudo_3 = []
        pseudo_4 = []
        pseudo_5 = []
        pseudo_6 = []
        pseudo_7 = []
        pseudo_8 = []
        pseudo_9 = []
        pseudo_10 = []
        pseudo_11 = []
        pseudo_12 = []
        pseudo_13 = []
        pseudo_14 = []
        pseudo_15 = []
        pseudo_16 = []
        pseudo_17 = []
        pseudo_18 = []
        pseudo_19 = []


        for train_index, test_index in kf.split(data, label):
            count = 0
            aug_x_train_list = []
            aug_x_test_list = []
            for i in range(data.shape[0]):
                max = data[i].shape[0]
                if max>=100:
                    if i in train_index:
                        for k in range(crops):
                            aug_x_train_list.append(count)
                            count = count + 1
                    else:
                        for k in range(crops):
                            aug_x_test_list.append(count)
                            count = count + 1
            # print('aug_x_train_list',len(aug_x_train_list))
            # print('aug_x_test_list', len(aug_x_test_list))
            x_train = data[aug_x_train_list]
            y_train = label[aug_x_train_list]
            x_train_pheno = pheno[aug_x_train_list].reshape(-1, 4)

            x_test_ = data[aug_x_test_list]
            y_test_ = label[aug_x_test_list]
            x_test_pheno_ = pheno[aug_x_test_list].reshape(-1, 4)
            x_test, y_test, x_test_pheno, _, _ = data_augment(x_test_,
                                                                    y_test_,
                                                                    x_test_pheno_,
                                                                    crops)

            index = [i for i in range(x_train.shape[0])]
            random.shuffle(index)

            x_train = x_train[index]
            x_train_pheno = x_train_pheno[index]
            y_train = y_train[index]

            max_seq_len = x_test.shape[1]
            embedding_size = x_test.shape[2]
            # print('max_seq_len',max_seq_len)
            # print('embedding_size',embedding_size)


            X_train_t_unlabeled_, X_train_t_labeled_, y_train_t_unlabeled_, y_train_t_labeled_, pheno_train_unlabeled_, pheno_train_labeled_ = train_test_split(x_train, y_train, x_train_pheno, random_state=777,
                                                                    test_size=0.25)
            X_train_t_unlabeled, y_train_t_unlabeled, pheno_train_unlabeled, _, _ = data_augment(X_train_t_unlabeled_, y_train_t_unlabeled_, pheno_train_unlabeled_,
                                                                            crops)
            X_train_t_labeled, y_train_t_labeled, pheno_train_labeled, _, _ = data_augment(X_train_t_labeled_, y_train_t_labeled_, pheno_train_labeled_,
                                                                            crops)
        
            tar_labeled_data_train = torch.tensor(X_train_t_labeled)
            tar_labeled_label_train = torch.tensor(y_train_t_labeled)
            tar_unlabeled_data_train = torch.tensor(X_train_t_unlabeled)
            print('tar_unlabeled_data_train',tar_unlabeled_data_train.shape)
            print('tar_labeled_label_train',tar_labeled_label_train.shape)
            tar_unlabeled_label_train = torch.tensor(y_train_t_unlabeled)
            tar_data_test = torch.tensor(x_test)
            tar_label_test = torch.tensor(y_test)
            tar_unlabeled_pheno = torch.tensor(pheno_train_unlabeled)
            tar_labeled_pheno = torch.tensor(pheno_train_labeled)
            tar_pheno_test = torch.tensor(x_test_pheno)


            tar_unlabeled_train_shape = tar_unlabeled_data_train.shape[0]
            tar_labeled_train_shape = tar_labeled_data_train.shape[0]
            tar_test_shape = tar_data_test.shape[0]

            tar_unlabeled_train_file_values = tar_unlabeled_train_shape
            tar_labeled_train_file_values = tar_unlabeled_train_file_values + tar_labeled_train_shape
            tar_test_file_values = tar_labeled_train_file_values + tar_test_shape

            tar_unlabeled_train_file_name = np.arange(0,tar_unlabeled_train_file_values)
            tar_labeled_train_file_name = np.arange(tar_unlabeled_train_file_values,tar_labeled_train_file_values)
            tar_test_file_name = np.arange(tar_labeled_train_file_values, tar_test_file_values)


            tar_labeled_dataset = generate_agu_tensor(tar_labeled_data_train, tar_labeled_label_train,tar_labeled_pheno,tar_labeled_train_file_name,Cond)
            tar_labeled_loader = DataLoader(tar_labeled_dataset,batch_size=args.batch_size,shuffle=False)


            tar_unlabeled_dataset = generate_agu_tensor(tar_unlabeled_data_train, tar_unlabeled_label_train,tar_unlabeled_pheno,tar_unlabeled_train_file_name,Cond)
            tar_unlabeled_loader = DataLoader(tar_unlabeled_dataset,batch_size=args.batch_size,shuffle=False)


            tar_test_dataset = generate_agu_tensor(tar_data_test, tar_label_test, tar_pheno_test, tar_test_file_name,Cond)
            tar_test_loader = DataLoader(tar_test_dataset, batch_size=args.batch_size, shuffle=False)
            if data_name == 'Abide_II_global':
                model_tar_init = LSTM_Model.LSTM(190, 32, 1).to(device)
            elif data_name == 'Abide_II_non_global':
                model_tar_init = LSTM_Model.LSTM(190, 32, 1).to(device)
            else:
                model_tar_init = LSTM_Model.LSTM(190, 32, 1).to(device)  # model built on target domain! D_CNN_Network

            tar_optimizer = optim.Adam(model_tar_init.parameters(), lr=0.0001,weight_decay=1e-6)
            model_tar_init = Source_model_train_IJCNN.ini_target(args, model_tar_init, device, tar_labeled_loader, tar_optimizer,criterion)
            ASD_accu, ASD_sub_acc, ASD_sub2_acc,SPE,REC,AUC = test_method.LSTM_test_method(model_tar_init, device,tar_test_loader, crops)
            print(' ini_tar_accu', ASD_accu, ASD_sub_acc, ASD_sub2_acc,SPE,REC,AUC)

            model_tar_init.cpu()
            pretrained_dict = model_tar_init.state_dict()
            torch.cuda.empty_cache()
            if data_name == 'Abide_II_global':
                tar_mutual_net = LSTM_Model.Mutual_LSTM(190, 32, 1).to(device)
            elif data_name == 'Abide_II_non_global':
                tar_mutual_net = LSTM_Model.Mutual_LSTM(190, 32, 1).to(device)
            else:
                tar_mutual_net = LSTM_Model.Mutual_LSTM(190, 32, 1).to(device)

            model_dict = tar_mutual_net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict}  
            model_dict.update(pretrained_dict)  
            tar_mutual_net.load_state_dict(model_dict)  
            tar_mutual_net.to(device)


            if soft ==True:
                aug_datas_0, aug_datas_1, aug_labels, aug_pheno,dir_num,dir_cor_num = Supplement.soft_pseudo_generation_with_weighted_selection(tar_mutual_net,
                                                                                                      tar_unlabeled_loader,
                                                                                                      2)
            else:
                aug_datas_0, aug_datas_1, aug_labels, aug_pheno, dir_num, dir_cor_num = Supplement.pseudo_generation_with_weighted_selection(
                    tar_mutual_net,
                    tar_unlabeled_loader,
                    2)
            dir_generated_num.append(dir_num)
            dir_gene_cor_nnum.append(dir_cor_num)

            pseudo_dataset = data_laoder.unlabelled_client_data(list(aug_datas_0.values()), list(aug_datas_1.values()),
                                                list(aug_labels.values()), list(aug_pheno.values()),list(aug_labels.keys()))

            pseudo_loader = DataLoader(combine_dataset, batch_size=args.batch_size, shuffle=False)
            combine_dataset = torch.utils.data.ConcatDataset([pseudo_dataset, tar_labeled_dataset])
            combined_loader = DataLoader(combine_dataset, batch_size=args.batch_size, shuffle=False)
            for num in range(args.generate_epoch):
                if num==0:
                    if data_name == 'Abide_II_global' :
                        model_tar_mutual = LSTM_Model.Mutual_LSTM(190, 32, 1).to(device)
                    elif data_name== 'Abide_II_non_global':
                        model_tar_mutual = LSTM_Model.Mutual_LSTM(190, 32, 1).to(device)
                    else:
                        model_tar_mutual = LSTM_Model.Mutual_LSTM(190, 32, 1).to(device)
                    mutual_optimizer = optim.SGD(model_tar_mutual.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
                    model_tar_mutual.train()
                    model_tar_mutual = Mutual_train_3.Mutual_train(150, combined_loader, model_tar_mutual, mutual_optimizer)

                    unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,unlabel_load_SPE,unlabel_load_REC,unlabel_load_AUC = test_method.Mutual_LSTM_test_method(model_tar_mutual, device, tar_unlabeled_loader,
                                                              crops)
                    print('unlabeled_data', unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc)
                    


                    if soft ==True:
                        aug_datas_0, aug_datas_1, aug_labels, aug_pheno, mutual_num,mutual_cor_num = Supplement.soft_pseudo_generation_with_weighted_selection_te(
                        model_tar_mutual,
                        tar_unlabeled_loader, 2)
                    else:
                        aug_datas_0, aug_datas_1, aug_labels, aug_pheno, mutual_num, mutual_cor_num = Supplement.pseudo_generation_with_weighted_selection_te(
                            model_tar_mutual,
                            tar_unlabeled_loader, 2)
                    print('generate_pesudo_labels', str(num), mutual_cor_num, mutual_num)

                    pseudo_dataset_ = data_laoder.unlabelled_client_data(list(aug_datas_0.values()), list(aug_datas_1.values()),
                                                                         list(aug_labels.values()), list(aug_pheno.values()),
                                                                         list(aug_labels.keys()))
                    combine_dataset_temp = torch.utils.data.ConcatDataset([pseudo_dataset_, tar_labeled_dataset])
                    combined_loader_temp = DataLoader(combine_dataset_temp, batch_size=args.batch_size, shuffle=True)
                    if data_name == 'Abide_II_global':
                        t_models_4 = LSTM_Model.LSTM(190, 32, 1).to(device)
                    elif data_name == 'Abide_II_non_global':
                        t_models_4 = LSTM_Model.LSTM(190, 32, 1).to(device)
                    else:
                        t_models_4 = LSTM_Model.LSTM(190, 32, 1).to(device)

                    t_optimizer_4 = optim.Adam(t_models_4.parameters(), lr=0.0001, weight_decay=1e-6)
                    t_model_4 = Source_model_train_IJCNN.ini_target(args, t_models_4, device,
                                                                                   combined_loader_temp, t_optimizer_4,
                                                                                   criterion)

                    ASD_accu, ASD_sub_acc, ASD_sub2_acc, ASD_SPE, ASD_REC, ASD_AUC = test_method.LSTM_test_method(
                        t_model_4, device,
                        tar_test_loader, crops)
                    print('tar_accu_trained_by_combined_data', num, ASD_accu, ASD_sub_acc, ASD_sub2_acc, ASD_SPE,
                          ASD_REC, ASD_AUC)

                    del t_model_4, t_optimizer_4
                    torch.cuda.empty_cache()
                    direct_train_0.append([ASD_accu, ASD_sub_acc, ASD_sub2_acc, ASD_SPE, ASD_REC, ASD_AUC])
                    pseudo_0.append([unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,unlabel_load_SPE,unlabel_load_REC,unlabel_load_AUC])
                else:
                    if data_name == 'Abide_II_global' :
                        model_tar_mutual = LSTM_Model.Mutual_LSTM(190, 32, 1).to(device)
                    elif data_name== 'Abide_II_non_global':
                        model_tar_mutual = LSTM_Model.Mutual_LSTM(190, 32, 1).to(device)
                    else:
                        model_tar_mutual = LSTM_Model.Mutual_LSTM(190, 32, 1).to(device)
                    mutual_optimizer = optim.SGD(model_tar_mutual.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
                    model_tar_mutual.train()
                    model_tar_mutual = Mutual_train_3.Mutual_train(150, combined_loader_temp, model_tar_mutual, mutual_optimizer,tar_unlabeled_loader,tar_test_loader)
                    unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,unlabel_load_SPE,unlabel_load_REC,unlabel_load_AUC = test_method.Mutual_LSTM_test_method(model_tar_mutual, device, tar_unlabeled_loader,
                                                              crops)
                    print('unlabeled_data', unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc)

                    if soft ==True:
                        aug_datas_0, aug_datas_1, aug_labels, aug_pheno, mutual_num,mutual_cor_num = Supplement.soft_pseudo_generation_with_weighted_selection_te(
                        model_tar_mutual,
                        tar_unlabeled_loader, 2)
                    else:
                        aug_datas_0, aug_datas_1, aug_labels, aug_pheno, mutual_num, mutual_cor_num = Supplement.pseudo_generation_with_weighted_selection_te(
                            model_tar_mutual,
                            tar_unlabeled_loader, 2)
                    print('generate_pesudo_labels',str(num),mutual_cor_num,mutual_num)
                    if num == args.generate_epoch-1:
                        mutual_generated_num.append(mutual_num)
                        mutual_gene_cor_nnum.append(mutual_cor_num)
                    pseudo_dataset_ = data_laoder.unlabelled_client_data(list(aug_datas_0.values()), list(aug_datas_1.values()),
                                                                         list(aug_labels.values()), list(aug_pheno.values()),
                                                                         list(aug_labels.keys()))
                    combine_dataset_temp = torch.utils.data.ConcatDataset([pseudo_dataset_, tar_labeled_dataset])
                    combined_loader_temp = DataLoader(combine_dataset_temp, batch_size=args.batch_size, shuffle=True)

                    if data_name == 'Abide_II_global' :
                        t_models_4 = LSTM_Model.LSTM(190, 32, 1).to(device)
                    elif data_name == 'Abide_II_non_global':
                        t_models_4 = LSTM_Model.LSTM(190, 32, 1).to(device)
                    else:
                        t_models_4 = LSTM_Model.LSTM(190, 32, 1).to(device)

                    t_optimizer_4 = optim.Adam(t_models_4.parameters(), lr=0.0001, weight_decay=1e-6)
                    t_model_4 = Source_model_train_IJCNN.ini_target(args, t_models_4, device, combined_loader_temp, t_optimizer_4,criterion)

                    ASD_accu, ASD_sub_acc, ASD_sub2_acc,ASD_SPE,ASD_REC,ASD_AUC = test_method.LSTM_test_method(t_model_4, device,
                                                                                       tar_test_loader, crops)
                    print('tar_accu_trained_by_combined_data', num, ASD_accu, ASD_sub_acc, ASD_sub2_acc,ASD_SPE,ASD_REC,ASD_AUC)


                    del t_model_4,t_optimizer_4
                    torch.cuda.empty_cache()
                    if num==1:
                        direct_train_1.append([ASD_accu, ASD_sub_acc, ASD_sub2_acc,ASD_SPE,ASD_REC,ASD_AUC])
                        pseudo_1.append([unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,
                                         unlabel_load_SPE, unlabel_load_REC, unlabel_load_AUC])
                    if num==2:
                        direct_train_2.append([ASD_accu, ASD_sub_acc, ASD_sub2_acc, ASD_SPE, ASD_REC, ASD_AUC])
                        pseudo_2.append([unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,
                                         unlabel_load_SPE, unlabel_load_REC, unlabel_load_AUC])
                        
                    if num==3:
                        direct_train_3.append([ASD_accu, ASD_sub_acc, ASD_sub2_acc, ASD_SPE, ASD_REC, ASD_AUC])
                        pseudo_3.append([unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,
                                         unlabel_load_SPE, unlabel_load_REC, unlabel_load_AUC])
                        
                        
                    if num==4:
                        direct_train_4.append([ASD_accu, ASD_sub_acc, ASD_sub2_acc, ASD_SPE, ASD_REC, ASD_AUC])
                        pseudo_4.append([unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,
                                         unlabel_load_SPE, unlabel_load_REC, unlabel_load_AUC])
                        
                        
                    if num==5:
                        direct_train_5.append([ASD_accu, ASD_sub_acc, ASD_sub2_acc,ASD_SPE,ASD_REC,ASD_AUC])
                        pseudo_5.append([unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,
                                         unlabel_load_SPE, unlabel_load_REC, unlabel_load_AUC])
                        
                        
                    if num==6:
                        direct_train_6.append([ASD_accu, ASD_sub_acc, ASD_sub2_acc, ASD_SPE, ASD_REC, ASD_AUC])
                        pseudo_6.append([unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,
                                         unlabel_load_SPE, unlabel_load_REC, unlabel_load_AUC])
                        
                        
                    if num==7:
                        direct_train_7.append([ASD_accu, ASD_sub_acc, ASD_sub2_acc, ASD_SPE, ASD_REC, ASD_AUC])
                        pseudo_7.append([unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,
                                         unlabel_load_SPE, unlabel_load_REC, unlabel_load_AUC])
                        
                        
                    if num==8:
                        direct_train_8.append([ASD_accu, ASD_sub_acc, ASD_sub2_acc, ASD_SPE, ASD_REC, ASD_AUC])
                        pseudo_8.append([unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,
                                         unlabel_load_SPE, unlabel_load_REC, unlabel_load_AUC])
                        
                        
                    if num==9:
                        direct_train_9.append([ASD_accu, ASD_sub_acc, ASD_sub2_acc,ASD_SPE,ASD_REC,ASD_AUC])
                        pseudo_9.append([unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,
                                         unlabel_load_SPE, unlabel_load_REC, unlabel_load_AUC])
                        
                        
                    if num==10:
                        direct_train_10.append([ASD_accu, ASD_sub_acc, ASD_sub2_acc, ASD_SPE, ASD_REC, ASD_AUC])
                        pseudo_10.append([unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,
                                         unlabel_load_SPE, unlabel_load_REC, unlabel_load_AUC])
                        
                        
                    if num==11:
                        direct_train_11.append([ASD_accu, ASD_sub_acc, ASD_sub2_acc, ASD_SPE, ASD_REC, ASD_AUC])
                        pseudo_11.append([unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,
                                         unlabel_load_SPE, unlabel_load_REC, unlabel_load_AUC])
                        
                        
                    if num==12:
                        direct_train_12.append([ASD_accu, ASD_sub_acc, ASD_sub2_acc, ASD_SPE, ASD_REC, ASD_AUC])
                        pseudo_12.append([unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,
                                         unlabel_load_SPE, unlabel_load_REC, unlabel_load_AUC])
                        
                        
                    if num==13:
                        direct_train_13.append([ASD_accu, ASD_sub_acc, ASD_sub2_acc,ASD_SPE,ASD_REC,ASD_AUC])
                        pseudo_13.append([unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,
                                         unlabel_load_SPE, unlabel_load_REC, unlabel_load_AUC])
                        
                        
                    if num==14:
                        direct_train_14.append([ASD_accu, ASD_sub_acc, ASD_sub2_acc, ASD_SPE, ASD_REC, ASD_AUC])
                        pseudo_14.append([unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,
                                         unlabel_load_SPE, unlabel_load_REC, unlabel_load_AUC])
                        
                        
                    if num==15:
                        direct_train_15.append([ASD_accu, ASD_sub_acc, ASD_sub2_acc, ASD_SPE, ASD_REC, ASD_AUC])
                        pseudo_15.append([unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,
                                         unlabel_load_SPE, unlabel_load_REC, unlabel_load_AUC])
                        
                        
                    if num==16:
                        direct_train_16.append([ASD_accu, ASD_sub_acc, ASD_sub2_acc, ASD_SPE, ASD_REC, ASD_AUC])
                        pseudo_16.append([unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,
                                         unlabel_load_SPE, unlabel_load_REC, unlabel_load_AUC])
                        
                        
                    if num==17:
                        direct_train_17.append([ASD_accu, ASD_sub_acc, ASD_sub2_acc,ASD_SPE,ASD_REC,ASD_AUC])
                        pseudo_17.append([unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,
                                         unlabel_load_SPE, unlabel_load_REC, unlabel_load_AUC])
                        
                        
                    if num==18:
                        direct_train_18.append([ASD_accu, ASD_sub_acc, ASD_sub2_acc, ASD_SPE, ASD_REC, ASD_AUC])
                        pseudo_18.append([unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,
                                         unlabel_load_SPE, unlabel_load_REC, unlabel_load_AUC])
                        
                        
                    if num==19:
                        direct_train_19.append([ASD_accu, ASD_sub_acc, ASD_sub2_acc, ASD_SPE, ASD_REC, ASD_AUC])
                        pseudo_19.append([unlabel_load_test_accu, unlabel_load_test_sub_acc, unlabel_load_test_sub2_acc,
                                         unlabel_load_SPE, unlabel_load_REC, unlabel_load_AUC])

        print('direct_train_avg_0', np.mean(np.array(direct_train_0),axis=0))
        direct_train_avg_0.append(np.mean(np.array(direct_train_0),axis=0))
        print('direct_train_avg_1', np.mean(np.array(direct_train_1), axis=0))
        direct_train_avg_1.append(np.mean(np.array(direct_train_1), axis=0))
        print('direct_train_avg_2', np.mean(np.array(direct_train_2), axis=0))
        direct_train_avg_2.append(np.mean(np.array(direct_train_2), axis=0))
        print('direct_train_avg_3', np.mean(np.array(direct_train_3), axis=0))
        direct_train_avg_3.append(np.mean(np.array(direct_train_3), axis=0))
        print('direct_train_avg_4', np.mean(np.array(direct_train_4), axis=0))
        direct_train_avg_4.append(np.mean(np.array(direct_train_4), axis=0))

        print('direct_train_avg_5', np.mean(np.array(direct_train_5), axis=0))
        direct_train_avg_5.append(np.mean(np.array(direct_train_5), axis=0))
        print('direct_train_avg_6', np.mean(np.array(direct_train_6), axis=0))
        direct_train_avg_6.append(np.mean(np.array(direct_train_6), axis=0))
        print('direct_train_avg_7', np.mean(np.array(direct_train_7), axis=0))
        direct_train_avg_7.append(np.mean(np.array(direct_train_7), axis=0))
        print('direct_train_avg_8', np.mean(np.array(direct_train_8), axis=0))
        direct_train_avg_8.append(np.mean(np.array(direct_train_8), axis=0))
        print('direct_train_avg_9', np.mean(np.array(direct_train_9), axis=0))
        direct_train_avg_9.append(np.mean(np.array(direct_train_9), axis=0))

        print('direct_train_avg_10', np.mean(np.array(direct_train_10), axis=0))
        direct_train_avg_10.append(np.mean(np.array(direct_train_10), axis=0))
        print('direct_train_avg_11', np.mean(np.array(direct_train_11), axis=0))
        direct_train_avg_11.append(np.mean(np.array(direct_train_11), axis=0))
        print('direct_train_avg_12', np.mean(np.array(direct_train_12), axis=0))
        direct_train_avg_12.append(np.mean(np.array(direct_train_12), axis=0))
        print('direct_train_avg_13', np.mean(np.array(direct_train_13), axis=0))
        direct_train_avg_13.append(np.mean(np.array(direct_train_13), axis=0))
        print('direct_train_avg_14', np.mean(np.array(direct_train_14), axis=0))
        direct_train_avg_14.append(np.mean(np.array(direct_train_14), axis=0))

        print('direct_train_avg_15', np.mean(np.array(direct_train_15), axis=0))
        direct_train_avg_15.append(np.mean(np.array(direct_train_15), axis=0))
        print('direct_train_avg_16', np.mean(np.array(direct_train_16), axis=0))
        direct_train_avg_16.append(np.mean(np.array(direct_train_16), axis=0))
        print('direct_train_avg_17', np.mean(np.array(direct_train_17), axis=0))
        direct_train_avg_17.append(np.mean(np.array(direct_train_17), axis=0))
        print('direct_train_avg_18', np.mean(np.array(direct_train_18), axis=0))
        direct_train_avg_18.append(np.mean(np.array(direct_train_18), axis=0))
        print('direct_train_avg_19', np.mean(np.array(direct_train_19), axis=0))
        direct_train_avg_19.append(np.mean(np.array(direct_train_19), axis=0))




        print('pseudo_0', np.mean(np.array(pseudo_0), axis=0))
        pseudo_avg_0.append(np.mean(np.array(pseudo_0), axis=0))
        print('pseudo_1', np.mean(np.array(pseudo_1), axis=0))
        pseudo_avg_1.append(np.mean(np.array(pseudo_1), axis=0))
        print('pseudo_2', np.mean(np.array(pseudo_2), axis=0))
        pseudo_avg_2.append(np.mean(np.array(pseudo_2), axis=0))
        print('pseudo_3', np.mean(np.array(pseudo_3), axis=0))
        pseudo_avg_3.append(np.mean(np.array(pseudo_3), axis=0))
        print('pseudo_4', np.mean(np.array(pseudo_4), axis=0))
        pseudo_avg_4.append(np.mean(np.array(pseudo_4), axis=0))
        print('pseudo_5', np.mean(np.array(pseudo_5), axis=0))
        pseudo_avg_5.append(np.mean(np.array(pseudo_5), axis=0))
        print('pseudo_6', np.mean(np.array(pseudo_6), axis=0))
        pseudo_avg_6.append(np.mean(np.array(pseudo_6), axis=0))
        print('pseudo_7', np.mean(np.array(pseudo_7), axis=0))
        pseudo_avg_7.append(np.mean(np.array(pseudo_7), axis=0))
        print('pseudo_8', np.mean(np.array(pseudo_8), axis=0))
        pseudo_avg_8.append(np.mean(np.array(pseudo_8), axis=0))
        print('pseudo_9', np.mean(np.array(pseudo_9), axis=0))
        pseudo_avg_9.append(np.mean(np.array(pseudo_9), axis=0))
        print('pseudo_10', np.mean(np.array(pseudo_10), axis=0))
        pseudo_avg_10.append(np.mean(np.array(pseudo_10), axis=0))
        print('pseudo_11', np.mean(np.array(pseudo_11), axis=0))
        pseudo_avg_11.append(np.mean(np.array(pseudo_11), axis=0))
        print('pseudo_12', np.mean(np.array(pseudo_12), axis=0))
        pseudo_avg_12.append(np.mean(np.array(pseudo_12), axis=0))
        print('pseudo_13', np.mean(np.array(pseudo_13), axis=0))
        pseudo_avg_13.append(np.mean(np.array(pseudo_13), axis=0))
        print('pseudo_14', np.mean(np.array(pseudo_14), axis=0))
        pseudo_avg_14.append(np.mean(np.array(pseudo_14), axis=0))
        print('pseudo_15', np.mean(np.array(pseudo_15), axis=0))
        pseudo_avg_15.append(np.mean(np.array(pseudo_15), axis=0))
        print('pseudo_16', np.mean(np.array(pseudo_16), axis=0))
        pseudo_avg_16.append(np.mean(np.array(pseudo_16), axis=0))
        print('pseudo_17', np.mean(np.array(pseudo_17), axis=0))
        pseudo_avg_17.append(np.mean(np.array(pseudo_17), axis=0))
        print('pseudo_18', np.mean(np.array(pseudo_18), axis=0))
        pseudo_avg_18.append(np.mean(np.array(pseudo_18), axis=0))
        print('pseudo_19', np.mean(np.array(pseudo_19), axis=0))
        pseudo_avg_19.append(np.mean(np.array(pseudo_19), axis=0))


        mutual_generated_avg.append(np.mean(np.array(mutual_generated_num), axis=0))
        dir_generated_avg.append(np.mean(np.array(dir_generated_num), axis=0))
        mutual_gene_cor_avg_nnum.append(np.mean(np.array(mutual_gene_cor_nnum), axis=0))
        dir_gene_cor_avg_nnum.append(np.mean(np.array(dir_gene_cor_nnum), axis=0))
    print('dir_generated_cor_num_final_mean', np.mean(np.array(dir_gene_cor_avg_nnum), axis=0))
    print('dir_generated_cor_num_final_max', np.max(np.array(dir_gene_cor_avg_nnum)))
    print('mutual_generated_cor_num_final_mean', np.mean(np.array(mutual_gene_cor_avg_nnum), axis=0))
    print('mutual_generated_cor_num_final_max', np.max(np.array(mutual_gene_cor_avg_nnum)))

    print('dir_generated_num_final_mean', np.mean(np.array(dir_generated_avg), axis=0))
    print('dir_generated_num_final_max', np.array(dir_generated_avg))
    print('mutual_generated_num_final_max', np.mean(np.array(mutual_generated_avg), axis=0))
    print('mutual_generated_num_final_max', np.array(mutual_generated_avg))



    print(' ###### direct_train#####')
    print('direct_train_avg_final_0', np.mean(np.array(direct_train_avg_0), axis=0))
    print('direct_train_max_final_0', np.array(direct_train_avg_0))
    print('direct_train_avg_final_1', np.mean(np.array(direct_train_avg_1), axis=0))
    print('direct_train_max_final_1', np.array(direct_train_avg_1))
    print('direct_train_avg_final_2', np.mean(np.array(direct_train_avg_2), axis=0))
    print('direct_train_max_final_2', np.array(direct_train_avg_2))
    print('direct_train_avg_final_3', np.mean(np.array(direct_train_avg_3), axis=0))
    print('direct_train_max_final_3', np.array(direct_train_avg_3))
    print('direct_train_avg_final_4', np.mean(np.array(direct_train_avg_4), axis=0))
    print('direct_train_max_final_4', np.array(direct_train_avg_4))

    print('direct_train_avg_final_5', np.mean(np.array(direct_train_avg_5), axis=0))
    print('direct_train_max_final_5', np.array(direct_train_avg_5))
    print('direct_train_avg_final_6', np.mean(np.array(direct_train_avg_6), axis=0))
    print('direct_train_max_final_6', np.array(direct_train_avg_6))
    print('direct_train_avg_final_7', np.mean(np.array(direct_train_avg_7), axis=0))
    print('direct_train_max_final_7', np.array(direct_train_avg_7))
    print('direct_train_avg_final_8', np.mean(np.array(direct_train_avg_8), axis=0))
    print('direct_train_max_final_8', np.array(direct_train_avg_8))
    print('direct_train_avg_final_9', np.mean(np.array(direct_train_avg_9), axis=0))
    print('direct_train_max_final_9', np.array(direct_train_avg_9))

    print('direct_train_avg_final_10', np.mean(np.array(direct_train_avg_10), axis=0))
    print('direct_train_max_final_10', np.array(direct_train_avg_10))
    print('direct_train_avg_final_11', np.mean(np.array(direct_train_avg_11), axis=0))
    print('direct_train_max_final_11', np.array(direct_train_avg_11))
    print('direct_train_avg_final_12', np.mean(np.array(direct_train_avg_12), axis=0))
    print('direct_train_max_final_12', np.array(direct_train_avg_12))
    print('direct_train_avg_final_13', np.mean(np.array(direct_train_avg_13), axis=0))
    print('direct_train_max_final_13', np.array(direct_train_avg_13))
    print('direct_train_avg_final_14', np.mean(np.array(direct_train_avg_14), axis=0))
    print('direct_train_max_final_14', np.array(direct_train_avg_14))

    print('direct_train_avg_final_15', np.mean(np.array(direct_train_avg_15), axis=0))
    print('direct_train_max_final_15', np.array(direct_train_avg_15))
    print('direct_train_avg_final_16', np.mean(np.array(direct_train_avg_16), axis=0))
    print('direct_train_max_final_16', np.array(direct_train_avg_16))
    print('direct_train_avg_final_17', np.mean(np.array(direct_train_avg_17), axis=0))
    print('direct_train_max_final_17', np.array(direct_train_avg_17))
    print('direct_train_avg_final_18', np.mean(np.array(direct_train_avg_18), axis=0))
    print('direct_train_max_final_18', np.array(direct_train_avg_18))
    print('direct_train_avg_final_19', np.mean(np.array(direct_train_avg_19), axis=0))
    print('direct_train_max_final_19', np.array(direct_train_avg_19))

    ###### pseudo labels#####
    print(' ###### pseudo labels#####')

    print('direct_train_avg_final_0', np.mean(np.array(pseudo_avg_0), axis=0))
    print('direct_train_max_final_0', np.array(pseudo_avg_0))
    print('direct_train_avg_final_1', np.mean(np.array(pseudo_avg_1), axis=0))
    print('direct_train_max_final_1', np.array(pseudo_avg_1))
    print('direct_train_avg_final_2', np.mean(np.array(pseudo_avg_2), axis=0))
    print('direct_train_max_final_2', np.array(pseudo_avg_2))
    print('direct_train_avg_final_3', np.mean(np.array(pseudo_avg_3), axis=0))
    print('direct_train_max_final_3', np.array(pseudo_avg_3))
    print('direct_train_avg_final_4', np.mean(np.array(pseudo_avg_4), axis=0))
    print('direct_train_max_final_4', np.array(pseudo_avg_4))

    print('direct_train_avg_final_5', np.mean(np.array(pseudo_avg_5), axis=0))
    print('direct_train_max_final_5', np.array(direct_train_avg_5))
    print('direct_train_avg_final_6', np.mean(np.array(pseudo_avg_6), axis=0))
    print('direct_train_max_final_6', np.array(pseudo_avg_6))
    print('direct_train_avg_final_7', np.mean(np.array(pseudo_avg_7), axis=0))
    print('direct_train_max_final_7', np.array(pseudo_avg_7))
    print('direct_train_avg_final_8', np.mean(np.array(pseudo_avg_8), axis=0))
    print('direct_train_max_final_8', np.array(pseudo_avg_8))
    print('direct_train_avg_final_9', np.mean(np.array(pseudo_avg_9), axis=0))
    print('direct_train_max_final_9', np.array(pseudo_avg_9))

    print('direct_train_avg_final_10', np.mean(np.array(pseudo_avg_10), axis=0))
    print('direct_train_max_final_10', np.array(pseudo_avg_10))
    print('direct_train_avg_final_11', np.mean(np.array(pseudo_avg_11), axis=0))
    print('direct_train_max_final_11', np.array(pseudo_avg_11))
    print('direct_train_avg_final_12', np.mean(np.array(pseudo_avg_12), axis=0))
    print('direct_train_max_final_12', np.array(pseudo_avg_12))
    print('direct_train_avg_final_13', np.mean(np.array(pseudo_avg_13), axis=0))
    print('direct_train_max_final_13', np.array(pseudo_avg_13))
    print('direct_train_avg_final_14', np.mean(np.array(pseudo_avg_14), axis=0))
    print('direct_train_max_final_14', np.array(pseudo_avg_14))

    print('direct_train_avg_final_15', np.mean(np.array(pseudo_avg_15), axis=0))
    print('direct_train_max_final_15', np.array(pseudo_avg_15))
    print('direct_train_avg_final_16', np.mean(np.array(pseudo_avg_16), axis=0))
    print('direct_train_max_final_16', np.array(pseudo_avg_16))
    print('direct_train_avg_final_17', np.mean(np.array(pseudo_avg_17), axis=0))
    print('direct_train_max_final_17', np.array(pseudo_avg_17))
    print('direct_train_avg_final_18', np.mean(np.array(pseudo_avg_18), axis=0))
    print('direct_train_max_final_18', np.array(pseudo_avg_18))
    print('direct_train_avg_final_19', np.mean(np.array(pseudo_avg_19), axis=0))
    print('direct_train_max_final_19', np.array(pseudo_avg_19))