#### This is the implementation of NatIMG_FL ####
from lcoal_training_gems_gpu_att_test import lcoal_training
from lcoal_training_gems_gpu_att_test import data_name, num_clients
import torch
from torch import multiprocessing as mp
torch.multiprocessing.set_start_method('spawn')
import torchvision
import warnings
import numpy as np
import random
import os,copy
import models
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torchvision.datasets as dsets
from sklearn import decomposition
from torchvision.datasets import MNIST
from PIL import Image
from tqdm import tqdm, trange
from contrastive_loss import SupConLoss
from resnetv2_temp import resnet18_,resnet50_
from test_model_gpu import contrastive_test_method,contrastive_global_test_method
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from fedlab.utils.functional import save_dict
from Federated_avg import federated_average
from fedlab.utils.dataset.slicing import noniid_slicing, random_slicing
from utils import loop_iterable, set_requires_grad, GrayscaleToRgb
from multiprocess import Pool

print('data_name',data_name)

### parameters ###
labeled_num = 20
# image_size = 28
batch_size = 128
iterations = 200
epochs = 2  #50(default) 2
cm_rounds=50
source_num=3000
contrastive_weights = 2
clustering_weights = 2

criterion_1 = torch.nn.CrossEntropyLoss()
criterion_2 = SupConLoss()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
### target data type ###
data_type = 'phenomia'
if data_type=='covid':
    data_dir_HC = '/home/zhianhuang/huyao/Public2Medical/output_no_finding/'
    data_dir_covid = '/home/zhianhuang/huyao/Public2Medical/output/'
if data_type=='phenomia': 
    data_dir_HC = '/home/zhianhuang/huyao/Public2Medical/Pneumonia_chest_xray/train/NORMAL/'
    data_dir_covid = '/home/zhianhuang/huyao/Public2Medical/Pneumonia_chest_xray/train/PNEUMONIA/'
    data_dir_HC_test = '/home/zhianhuang/huyao/Public2Medical/Pneumonia_chest_xray/test/NORMAL/'
    data_dir_covid_test = '/home/zhianhuang/huyao/Public2Medical/Pneumonia_chest_xray/test/PNEUMONIA/'

class TensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return len(self.data_tensor)

class TensorDataset_aug(Dataset):
    def __init__(self, data_tensor, aug_data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.aug_data_tensor = aug_data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.aug_data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return len(self.data_tensor)

def load_xray(path):
    if path == data_dir_HC:
        HC=True
    else:
        HC=False
    x_dirs=os.listdir(path)
    dataset=[]
    label = []
    # PT_label=[]
    for file in range(len(x_dirs)):
        fpath=os.path.join(path,x_dirs[file])
        #print(f)
        _x=Image.open(fpath).convert('RGB')
        # print('_x',_x.shape)
        # img= test_transformer(_x)
        # print(img.shape)
        # dataset.append(img)
        if HC ==True:
            dataset.append(_x)
            label.append(1)
        else:
            dataset.append(_x)
            label.append(0)
    return dataset, label

def load_xray_test(path,image_size):
    test_transformer = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=3),
            transforms.ToTensor()
            ])
    if path == data_dir_HC_test:
        HC=True
    if path == data_dir_covid_test:
        HC=False
    x_dirs=os.listdir(path)
    dataset=[]
    label = []
    # PT_label=[]
    for file in range(len(x_dirs)):
        fpath=os.path.join(path,x_dirs[file])
        #print(f)
        _x=Image.open(fpath).convert('RGB')
        img= test_transformer(_x)
        # print(img.shape)
        # dataset.append(img)
        if HC ==True:
            dataset.append(img)
            label.append(1)
        else:
            dataset.append(img)
            label.append(0)
    return dataset, label



def noniid_slicing(data, labels, num_clients, num_shards):
    """Slice a dataset for non-IID.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset to slice.
        num_clients (int):  Number of client.
        num_shards (int): Number of shards.
    
    Notes:
        The size of a shard equals to ``int(len(dataset)/num_shards)``.
        Each client will get ``int(num_shards/num_clients)`` shards.

    Returns：
        dict: ``{ 0: indices of dataset, 1: indices of dataset, ..., k: indices of dataset }``
    """
    total_sample_nums = len(data)
    size_of_shards = int(total_sample_nums / num_shards)
    if total_sample_nums % num_shards != 0:
        warnings.warn(
            "warning: the length of dataset isn't divided exactly by num_shard.some samples will be dropped."
        )
    # the number of shards that each one of clients can get
    shard_pc = int(num_shards / num_clients)
    if num_shards % num_clients != 0:
        warnings.warn(
            "warning: num_shard isn't divided exactly by num_clients. some samples will be dropped."
        )

    dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}

    idxs = np.arange(total_sample_nums)

    # sort sample indices according to labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]  # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]

    # assign
    idx_shard = [i for i in range(num_shards)]
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, shard_pc, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i],
                 idxs[rand * size_of_shards:(rand + 1) * size_of_shards]),
                axis=0)

    return dict_users
def seconde_min(lt):
    d={}         #设定一个空字典
    for i, v in enumerate(lt):#利用函数enumerate列出lt的每个元素下标i和元素v
        d[v]=i   #把v作为字典的键，v对应的值是i
    lt.sort()    #运用sort函数对lt元素排
    y=lt[1]      #此时lt中第二小的下标是1，求出对应的元素就是字典对应的键
    return d[y]  #根据键找到对应值就是所找的下标
    
def cal_mean_column(HC_labeled_data):
    HC_labeled_data_ = copy.deepcopy(HC_labeled_data)
    for i in range(HC_labeled_data_.shape[1]):
        avg=torch.mean(HC_labeled_data_[:,i])
        if avg!=0:
            HC_labeled_data_[:,i] = HC_labeled_data_[:,i]/avg
        else:
            HC_labeled_data_[:,i] = HC_labeled_data_[:,i]
    return HC_labeled_data_
def cal_mean_row(HC_labeled_data):
    HC_labeled_data_ = copy.deepcopy(HC_labeled_data)
    for i in range(HC_labeled_data_.shape[0]):
        avg=torch.mean(HC_labeled_data_[i,:])
        if avg!=0:
            HC_labeled_data_[i,:] = HC_labeled_data_[i,:]/avg
        else:
            HC_labeled_data_[i,:] = HC_labeled_data_[i,:]
    return HC_labeled_data_
def obtain_similarity(labeled_data_,mnist_list):
    similarity=[]
    for i in range(len(mnist_list)):
        data_com = torch.cat([labeled_data_,mnist_list[i]],dim=0)
        estimator=decomposition.NMF(n_components=10, init='nndsvda', tol=5e-4)
        estimator.fit(data_com)
        WT = estimator.fit_transform(data_com.T)
        HT = estimator.components_
        HT_1 = HT[:,:len(labeled_data_)]
        HT_2 = HT[:,len(labeled_data_):]
        p_list=[]
        for j in range(50):
            p_=[]
            for i in range(len(HT_1)):
        #         print(max(max(HT_1[i]),max(HT_2[i])))
                T=np.random.uniform(max(max(HT_1[i]), max(HT_2[i])))
        #         print(T)
                n1=len(np.where(HT_1[i]>T)[0])
        #         print(n1)
                n2=len(np.where(HT_2[i]>T)[0])
                p=n1/HT_1.shape[1]-n2/HT_2.shape[1]
                p_.append(p)
            p_list.append(p_)
        #         print(n2)
        p_array=np.array(p_list)
        p_avg = np.mean(p_array,axis=0)
        x_norm=np.linalg.norm(p_avg, ord=1, axis=None, keepdims=False)
#         print(x_norm)
        similarity.append(x_norm)
    similarity_arr=np.array(similarity)
    print('similarity_arr',similarity_arr)
    index = np.argmin(similarity_arr)
    return similarity_arr, index
# for index, (data, label) in enumerate(train_dataset_loader):
#     print(data.shape)
#     print(label)
def obtain_mnist_ini(dataname,image_size, class_one):
    if dataname=="mnist":
        mnist_dataset = MNIST(root='./MNIST_data',
                         train=True,
                         transform=transforms.Compose([GrayscaleToRgb(), transforms.ToTensor()]),
                         download=True)
    if dataname=="cifar-10":
        
        transform = transforms.Compose([transforms.Scale(image_size),
                                transforms.ToTensor(),
                                # transforms.Grayscale(num_output_channels=1),
                                transforms.Normalize(mean=([0.5,0.5,0.5]), std=([0.5,0.5,0.5]))])
        mnist_dataset = dsets.CIFAR10(root='./CIFAR_data',
                         train=True,
                         transform=transform,
                         download=True)
    if dataname == 'imagenet':
        data_transform = transforms.Compose([
        transforms.Resize((image_size,image_size), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    	     std=[0.229, 0.224, 0.225])
    ])


        mnist_dataset = torchvision.datasets.ImageFolder(root='/home/zhianhuang/Imagenet/val',transform=data_transform)
    mnist_sample_one=[]

    for images, labels in mnist_dataset:
        if labels== class_one:
            # print('images',images.numpy.shape)
            mnist_sample_one.append(abs(images.numpy()))

    mnist_sample_one=np.array(mnist_sample_one)

    mnist_sample_one=mnist_sample_one.astype(float)
    
    return mnist_sample_one
    
def obtain_mnist(dataname,image_size, class_one):
    if dataname=="mnist":
        mnist_dataset = MNIST(root='./MNIST_data',
                         train=True,
                         transform=transforms.Compose([GrayscaleToRgb(), transforms.ToTensor()]),
                         download=True)
    if dataname=="cifar-10":
        
        transform = transforms.Compose([transforms.Scale(image_size),
                                transforms.ToTensor(),
                                # transforms.Grayscale(num_output_channels=1),
                                transforms.Normalize(mean=([0.5,0.5,0.5]), std=([0.5,0.5,0.5]))])
        mnist_dataset = dsets.CIFAR10(root='./CIFAR_data',
                         train=True,
                         transform=transform,
                         download=True)
    if dataname == 'imagenet':
        data_transform = transforms.Compose([
        transforms.Resize((image_size,image_size), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    	     std=[0.229, 0.224, 0.225])
    ])


        mnist_dataset = torchvision.datasets.ImageFolder(root='/home/zhianhuang/Imagenet/val',transform=data_transform)
    mnist_sample_one=[]

    for images, labels in mnist_dataset:
        if labels== class_one:
            # print('images',images.numpy.shape)
            mnist_sample_one.append(images.numpy())

    mnist_sample_one=np.array(mnist_sample_one)

    mnist_sample_one=mnist_sample_one.astype(float)
    
    return mnist_sample_one

def generate_training_samples(data_name, image_size,PT_similarity_index,HC_similarity_index):
    PT_mnist_sample_arr_ = obtain_mnist(data_name,image_size,PT_similarity_index)
    HC_mnist_sample_arr_ = obtain_mnist(data_name,image_size,HC_similarity_index)
    PT_mnist_label = np.zeros(len(PT_mnist_sample_arr_))
    HC_mnist_label = np.ones(len(HC_mnist_sample_arr_))
    PT_mnist_tensor = torch.tensor(PT_mnist_sample_arr_)
    HC_mnist_tensor = torch.tensor(HC_mnist_sample_arr_)
    PT_mnist_label_tensor = torch.tensor(PT_mnist_label)
    HC_mnist_label_tensor = torch.tensor(HC_mnist_label)
    mnist_data = torch.cat([PT_mnist_tensor, HC_mnist_tensor],0)
    mnist_label = torch.cat([PT_mnist_label_tensor, HC_mnist_label_tensor],0)
    index = [i for i in range(len(mnist_data))]
    random.shuffle(index)
    mnist_data = mnist_data[index]
    mnist_label = mnist_label[index]
    # print('mnist_label',mnist_label)
    dataset = TensorDataset(mnist_data, mnist_label)
    return dataset


def obtain_xray(image_size, all_dataset, all_label):
    HC_index = np.where(all_label==1)[0]
    PT_index = np.where(all_label==0)[0]
    HC_dataset = all_dataset[HC_index]
    PT_dataset = all_dataset[PT_index]
    HC_label = all_label[HC_index]
    PT_label = all_label[PT_index]

    test_transformer = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=3),
            transforms.ToTensor()
            ])
    aug_transformer = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=3),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ToTensor()
            ])
    HC_ind = random.sample(range(0, len(HC_dataset)-1), labeled_num)
    PT_ind = random.sample(range(0, len(PT_dataset)-1), labeled_num)
    # HC_ind = random.sample(range(0, len(HC_dataset)-1), int(len(HC_dataset)*labeled_per))
    # PT_ind = random.sample(range(0, len(PT_dataset)-1), int(len(PT_dataset)*labeled_per))
    # HC_ind = [random.randint(0,len(HC_dataset)-1) for i in range(labeled_num)]
    # PT_ind = [random.randint(0,len(PT_dataset)-1) for i in range(labeled_num)]
    HC_all = [i for i in range(len(HC_dataset))]
    PT_all = [i for i in range(len(PT_dataset))]
    for i in HC_ind:
        HC_all.remove(i)
    for i in PT_ind:
        PT_all.remove(i)
    HC_unlabel_ind = HC_all
    PT_unlabel_ind = PT_all
    HC_labeled_sam=[]
    PT_labeled_sam=[]
    HC_labeled_sample_aug, PT_labeled_sample_aug = [],[]
    HC_labeled_sample, PT_labeled_sample = [],[]
    HC_unlabeled_sample_aug, PT_unlabeled_sample_aug = [],[]
    HC_unlabeled_sample, PT_unlabeled_sample = [],[]
    for i in HC_ind:
        HC_labeled_sam.append(HC_dataset[i])
    for i in PT_ind:
        PT_labeled_sam.append(PT_dataset[i])

    HC_lab = torch.tensor(np.array(HC_label)[HC_ind])
    PT_lab = torch.tensor(np.array(PT_label)[PT_ind])
    HC_unlabeled_sam,PT_unlabeled_sam=[],[]
    for i in HC_unlabel_ind:
        HC_unlabeled_sam.append(HC_dataset[i])
    for i in PT_unlabel_ind:
        PT_unlabeled_sam.append(PT_dataset[i])

    # HC_unlabeled_sam= HC_dataset[HC_unlabel_ind]
    # PT_unlabeled_sam= PT_dataset[PT_unlabel_ind]
    HC_un_lab = np.array(HC_label)[HC_unlabel_ind]
    PT_un_lab = np.array(PT_label)[PT_unlabel_ind]
    for i in HC_labeled_sam:
        img = test_transformer(i)
        aug_img = aug_transformer(i)
        HC_labeled_sample.append(img)
        HC_labeled_sample_aug.append(aug_img)
    for j in PT_labeled_sam:
        img = test_transformer(j)
        aug_img = aug_transformer(j)
        PT_labeled_sample.append(img)
        PT_labeled_sample_aug.append(aug_img)
    for i in HC_unlabeled_sam:
        img = test_transformer(i)
        aug_img = aug_transformer(i)
        HC_unlabeled_sample.append(img)
        HC_unlabeled_sample_aug.append(aug_img)
    for j in PT_unlabeled_sam:
        img = test_transformer(j)
        aug_img = aug_transformer(j)
        PT_unlabeled_sample.append(img)
        PT_unlabeled_sample_aug.append(aug_img)


    HC_unlabeled_sample_arr = np.array([t.numpy() for t in HC_unlabeled_sample])
    HC_unlabeled_aug_sample_arr = np.array([t.numpy() for t in HC_unlabeled_sample_aug])
    PT_unlabeled_sample_arr = np.array([t.numpy() for t in PT_unlabeled_sample])
    PT_unlabeled_aug_sample_arr = np.array([t.numpy() for t in PT_unlabeled_sample_aug])
    HC_labeled_sample_arr = np.array([t.numpy() for t in HC_labeled_sample])
    HC_labeled_sample_ten = torch.tensor(HC_labeled_sample_arr)
    HC_labeled_aug_sample_arr = np.array([t.numpy() for t in HC_labeled_sample_aug])
    HC_labeled_aug_sample_ten = torch.tensor(HC_labeled_aug_sample_arr)
    PT_labeled_sample_arr = np.array([t.numpy() for t in PT_labeled_sample])
    PT_labeled_sample_ten = torch.tensor(PT_labeled_sample_arr)
    PT_labeled_aug_sample_arr = np.array([t.numpy() for t in PT_labeled_sample_aug])
    PT_labeled_aug_sample_ten = torch.tensor(PT_labeled_aug_sample_arr)

    labeled_data = np.concatenate([HC_labeled_sample_arr, PT_labeled_sample_arr])
    labeled_aug_data = np.concatenate([HC_labeled_aug_sample_arr, PT_labeled_aug_sample_arr])
    labeled_label = np.concatenate([HC_lab, PT_lab])
    labeled_dataset = TensorDataset_aug(labeled_data,labeled_aug_data,labeled_label)

    unlabeled_data = np.concatenate([HC_unlabeled_sample_arr, PT_unlabeled_sample_arr])
    unlabeled_aug_data = np.concatenate([HC_unlabeled_aug_sample_arr, PT_unlabeled_aug_sample_arr])
    unlabeled_label = np.concatenate([HC_un_lab, PT_un_lab])
    unlabeled_dataset = TensorDataset_aug(unlabeled_data,unlabeled_aug_data,unlabeled_label)
    return HC_labeled_sample_arr, PT_labeled_sample_arr,labeled_dataset,unlabeled_dataset

def clean_data(HC_similarity_list_,PT_similarity_list_):
    HC_similarity_list = copy.deepcopy(HC_similarity_list_)
    PT_similarity_list = copy.deepcopy(PT_similarity_list_)
    HC_similarity_clean = list(set(HC_similarity_list))
    PT_similarity_clean = list(set(PT_similarity_list))
    HC_similarity_clean_copy = copy.deepcopy(HC_similarity_clean)
    PT_similarity_clean_copy = copy.deepcopy(PT_similarity_clean)
    for i in HC_similarity_clean_copy:
        if i in PT_similarity_clean_copy:
            idx_HC = len(np.where(np.array(HC_similarity_list)==i)[0])
            idx_PT = len(np.where(np.array(PT_similarity_list)==i)[0])
            if idx_HC>idx_PT:
                PT_similarity_clean.remove(i)
            else:
                HC_similarity_clean.remove(i)
    return HC_similarity_clean,PT_similarity_clean
def get_dataloader(data_name,image_size,class_num,HC_labeled_data,PT_labeled_data,labeled_dataset,unlabeled_dataset):
    HC_labeled_data_ = np.reshape(HC_labeled_data,[-1, HC_labeled_data.shape[1]*HC_labeled_data.shape[2]*HC_labeled_data.shape[3]])
    HC_labeled_datas_ = torch.tensor(HC_labeled_data_)
    HC_labeled_data_ = cal_mean_row(HC_labeled_datas_)
#    HC_labeled_data_ = HC_labeled_data_/torch.mean(HC_labeled_data_)
    
    PT_labeled_data_ = np.reshape(PT_labeled_data,[-1, PT_labeled_data.shape[1]*PT_labeled_data.shape[2]*PT_labeled_data.shape[3]])
    PT_labeled_datas_ = torch.tensor(PT_labeled_data_)
    PT_labeled_data_ = cal_mean_row(PT_labeled_datas_)
#    PT_labeled_data_ = PT_labeled_data_/torch.mean(PT_labeled_data_)


    mnist_list=[]
    
    if data_name=="mnist":
        
        mnist_dataset = MNIST(root='./MNIST_data',
                         train=True,
                         transform=transforms.Compose([GrayscaleToRgb(), transforms.ToTensor()]),
                         download=True)
    if data_name=="cifar-10":
        
        transform = transforms.Compose([transforms.Scale(image_size),
                                transforms.ToTensor(),
                                # transforms.Grayscale(num_output_channels=1),
                                transforms.Normalize(mean=([0.5,0.5,0.5]), std=([0.5,0.5,0.5]))])
        mnist_dataset = dsets.CIFAR10(root='./CIFAR_data',
                         train=True,
                         transform=transform,
                         download=True)
    if data_name == 'imagenet':
        data_transform = transforms.Compose([
        transforms.Resize((image_size,image_size), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    	     std=[0.229, 0.224, 0.225])
    ])


        mnist_dataset = torchvision.datasets.ImageFolder(root='/home/zhianhuang/Imagenet/val',transform=data_transform)
    mnist_sample_one=[]
    images_list=[]
    labels_list=[]
    for images, labels in mnist_dataset:
        labels_list.append(labels)
        mnist_sample_one.append(abs(images.numpy()))
#        images_list.append(images.numpy())

    mnist_sample_one=np.array(mnist_sample_one)
    labels_list_array=np.array(labels_list)
    mnist_sample_one=mnist_sample_one.astype(float)
    
    for kk in range(class_num):
        ind_k=np.where(labels_list_array==kk)[0]
        mnist_sample_arr_ = mnist_sample_one[ind_k].astype(float)
        mnist_sample_arr = np.reshape(mnist_sample_arr_,[-1, mnist_sample_arr_.shape[1]*mnist_sample_arr_.shape[2]*mnist_sample_arr_.shape[3]])
        mnist_sample_arrs = torch.tensor(mnist_sample_arr)
        mnist_scale = cal_mean_row(mnist_sample_arrs)
#        mnist_scale = mnist_sample_arr/torch.mean(mnist_sample_arr)
        mnist_list.append(mnist_scale)
    HC_similarity, HC_similarity_index = obtain_similarity(HC_labeled_data_,mnist_list)
    PT_similarity, PT_similarity_index = obtain_similarity(PT_labeled_data_,mnist_list) 
    print('HC_similarity',HC_similarity)
    print('PT_similarity',PT_similarity)
    print('HC_similarity_index',HC_similarity_index)
    print('PT_similarity_index',PT_similarity_index)
    if HC_similarity_index==PT_similarity_index:
        if HC_similarity[HC_similarity_index]<=PT_similarity[PT_similarity_index]:
            HC_similarity_index=HC_similarity_index
            seconde_ind = seconde_min(PT_similarity)
            PT_similarity_index=seconde_ind
        if HC_similarity[HC_similarity_index]>PT_similarity[PT_similarity_index]:
            PT_similarity_index=PT_similarity_index
            seconde_ind = seconde_min(HC_similarity)
            HC_similarity_index=seconde_ind
    print('revised_HC_similarity_index',HC_similarity_index)
    print('revised_PT_similarity_index',PT_similarity_index)
    
    # sor_dataset = generate_training_samples(data_name,image_size,PT_similarity_index,HC_similarity_index)
    
    

    # sor_dataset, labeled_dataset, unlabeled_dataset = divide_labeled(HC_img, HC_label,cov_img,cov_label,class_one, class_two, data_name, image_size)
    
    # source_loader = DataLoader(sor_dataset, batch_size=batch_size,
    #                            shuffle=True, num_workers=0, pin_memory=True)
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=14, pin_memory=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=14, pin_memory=True)
    return labeled_loader,unlabeled_loader,HC_similarity_index,PT_similarity_index

def obtain_mnist_global(dataname,image_size, class_one):
    if dataname=="mnist":
        mnist_dataset = MNIST(root='./MNIST_data',
                         train=True,
                         transform=transforms.Compose([GrayscaleToRgb(), transforms.ToTensor()]),
                         download=True)
    if dataname=="cifar-10":
        
        transform = transforms.Compose([transforms.Scale(image_size),
                                transforms.ToTensor(),
                                # transforms.Grayscale(num_output_channels=1),
                                transforms.Normalize(mean=([0.5,0.5,0.5]), std=([0.5,0.5,0.5]))])
        mnist_dataset = dsets.CIFAR10(root='./CIFAR_data',
                         train=True,
                         transform=transform,
                         download=True)
    if dataname == 'imagenet':
        data_transform = transforms.Compose([
        transforms.Resize((image_size,image_size), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    	     std=[0.229, 0.224, 0.225])
    ])


        mnist_dataset = torchvision.datasets.ImageFolder(root='/home/zhianhuang/Imagenet/val',transform=data_transform)
    mnist_sample_one=[]

    for images, labels in mnist_dataset:
        if labels in class_one:
            # print('images',images.numpy.shape)
            mnist_sample_one.append(images.numpy())

    # mnist_sample_one=np.array(mnist_sample_one)
    rand_idx = [i for i in range(len(mnist_sample_one))]
    source_sel = random.sample(rand_idx, source_num)
    mnist_sample_one=np.array(mnist_sample_one)[source_sel]

    # mnist_sample_one=np.array(mnist_sample_one)

    mnist_sample_one=mnist_sample_one.astype(float)
    
    return mnist_sample_one

def generate_training_samples_global(data_name, image_size,PT_similarity_index,HC_similarity_index):
    aug_transformer = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=3),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ToTensor()
            ])
    PT_mnist_sample_arr_ = obtain_mnist_global(data_name,image_size,PT_similarity_index)
    HC_mnist_sample_arr_ = obtain_mnist_global(data_name,image_size,HC_similarity_index)
    PT_mnist_label = np.zeros(len(PT_mnist_sample_arr_))
    HC_mnist_label = np.ones(len(HC_mnist_sample_arr_))
    PT_mnist_tensor = torch.tensor(PT_mnist_sample_arr_)
    HC_mnist_tensor = torch.tensor(HC_mnist_sample_arr_)
    PT_mnist_label_tensor = torch.tensor(PT_mnist_label)
    HC_mnist_label_tensor = torch.tensor(HC_mnist_label)
    mnist_data = torch.cat([PT_mnist_tensor, HC_mnist_tensor],0)
    mnist_label = torch.cat([PT_mnist_label_tensor, HC_mnist_label_tensor],0)
    index = [i for i in range(len(mnist_data))]
    random.shuffle(index)
    mnist_data = mnist_data[index]
    # mnist_data_arr = np.array(mnist_data.transpose(0, 2, 3, 1))
    # print('mnist_data_arr',mnist_data_arr.shape)
    mnist_label = mnist_label[index]
    # aug_source=[]
    # for i in mnist_data:
    #     image = torchvision.transforms.functional.to_pil_image(i)
    #     aug_img = aug_transformer(image)
    #     aug_source.append(aug_img)

    # aug_data = np.array([t.numpy() for t in aug_source])
    # aug_data=torch.tensor(aug_data)
    # print('aug_data',aug_data.shape)
    dataset = TensorDataset(mnist_data,mnist_label)

    # print('mnist_label',mnist_label)
    # dataset = TensorDataset(mnist_data, mnist_label)
    return dataset


def start_client(paras):
    # print("start client")
    c = lcoal_training(paras[0],paras[1], paras[2], paras[3],paras[4],paras[5], paras[6],paras[7],paras[8],paras[9],paras[10])

def avg_parameter(model_list,coefficient_matrix):
    for i in range(num_clients):
        model_list[i].load_state_dict(torch.load('local_model/'+'att_local_test_'+str(i)+'.pt'))
    federated_average(model_list, coefficient_matrix, False) # fedavg algorithm
    return model_list

def main():
    # criterion_kl = DistillKL(args.temperature)
    if data_name == 'mnist':
        # model_global = models.__dict__['wrn28x2'](num_classes=50)
        # model_t.load_state_dict(torch.load(args.trained_dir))
        model = models.__dict__['wrn16x2'](num_classes=50)

        image_size = 28
        class_num=10
    if data_name == 'cifar-10':
        # from models_contrastive import resnet18
        # model_global = models.__dict__['wrn28x2'](num_classes=50)
        # model_t.load_state_dict(torch.load(args.trained_dir))
        model = resnet18_(pretrained=True)
        image_size = 224
        class_num=10
        # print('model_done')
    if data_name == 'imagenet':
        # model_global = models.__dict__['resnet110'](num_classes=50)
        # model_t.load_state_dict(torch.load(args.trained_dir))
        model = resnet50_(pretrained=True)
        # from models_contrastive import resnet50
        image_size = 224
        class_num=1000

    HC_img, HC_label = load_xray(data_dir_HC)
    PT_img, PT_label = load_xray(data_dir_covid)
    HC_img_test, HC_label_test = load_xray_test(data_dir_HC_test,image_size)
    PT_img_test, PT_label_test = load_xray_test(data_dir_covid_test,image_size)
    # print('HC_img_test',np.array(HC_img_test).shape)
    # print('PT_img_test',np.array(PT_img_test).shape)
    total_img = np.concatenate([HC_img,PT_img])
    total_label = np.concatenate([HC_label,PT_label])
    total_img_test = np.concatenate([HC_img_test,PT_img_test])
    total_label_test = np.concatenate([HC_label_test,PT_label_test])
    # print('total_img_test',total_img_test)
    random_index = [i for i in range(len(total_label))]
    random_index_test = [i for i in range(len(total_label_test))]
    random.shuffle(random_index)
    random.shuffle(random_index_test)
    total_img=total_img[random_index]
    total_label=total_label[random_index]
    total_img_test=total_img_test[random_index_test]
    total_label_test=total_label_test[random_index_test]
    # print('total_img_test',total_img_test.shape)
    # print('total_label_test',total_label_test.shape)
    test_dataset = TensorDataset(total_img_test, total_label_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=14, pin_memory=True)
    # for ind, (x,y) in enumerate(test_loader):
    #     print('x',x.shape)
    #     print('y',y)
    data_indices = noniid_slicing(total_img, total_label, num_clients, num_shards=200)
    save_dict(data_indices, "att_data_noniid.pkl")

    labeled_loader_list, unlabeled_loader_list=[], []
    model_list, optim_list, schedule_list=[],[],[]
    coefficient_matrix_=[]
    HC_similarity_list, PT_similarity_list = [],[]

    test_loader_list=[]
    for i in data_indices.keys(): 
        num_index = data_indices[i].tolist()
        coefficient_matrix_.append(len(num_index))
        # print(num_index)
        data_temp = total_img[num_index]
        label_temp = total_label[num_index]
        # print('label', label_temp)
        HC_labeled_data,PT_labeled_data,labeled_dataset,unlabeled_dataset = obtain_xray(image_size, data_temp, label_temp)
        labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=14, pin_memory=True)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=14, pin_memory=True)
        labeled_loader,unlabeled_loader,HC_similarity_index,PT_similarity_index = get_dataloader(data_name,image_size,class_num,HC_labeled_data,PT_labeled_data,labeled_dataset,unlabeled_dataset)
        HC_similarity_list.append(HC_similarity_index)
        PT_similarity_list.append(PT_similarity_index)

        labeled_loader_list.append(labeled_loader)
        unlabeled_loader_list.append(unlabeled_loader)
        test_loader_list.append(test_loader)
        optim = torch.optim.Adam(model.parameters())
        lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True)
        model_list.append(model)
        optim_list.append(optim)
        schedule_list.append(lr_schedule)
    HC_similarity_clean,PT_similarity_clean = clean_data(HC_similarity_list,PT_similarity_list)
    print('HC_similarity_clean',HC_similarity_clean)
    print('PT_similarity_clean',PT_similarity_clean)
    intersected_natural_dataset = generate_training_samples_global(data_name,image_size,PT_similarity_clean,HC_similarity_clean)
    intersected_natural_loader = DataLoader(intersected_natural_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=14, pin_memory=True)
    coefficient_matrix = coefficient_matrix_/np.array(coefficient_matrix_).sum()
    idx = [i for i in range(num_clients)]
    intersected_natural_loader_list =[intersected_natural_loader]*num_clients
    epochs_list=[epochs]*num_clients
    iterations_list=[iterations]*num_clients
    for commun in range(cm_rounds):
        comm_list=[commun]*num_clients
        para_list=[]
        for i in range(num_clients):
            para_list.append([model_list[i],optim_list[i],schedule_list[i], idx[i], intersected_natural_loader_list[i],labeled_loader_list[i],unlabeled_loader_list[i],test_loader_list[i], epochs_list[i],iterations_list[i],comm_list[i]])
        # print('commun',commun)
        p=Pool(num_clients)
        p.map(start_client,para_list)
        p.close()
        p.join()
        model_list = avg_parameter(model_list, coefficient_matrix) # fedavg algorithm

        
if __name__ == '__main__':
    main()
