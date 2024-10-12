import argparse
import numpy as np
import os
# from torch.utils.data import DataLoader
# from torchvision.datasets import MNIST
# from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
# import random
# from data import MNISTM
# from models import Net
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix,roc_auc_score


def train_test(model, device, test_loader):
    import torch
    model.to(device)
    model.eval()
    ADHD_pred = []
    label=[]
    # total_accuracy = 0
    with torch.no_grad():
        for ind, (x, y_true) in enumerate(test_loader):
            x, y_true = x.to(device), y_true.to(device)
            _,y_pred,_,__ = model(x)
            # print(y_pred)
            # print(y_pred.max(1)[1])
            # total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
            for i in range(len(y_pred.max(1)[1])):
                label.append(y_true[i].item())
                ADHD_pred.append(y_pred.max(1)[1][i].item())
            # print('ADHD_pred',ADHD_pred)
            # print('label',label)
    AUC = roc_auc_score(label, ADHD_pred)
    return  AUC

def contrastive_test_method(model,test_loader,device):
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    import torch
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('test_device',device)
    model.cuda(0)
    model.eval()
    ADHD_pred = []
    label=[]
    # total_accuracy = 0
    with torch.no_grad():
        for ind, (x, _, y_true) in enumerate(test_loader):
        # for x, _, y_true in tqdm(test_loader, leave=False):
            # x, y_true = x.to(device), y_true.to(device)
            x, y_true = x.cuda(0), y_true.cuda(0)
            _,y_pred,_,__ = model(x)
            # print(y_pred)
            # print(y_pred.max(1)[1])
            # total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
            for i in range(len(y_pred.max(1)[1])):
                label.append(y_true[i].item())
                ADHD_pred.append(y_pred.max(1)[1][i].item())
            # print('ADHD_pred',ADHD_pred)
            # print('label',label)

    confusion = confusion_matrix(label, ADHD_pred)
    # print('confusion',confusion)
    TP_num = confusion[1, 1]
    TN_num = confusion[0, 0]
    FP_num = confusion[0, 1]
    FN_num = confusion[1, 0]
    ACC = (TP_num + TN_num) / float(TP_num + TN_num + FP_num + FN_num)

    if float(TN_num + FP_num) == 0:
        SPE = 0
    else:
        SPE = TN_num / float(TN_num + FP_num)
    if float(TP_num + FN_num) == 0:
        REC = 0
    else:
        REC = TP_num / float(TP_num + FN_num)
    # Cor_num = TP_num + TN_num
    # print('target_array', true_labels)
    # print('A_array', pred)
    AUC = roc_auc_score(label, ADHD_pred)
    # del device
    return ACC, SPE, REC, AUC


def contrastive_global_test_method(model,test_loader,device):
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import torch
    model.cuda()
    model.eval()
    ADHD_pred = []
    label=[]
    # total_accuracy = 0
    with torch.no_grad():
        for ind, (x, y_true) in enumerate(test_loader):
            x, y_true = x.cuda(0), y_true.cuda(0)
            # x, y_true = x.to(device), y_true.to(device)
            _,y_pred,_,__ = model(x)
            # print(y_pred)
            # print(y_pred.max(1)[1])
            # total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
            for i in range(len(y_pred.max(1)[1])):
                label.append(y_true[i].item())
                ADHD_pred.append(y_pred.max(1)[1][i].item())
            # print('ADHD_pred',ADHD_pred)
            # print('label',label)

    confusion = confusion_matrix(label, ADHD_pred)
    # print('confusion',confusion)
    TP_num = confusion[1, 1]
    TN_num = confusion[0, 0]
    FP_num = confusion[0, 1]
    FN_num = confusion[1, 0]
    ACC = (TP_num + TN_num) / float(TP_num + TN_num + FP_num + FN_num)

    if float(TN_num + FP_num) == 0:
        SPE = 0
    else:
        SPE = TN_num / float(TN_num + FP_num)
    if float(TP_num + FN_num) == 0:
        REC = 0
    else:
        REC = TP_num / float(TP_num + FN_num)
    # Cor_num = TP_num + TN_num
    # print('target_array', true_labels)
    # print('A_array', pred)
    AUC = roc_auc_score(label, ADHD_pred)
    # del device
    return ACC, SPE, REC, AUC


def test_method(model, test_loader):
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model.to(device)
    model.eval()
    ADHD_pred = []
    label=[]
    # total_accuracy = 0
    with torch.no_grad():
        for ind, (x, y_true) in enumerate(test_loader):
            x, y_true = x.to(device), y_true.to(device)
            _,y_pred = model(x)
            # print(y_pred)
            # print(y_pred.max(1)[1])
            # total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
            for i in range(len(y_pred.max(1)[1])):
                label.append(y_true[i].item())
                ADHD_pred.append(y_pred.max(1)[1][i].item())
            # print('ADHD_pred',ADHD_pred)
            # print('label',label)

    confusion = confusion_matrix(label, ADHD_pred)
    # print('confusion',confusion)
    TP_num = confusion[1, 1]
    TN_num = confusion[0, 0]
    FP_num = confusion[0, 1]
    FN_num = confusion[1, 0]
    ACC = (TP_num + TN_num) / float(TP_num + TN_num + FP_num + FN_num)

    if float(TN_num + FP_num) == 0:
        SPE = 0
    else:
        SPE = TN_num / float(TN_num + FP_num)
    if float(TP_num + FN_num) == 0:
        REC = 0
    else:
        REC = TP_num / float(TP_num + FN_num)
    # Cor_num = TP_num + TN_num
    # print('target_array', true_labels)
    # print('A_array', pred)
    AUC = roc_auc_score(label, ADHD_pred)
    del device
    return ACC, SPE, REC, AUC



# def main(args):
#     if CIFAR==True:
#         from models_cifar import Net
#         if aug:
#             from adda_aug_cifar import load_xray, TensorDataset
#             data_dir_HC = '/home/zhianhuang/huyao/Public2Medical/output_no_finding/'
#             data_dir_covid = '/home/zhianhuang/huyao/Public2Medical/output/'
#             HC_img, HC_label,HC_aug_img,HC_aug_label = load_xray(data_dir_HC)
#             HC_img_arr = np.array([t.numpy() for t in HC_img])
#             HC_aug_arr = np.array([t.numpy() for t in HC_aug_img])
#             cov_img, cov_label,_,__ = load_xray(data_dir_covid)
#             cov_img_arr = np.array([t.numpy() for t in cov_img])
#             x_ray_img = np.concatenate([HC_img_arr, cov_img_arr, HC_aug_arr])
#             x_ray_label = np.concatenate([HC_label, cov_label, HC_aug_label])
#             ind = [i for i in range(len(x_ray_img))]
#             random.shuffle(ind)
#             x_ray_img = x_ray_img[ind]
#             x_ray_label = x_ray_label[ind]
#             # print(x_ray_img.shape)
#             # print(x_ray_label.shape)
#             # print('0_samples',len(np.where(x_ray_label==0)[0]))
#             # print('1_samples',len(np.where(x_ray_label==1)[0]))
#             xray_dataset = TensorDataset(x_ray_img, x_ray_label)
#             dataloader = torch.utils.data.DataLoader(dataset=xray_dataset,
#                                                 batch_size=args.batch_size,
#                                                 shuffle=False)
#         else:
#             from adda import load_xray_cifar, TensorDataset
#             data_dir_HC = '/home/zhianhuang/huyao/Public2Medical/output_no_finding/'
#             data_dir_covid = '/home/zhianhuang/huyao/Public2Medical/output/'
#             HC_img, HC_label = load_xray_cifar(data_dir_HC)
#             HC_img_arr = np.array([t.numpy() for t in HC_img])
#             cov_img, cov_label = load_xray_cifar(data_dir_covid)
#             cov_img_arr = np.array([t.numpy() for t in cov_img])
#             x_ray_img = np.concatenate([HC_img_arr, cov_img_arr])
#             x_ray_label = np.concatenate([HC_label, cov_label])
#             ind = [i for i in range(len(x_ray_img))]
#             random.shuffle(ind)
#             x_ray_img = x_ray_img[ind]
#             x_ray_label = x_ray_label[ind]
#             # print(x_ray_img.shape)
#             # print(x_ray_label)
#             xray_dataset = TensorDataset(x_ray_img, x_ray_label)
#             dataloader = torch.utils.data.DataLoader(dataset=xray_dataset,
#                                                 batch_size=args.batch_size,
#                                                 shuffle=False)
#     if CIFAR==False:
#         from models import Net
#         if aug:
#             from adda_aug import load_xray, TensorDataset
#             data_dir_HC = '/home/zhianhuang/huyao/Public2Medical/output_no_finding/'
#             data_dir_covid = '/home/zhianhuang/huyao/Public2Medical/output/'
#             HC_img, HC_label,HC_aug_img,HC_aug_label = load_xray(data_dir_HC)
#             HC_img_arr = np.array([t.numpy() for t in HC_img])
#             HC_aug_arr = np.array([t.numpy() for t in HC_aug_img])
#             cov_img, cov_label,_,__ = load_xray(data_dir_covid)
#             cov_img_arr = np.array([t.numpy() for t in cov_img])
#             x_ray_img = np.concatenate([HC_img_arr, cov_img_arr, HC_aug_arr])
#             x_ray_label = np.concatenate([HC_label, cov_label, HC_aug_label])
#             ind = [i for i in range(len(x_ray_img))]
#             random.shuffle(ind)
#             x_ray_img = x_ray_img[ind]
#             x_ray_label = x_ray_label[ind]
#             # print(x_ray_img.shape)
#             # print(x_ray_label.shape)
#             # print('0_samples',len(np.where(x_ray_label==0)[0]))
#             # print('1_samples',len(np.where(x_ray_label==1)[0]))
#             xray_dataset = TensorDataset(x_ray_img, x_ray_label)
#             dataloader = torch.utils.data.DataLoader(dataset=xray_dataset,
#                                                 batch_size=args.batch_size,
#                                                 shuffle=False)
#         else:
#             from adda import load_xray, TensorDataset
#             data_dir_HC = '/home/zhianhuang/huyao/Public2Medical/output_no_finding/'
#             data_dir_covid = '/home/zhianhuang/huyao/Public2Medical/output/'
#             HC_img, HC_label = load_xray(data_dir_HC)
#             HC_img_arr = np.array([t.numpy() for t in HC_img])
#             cov_img, cov_label = load_xray(data_dir_covid)
#             cov_img_arr = np.array([t.numpy() for t in cov_img])
#             x_ray_img = np.concatenate([HC_img_arr, cov_img_arr])
#             x_ray_label = np.concatenate([HC_label, cov_label])
#             ind = [i for i in range(len(x_ray_img))]
#             random.shuffle(ind)
#             x_ray_img = x_ray_img[ind]
#             x_ray_label = x_ray_label[ind]
#             # print(x_ray_img.shape)
#             # print(x_ray_label)
#             xray_dataset = TensorDataset(x_ray_img, x_ray_label)
#             dataloader = torch.utils.data.DataLoader(dataset=xray_dataset,
#                                                 batch_size=args.batch_size,
#                                                 shuffle=False)

#     # dataset = MNISTM(train=False)
#     # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
#                             # drop_last=False, num_workers=1, pin_memory=True)

#     model = Net().to(device)
#     model.load_state_dict(torch.load(args.MODEL_FILE))
#     model.eval()

#     ACC, SPE, REC, AUC = test_method(model, device, dataloader)
#     print(f'category_1, cltegory_2, ACC, SPE, REC and AUC on target data: {class_one:.0f},{class_two:.0f},{ACC:.4f},{SPE:.4f},{REC:.4f},{AUC:.4f}')
    

# if __name__ == '__main__':
#     # aug=True
#     # CIFAR=True
#     # class_one=  1234  #1234
#     # class_two=  56789 #56789
#     # arg_parser = argparse.ArgumentParser(description='Test a model on MNIST-M')
#     # arg_parser.add_argument('--MODEL_FILE', help='A model in trained_models',default='pytorch-domain-adaptation-master/source_models_gray/source'+str(class_one)+str(class_two)+'.pt')
#     # # arg_parser.add_argument('--MODEL_FILE', help='A model in trained_models',default='pytorch-domain-adaptation-master/trained_models/adda.pt')
#     # # arg_parser.add_argument('--MODEL_FILE', help='A model in trained_models',default='pytorch-domain-adaptation-master/adda_aug_model/adda'+str(class_one)+str(class_two)+'.pt')
#     # arg_parser.add_argument('--batch-size', type=int, default=256)
#     # args = arg_parser.parse_args()
#     # main(args)
#     aug=False
#     CIFAR=True
#     cla_one = [0,1,2,3,4,5,6,7,8,9]
#     cla_two = [0,1,2,3,4,5,6,7,8,9]
#     for class_one in cla_one:
#         for class_two in cla_two:
#             if class_two != class_one:
#                 # print('class_one', class_one)
#                 # print('class_two', class_two)
#                 arg_parser = argparse.ArgumentParser(description='Test a model on MNIST-M')
#                 # arg_parser.add_argument('--MODEL_FILE', help='A model in trained_models',default='pytorch-domain-adaptation-master/source_models/source'+str(class_one)+str(class_two)+'.pt')
#                 # arg_parser.add_argument('--MODEL_FILE', help='A model in trained_models',default='pytorch-domain-adaptation-master/trained_models/adda.pt')
#                 # arg_parser.add_argument('--MODEL_FILE', help='A model in trained_models',default='pytorch-domain-adaptation-master/adda_model/adda'+str(class_one)+str(class_two)+'.pt')
#                 # arg_parser.add_argument('--MODEL_FILE', help='A model in trained_models',default='pytorch-domain-adaptation-master/source_models_cifar/source'+str(class_one)+str(class_two)+'.pt')

#                 arg_parser.add_argument('--MODEL_FILE', help='A model in trained_models',default='pytorch-domain-adaptation-master/adda_aug_model_cifar/adda'+str(class_one)+str(class_two)+'.pt')
#                 # arg_parser.add_argument('--MODEL_FILE', help='A model in trained_models',default='pytorch-domain-adaptation-master/adda_aug_model/adda'+str(class_one)+str(class_two)+'.pt')
#                 arg_parser.add_argument('--batch-size', type=int, default=256)
#                 args = arg_parser.parse_args()
#                 main(args)
