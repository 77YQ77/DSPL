from collections import OrderedDict
import torch
import numpy as np
import copy
import heapq
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import torch.nn as nn
from model import config
args = config.Arguments()
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

threshold_value = 0.6
def transfer_one2two(pred):
    aa = torch.zeros(pred.shape[0], 2)
    for i in range(len(pred)):
        if i >= 0.5:
            aa[i][1] = pred[i]
            aa[i][0] = 1 - pred[i]
        else:
            aa[i][1] = 1 - pred[i]
            aa[i][0] = pred[i]
    return aa

def extract_cnn_feature_first(exp_1, unlabelled_loader, index):
    exp_1.eval()
    features = OrderedDict()
    labels = OrderedDict()
    datas = OrderedDict()
    datas_1 = OrderedDict()
    pheno_ = OrderedDict()
    pred = OrderedDict()
    pred_values = OrderedDict()
    with torch.no_grad():
        for batch_idx, (data_0, data_1, target,pheno, fname) in enumerate(unlabelled_loader):
            data = data_0.type(torch.FloatTensor).to(device)

            del data_0
            torch.cuda.empty_cache()
            # print(data.shape)
            target = target.type(torch.FloatTensor)
            pheno = pheno.type(torch.FloatTensor).to(device)
            outputs, _, feas = exp_1(data, pheno)
            # feas = feature[0]
            outputs_ = outputs[0].cpu()
            outputs_supp = transfer_one2two(outputs_)
            all_output = outputs_supp.float().cpu()
            all_fea = feas.float().cpu()
            del feas
            del  outputs
            for i in range(all_fea.shape[0]):
                features[fname[i]] = all_fea[i]
                pred_values[fname[i]] = outputs_[i]
                labels[fname[i]] = target[i].cpu()
                datas[fname[i]] = data[i].cpu()
                datas_1[fname[i]] = data_1[i].cpu()
                pred[fname[i]] = all_output[i].cpu()
                pheno_[fname[i]] = pheno[i].cpu()
            del all_fea
            del data
            del all_output
            torch.cuda.empty_cache()
    return features, labels, datas, datas_1, pred, pred_values, pheno_


def pseudo_generation_with_weighted_selection(model_1, dataloader, cluster_num):
    features, labels, datas, datas_1, pred, pred_values, pheno_ = extract_cnn_feature_first(model_1, dataloader, 0)
    fnames = np.array(list(labels.keys()))
    data_0 = torch.stack(list(datas.values()))
    data_1 = torch.stack(list(datas_1.values()))
    pheno = torch.stack(list(pheno_.values()))
    all_label = torch.stack(list(labels.values()))
    cf_1 = torch.stack(list(features.values()))
    all_fea_1 = torch.cat((cf_1, torch.ones(cf_1.size(0), 1)), 1)
    all_fea_1 = (all_fea_1.t() / torch.norm(all_fea_1, p=2, dim=1)).t()
    all_fea_1 = all_fea_1.float().cpu().numpy()
    output_1 = torch.stack(list(pred.values()))
    # print('output_1',output_1)
    # print('output_1', output_1.shape)
    _, pred_1 = torch.max(output_1, 1)
    true_label_1 = copy.deepcopy(output_1)
    confidence_list_1 = []
    for i in range(output_1.shape[0]):
        if np.where(output_1[i] >= threshold_value)[0].tolist() != []:
            confidence_list_1.append(i)
    if len(confidence_list_1)==0:
        for i in range(output_1.shape[0]):
            if np.where(output_1[i] >= 0.55)[0].tolist() != []:
                confidence_list_1.append(i)
    if len(confidence_list_1)==0:
        for i in range(output_1.shape[0]):
            if np.where(output_1[i] >= 0.53)[0].tolist() != []:
                confidence_list_1.append(i)
    ent_1 = torch.sum(-output_1 * torch.log(output_1 + 1e-5), dim=1) / np.log(2)
    ent_1 = ent_1.float().cpu()
    ent_dict = OrderedDict()
    for i in range(len(fnames)):
        ent_dict[fnames[i]] = ent_1[i]

    kmeans_1 = KMeans(cluster_num, random_state=0).fit(ent_1.reshape(-1, 1))
    labels_1 = kmeans_1.predict(ent_1.reshape(-1, 1))
    idx_1 = []
    for i in range(cluster_num):
        idx_ = np.where(labels_1 == i)[0]
        idx_1.append(idx_)
    iidx_1 = 0
    temp_1 = [ent_1[idx_1[i]].mean() for i in range(len(idx_1))]
    temp_array_1 = np.array(temp_1)
    iidx_1 = np.argmax(temp_array_1)
    known_idx_1 = np.where(kmeans_1.labels_ != iidx_1)[0]

    all_fea_1 = all_fea_1[known_idx_1, :]
    output_1 = output_1[known_idx_1, :]
    pred_1 = pred_1[known_idx_1]

    aff_1 = output_1.float().cpu().numpy()
    initc_1 = aff_1.transpose().dot(all_fea_1)
    initc_1 = initc_1 / (1e-8 + aff_1.sum(axis=0)[:, None])
    K_1 = output_1.size(1)
    cls_count_1 = np.eye(K_1)[pred_1].sum(axis=0)
    labelset_1 = np.where(cls_count_1 > 0)[0]
    dd_1 = cdist(all_fea_1, initc_1[labelset_1], 'cosine')
    pred_label_1 = dd_1.argmin(axis=1)
    pred_label_1 = labelset_1[pred_label_1]

    for round in range(5):
        aff_1 = np.eye(K_1)[pred_label_1]
        initc_1 = aff_1.transpose().dot(all_fea_1)
        initc_1 = initc_1 / (1e-8 + aff_1.sum(axis=0)[:, None])
        dd_1 = cdist(all_fea_1, initc_1[labelset_1], 'cosine')
        pred_label_1 = dd_1.argmin(axis=1)
        # print('pred_label_1',pred_label_1)
        pred_label_1 = labelset_1[pred_label_1]

    guess_label_1 = 2 * np.ones(len(all_label), )
    guess_label_1[known_idx_1] = pred_label_1

    two_correct_index_1 = []
    for i in range(guess_label_1.shape[0]):
        if np.array(true_label_1.argmax(1, keepdim=True)[i][0]) == guess_label_1[i]:
            two_correct_index_1.append(i)
    inddd_1 = []
    for i in two_correct_index_1:
        if i in confidence_list_1:
            inddd_1.append(i)

    aug_array_1 = np.array(inddd_1)
    if len(aug_array_1)==0:
        aug_array_1 = known_idx_1
    aug_label_1 = guess_label_1[aug_array_1]

    ppp_label_1 = all_label[aug_array_1]
    aug_fnames = fnames[aug_array_1]
    data_0 = data_0[aug_array_1]
    data_1 = data_1[aug_array_1]
    pheno = pheno[aug_array_1]
    # aug_label = pred_label[aug_array]
    aug_labels = OrderedDict()
    aug_datas_0 = OrderedDict()
    aug_datas_1 = OrderedDict()
    aug_pheno = OrderedDict()
    for i in range(len(aug_label_1)):
        aug_labels[aug_fnames[i]] = int(aug_label_1[i])
        aug_datas_0[aug_fnames[i]] = data_0[i]
        aug_datas_1[aug_fnames[i]] = data_1[i]
        aug_pheno[aug_fnames[i]] = pheno[i]
    return aug_datas_0, aug_datas_1, aug_labels, aug_pheno,len(aug_labels),np.sum(aug_label_1 == ppp_label_1.float().numpy())

def extract_cnn_feature_first_te(exp_1, unlabelled_loader, index):
    exp_1.eval()
    start_test = True
    features = OrderedDict()
    labels = OrderedDict()
    datas = OrderedDict()
    datas_1 = OrderedDict()
    pred = OrderedDict()
    pred_value = OrderedDict()
    pheno_ = OrderedDict()
    with torch.no_grad():
        for batch_idx, (data_0, data_1, target, pheno, fname) in enumerate(unlabelled_loader):
            data = data_0.type(torch.FloatTensor).to(device)
            del data_0
            torch.cuda.empty_cache()
            # print(data.shape)
            target = target.type(torch.FloatTensor)
            pheno = pheno.type(torch.FloatTensor).to(device)
            [outputs_, cluster],_, feas = exp_1(data, pheno)
            outputs_ = outputs_.cpu()
            outputs_supp = transfer_one2two(outputs_)
            all_output = outputs_supp.float().cpu()
            all_fea = feas.float().cpu()
            del feas
            del  cluster
            torch.cuda.empty_cache()
            for i in range(all_fea.shape[0]):
                pred_value[fname[i]] = outputs_[i].cpu()
                features[fname[i]] = all_fea[i]
                labels[fname[i]] = target[i].cpu()
                datas[fname[i]] = data[i].cpu()
                datas_1[fname[i]] = data_1[i].cpu()
                pred[fname[i]] = all_output[i].cpu()
                pheno_[fname[i]] = pheno[i].cpu()
            del all_fea
            del data
            del all_output
            torch.cuda.empty_cache()
    return features, labels, datas, datas_1, pred, pred_value, pheno_

def soft_pseudo_generation_with_weighted_selection(model_1, dataloader, cluster_num):
    features, labels, datas, datas_1, pred, pred_value, pheno_ = extract_cnn_feature_first(model_1, dataloader, 0)
    fnames = np.array(list(labels.keys()))
    data_0 = torch.stack(list(datas.values()))
    data_1 = torch.stack(list(datas_1.values()))
    pheno = torch.stack(list(pheno_.values()))
    all_label = torch.stack(list(labels.values()))
    cf_1 = torch.stack(list(features.values()))
    all_fea_1 = torch.cat((cf_1, torch.ones(cf_1.size(0), 1)), 1)
    all_fea_1 = (all_fea_1.t() / torch.norm(all_fea_1, p=2, dim=1)).t()
    all_fea_1 = all_fea_1.float().cpu().numpy()
    preds = torch.stack(list(pred_value.values()))

    output_1 = torch.stack(list(pred.values()))

    _, pred_1 = torch.max(output_1, 1)
    true_label_1 = copy.deepcopy(output_1)
    confidence_list_1 = []
    for i in range(output_1.shape[0]):
        if np.where(output_1[i] >= threshold_value)[0].tolist() != []:
            confidence_list_1.append(i)
    if len(confidence_list_1)==0:
        for i in range(output_1.shape[0]):
            if np.where(output_1[i] >= 0.6)[0].tolist() != []:
                confidence_list_1.append(i)
    confidence_array = np.array(confidence_list_1)

    ent_1 = torch.sum(-output_1 * torch.log(output_1 + 1e-5), dim=1) / np.log(2)
    ent_1 = ent_1.float().cpu()
    ent_dict = OrderedDict()
    for i in range(len(fnames)):
        ent_dict[fnames[i]] = ent_1[i]

    kmeans_1 = KMeans(cluster_num, random_state=0).fit(ent_1.reshape(-1, 1))
    labels_1 = kmeans_1.predict(ent_1.reshape(-1, 1))
    idx_1 = []
    for i in range(cluster_num):
        idx_ = np.where(labels_1 == i)[0]
        idx_1.append(idx_)
    iidx_1 = 0
    temp_1 = [ent_1[idx_1[i]].mean() for i in range(len(idx_1))]
    # print(temp_1)
    temp_array_1 = np.array(temp_1)
    iidx_1 = np.argmax(temp_array_1)
    # print(iidx_1)
    known_idx_1 = np.where(kmeans_1.labels_ != iidx_1)[0]

    all_fea_1 = all_fea_1[known_idx_1, :]
    output_1 = output_1[known_idx_1, :]
    pred_1 = pred_1[known_idx_1]
    # print('pred_2',pred_2)
    aff_1 = output_1.float().cpu().numpy()
    initc_1 = aff_1.transpose().dot(all_fea_1)
    initc_1 = initc_1 / (1e-8 + aff_1.sum(axis=0)[:, None])
    K_1 = output_1.size(1)
    cls_count_1 = np.eye(K_1)[pred_1].sum(axis=0)
    labelset_1 = np.where(cls_count_1 > 0)[0]
    dd_1 = cdist(all_fea_1, initc_1[labelset_1], 'cosine')
    pred_label_1 = dd_1.argmin(axis=1)
    pred_label_1 = labelset_1[pred_label_1]

    for round in range(5):
        aff_1 = np.eye(K_1)[pred_label_1]
        initc_1 = aff_1.transpose().dot(all_fea_1)
        initc_1 = initc_1 / (1e-8 + aff_1.sum(axis=0)[:, None])
        dd_1 = cdist(all_fea_1, initc_1[labelset_1], 'cosine')
        pred_label_1 = dd_1.argmin(axis=1)
        # print('pred_label_1',pred_label_1)
        pred_label_1 = labelset_1[pred_label_1]

    guess_label_1 = 2 * np.ones(len(all_label), )
    guess_label_1[known_idx_1] = pred_label_1

    two_correct_index_1 = []
    for i in range(guess_label_1.shape[0]):
        if np.array(true_label_1.argmax(1, keepdim=True)[i][0]) == guess_label_1[i]:
            two_correct_index_1.append(i)
    inddd_1 = []
    for i in two_correct_index_1:
        if i in confidence_list_1:
            inddd_1.append(i)

    aug_array_1 = np.array(inddd_1)
    if len(aug_array_1)==0:
        aug_array_1 = known_idx_1
    aug_label_1 = guess_label_1[aug_array_1]
    ppp_label_1 = all_label[aug_array_1]

    aug_fnames = fnames[aug_array_1]
    data_0 = data_0[aug_array_1]
    data_1 = data_1[aug_array_1]
    pheno = pheno[aug_array_1]
    soft_label = preds[aug_array_1]

    aug_labels = OrderedDict()
    aug_datas_0 = OrderedDict()
    aug_datas_1 = OrderedDict()
    aug_pheno = OrderedDict()
    for i in range(len(soft_label)):
        aug_labels[aug_fnames[i]] = torch.tensor(soft_label[i]).type(torch.FloatTensor)
        aug_datas_0[aug_fnames[i]] = data_0[i]
        aug_datas_1[aug_fnames[i]] = data_1[i]
        aug_pheno[aug_fnames[i]] = pheno[i]
    return aug_datas_0, aug_datas_1, aug_labels, aug_pheno,len(aug_labels),np.sum(aug_label_1 == ppp_label_1.float().numpy())


def pseudo_generation_with_weighted_selection_te(model_1, dataloader, cluster_num):
    features, labels, datas, datas_1, pred, pred_value, pheno_ = extract_cnn_feature_first_te(model_1, dataloader, 0)
    fnames = np.array(list(labels.keys()))
    data_0 = torch.stack(list(datas.values()))
    data_1 = torch.stack(list(datas_1.values()))
    all_label = torch.stack(list(labels.values()))
    cf_1 = torch.stack(list(features.values()))
    pheno = torch.stack(list(pheno_.values()))
    all_fea_1 = torch.cat((cf_1, torch.ones(cf_1.size(0), 1)), 1)
    all_fea_1 = (all_fea_1.t() / torch.norm(all_fea_1, p=2, dim=1)).t()
    all_fea_1 = all_fea_1.float().cpu().numpy()

    output_1 = torch.stack(list(pred.values()))

    _, pred_1 = torch.max(output_1, 1)
    true_label_1 = copy.deepcopy(output_1)

    confidence_list_1 = []
    for i in range(output_1.shape[0]):
        if np.where(output_1[i] > threshold_value)[0].tolist() != []:
            confidence_list_1.append(i)

    if len(confidence_list_1)==0:
        for i in range(output_1.shape[0]):
            if np.where(output_1[i] >= 0.55)[0].tolist() != []:
                confidence_list_1.append(i)
    if len(confidence_list_1)==0:
        for i in range(output_1.shape[0]):
            if np.where(output_1[i] >= 0.53)[0].tolist() != []:
                confidence_list_1.append(i)
    confidence_array = np.array(confidence_list_1)
    ent_1 = torch.sum(-output_1 * torch.log(output_1 + 1e-5), dim=1) / np.log(2)
    ent_1 = ent_1.float().cpu()

    ent_dict = OrderedDict()
    for i in range(len(fnames)):
        ent_dict[fnames[i]] = ent_1[i]

    kmeans_1 = KMeans(cluster_num, random_state=0).fit(ent_1.reshape(-1, 1))
    labels_1 = kmeans_1.predict(ent_1.reshape(-1, 1))
    idx_1 = []
    for i in range(cluster_num):
        idx_ = np.where(labels_1 == i)[0]
        idx_1.append(idx_)
    iidx_1 = 0
    temp_1 = [ent_1[idx_1[i]].mean() for i in range(len(idx_1))]

    temp_array_1 = np.array(temp_1)
    iidx_1 = np.argmax(temp_array_1)

    known_idx_1 = np.where(kmeans_1.labels_ != iidx_1)[0]

    all_fea_1 = all_fea_1[known_idx_1, :]
    output_1 = output_1[known_idx_1, :]
    pred_1 = pred_1[known_idx_1]

    aff_1 = output_1.float().cpu().numpy()
    initc_1 = aff_1.transpose().dot(all_fea_1)
    initc_1 = initc_1 / (1e-8 + aff_1.sum(axis=0)[:, None])
    K_1 = output_1.size(1)
    cls_count_1 = np.eye(K_1)[pred_1].sum(axis=0)
    labelset_1 = np.where(cls_count_1 > 0)[0]
    dd_1 = cdist(all_fea_1, initc_1[labelset_1], 'cosine')
    pred_label_1 = dd_1.argmin(axis=1)
    pred_label_1 = labelset_1[pred_label_1]

    for round in range(5):
        aff_1 = np.eye(K_1)[pred_label_1]
        initc_1 = aff_1.transpose().dot(all_fea_1)
        initc_1 = initc_1 / (1e-8 + aff_1.sum(axis=0)[:, None])
        dd_1 = cdist(all_fea_1, initc_1[labelset_1], 'cosine')
        pred_label_1 = dd_1.argmin(axis=1)
        # print('pred_label_1',pred_label_1)
        pred_label_1 = labelset_1[pred_label_1]

    guess_label_1 = 2 * np.ones(len(all_label), )
    guess_label_1[known_idx_1] = pred_label_1
   

    two_correct_index_1 = []
    for i in range(guess_label_1.shape[0]):
        if np.array(true_label_1.argmax(1, keepdim=True)[i][0]) == guess_label_1[i]:
            two_correct_index_1.append(i)
    inddd_1 = []
    for i in two_correct_index_1:
        if i in confidence_list_1:
            inddd_1.append(i)

    aug_array_1 = np.array(inddd_1)

    if len(aug_array_1)==0:
        aug_array_1 = known_idx_1
    aug_label_1 = guess_label_1[aug_array_1]

    ppp_label_1 = all_label[aug_array_1]

    acc_1 = np.sum(aug_label_1 == ppp_label_1.float().numpy()) / len(ppp_label_1)

    aug_fnames = fnames[aug_array_1]
    data_0 = data_0[aug_array_1]
    data_1 = data_1[aug_array_1]
    pheno = pheno[aug_array_1]
    # aug_label = pred_label[aug_array]
    aug_labels = OrderedDict()
    aug_datas_0 = OrderedDict()
    aug_datas_1 = OrderedDict()
    aug_pheno = OrderedDict()
    for i in range(len(aug_label_1)):
        aug_labels[aug_fnames[i]] = torch.tensor(aug_label_1[i]).type(torch.FloatTensor)
        aug_datas_0[aug_fnames[i]] = data_0[i]
        aug_datas_1[aug_fnames[i]] = data_1[i]
        aug_pheno[aug_fnames[i]] = pheno[i]

    return aug_datas_0, aug_datas_1, aug_labels,aug_pheno,len(aug_labels),np.sum(aug_label_1 == ppp_label_1.float().numpy())

def soft_pseudo_generation_with_weighted_selection_te(model_1, dataloader, cluster_num):
    features, labels, datas, datas_1, pred, pred_values, pheno_ = extract_cnn_feature_first_te(model_1, dataloader, 0)
    fnames = np.array(list(labels.keys()))
    data_0 = torch.stack(list(datas.values()))
    data_1 = torch.stack(list(datas_1.values()))
    all_label = torch.stack(list(labels.values()))
    cf_1 = torch.stack(list(features.values()))
    pheno = torch.stack(list(pheno_.values()))
    all_fea_1 = torch.cat((cf_1, torch.ones(cf_1.size(0), 1)), 1)
    all_fea_1 = (all_fea_1.t() / torch.norm(all_fea_1, p=2, dim=1)).t()
    all_fea_1 = all_fea_1.float().cpu().numpy()
    preds = torch.stack(list(pred_values.values()))
    output_1 = torch.stack(list(pred.values()))
    
    _, pred_1 = torch.max(output_1, 1)
    true_label_1 = copy.deepcopy(output_1)

    confidence_list_1 = []
    for i in range(output_1.shape[0]):
        if np.where(output_1[i] > threshold_value)[0].tolist() != []:
            confidence_list_1.append(i)

    if len(confidence_list_1)==0:
        for i in range(output_1.shape[0]):
            if np.where(output_1[i] >= 0.6)[0].tolist() != []:
                confidence_list_1.append(i)
    confidence_array = np.array(confidence_list_1)
    
    ent_1 = torch.sum(-output_1 * torch.log(output_1 + 1e-5), dim=1) / np.log(2)
    ent_1 = ent_1.float().cpu()

    ent_dict = OrderedDict()
    for i in range(len(fnames)):
        ent_dict[fnames[i]] = ent_1[i]

    kmeans_1 = KMeans(cluster_num, random_state=0).fit(ent_1.reshape(-1, 1))
    labels_1 = kmeans_1.predict(ent_1.reshape(-1, 1))
    idx_1 = []
    for i in range(cluster_num):
        idx_ = np.where(labels_1 == i)[0]
        idx_1.append(idx_)
    iidx_1 = 0
    temp_1 = [ent_1[idx_1[i]].mean() for i in range(len(idx_1))]
    
    temp_array_1 = np.array(temp_1)
    iidx_1 = np.argmax(temp_array_1)
    known_idx_1 = np.where(kmeans_1.labels_ != iidx_1)[0]

    all_fea_1 = all_fea_1[known_idx_1, :]
    output_1 = output_1[known_idx_1, :]
    pred_1 = pred_1[known_idx_1]

    aff_1 = output_1.float().cpu().numpy()
    initc_1 = aff_1.transpose().dot(all_fea_1)
    initc_1 = initc_1 / (1e-8 + aff_1.sum(axis=0)[:, None])
    K_1 = output_1.size(1)
    cls_count_1 = np.eye(K_1)[pred_1].sum(axis=0)
    labelset_1 = np.where(cls_count_1 > 0)[0]
    dd_1 = cdist(all_fea_1, initc_1[labelset_1], 'cosine')
    pred_label_1 = dd_1.argmin(axis=1)
    pred_label_1 = labelset_1[pred_label_1]

    for round in range(5):
        aff_1 = np.eye(K_1)[pred_label_1]
        initc_1 = aff_1.transpose().dot(all_fea_1)
        initc_1 = initc_1 / (1e-8 + aff_1.sum(axis=0)[:, None])
        dd_1 = cdist(all_fea_1, initc_1[labelset_1], 'cosine')
        pred_label_1 = dd_1.argmin(axis=1)
        # print('pred_label_1',pred_label_1)
        pred_label_1 = labelset_1[pred_label_1]

    guess_label_1 = 2 * np.ones(len(all_label), )
    guess_label_1[known_idx_1] = pred_label_1

    two_correct_index_1 = []
    for i in range(guess_label_1.shape[0]):
        if np.array(true_label_1.argmax(1, keepdim=True)[i][0]) == guess_label_1[i]:
            two_correct_index_1.append(i)
    inddd_1 = []
    for i in two_correct_index_1:
        if i in confidence_list_1:
            inddd_1.append(i)

    aug_array_1 = np.array(inddd_1)
    
    if len(aug_array_1)==0:
        aug_array_1 = known_idx_1
    
    aug_label_1 = guess_label_1[aug_array_1]

    ppp_label_1 = all_label[aug_array_1]

    acc_1 = np.sum(aug_label_1 == ppp_label_1.float().numpy()) / len(ppp_label_1)

    aug_fnames = fnames[aug_array_1]
    data_0 = data_0[aug_array_1]
    data_1 = data_1[aug_array_1]
    pheno = pheno[aug_array_1]
    soft_label = preds[aug_array_1]
    aug_labels = OrderedDict()
    aug_datas_0 = OrderedDict()
    aug_datas_1 = OrderedDict()
    aug_pheno = OrderedDict()
    for i in range(len(aug_label_1)):
        aug_labels[aug_fnames[i]] = torch.tensor(soft_label[i]).type(torch.FloatTensor)
        aug_datas_0[aug_fnames[i]] = data_0[i]
        aug_datas_1[aug_fnames[i]] = data_1[i]
        aug_pheno[aug_fnames[i]] = pheno[i]
    class_nums = np.array(list(aug_labels.values()))
    return aug_datas_0, aug_datas_1, aug_labels,aug_pheno,len(aug_labels),np.sum(aug_label_1 == ppp_label_1.float().numpy())