import torch
import sklearn.metrics as metrics
import numpy as np
from sklearn.metrics import confusion_matrix,roc_auc_score

def test_results(model, device, test_loader,num_test_samples):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    correct_clu = 0
    with torch.no_grad():
        for batch_idx, (data, _,target,pheno,__) in enumerate(test_loader):
            data = data.type(torch.FloatTensor)
            target = target.type(torch.long)
            data, target, pheno = data.to(device), target.to(device),pheno.to(device)
            [pre_label, cluster_],___= model(data, pheno)
            pred = pre_label.argmax(1, keepdim=True) # get the index of the max log-probability
            clus = cluster_.argmax(1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            correct_clu += clus.eq(target.view_as(clus)).sum().item()
            del data,target,pre_label,__
            torch.cuda.empty_cache()
    # print('correct_num:',correct)
    # print('correct_num_clu:',correct_clu)
    correct_rate = correct / num_test_samples
    model.train()
    return correct_rate



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

def LSTM_test_method(model, device, test_loader, crops):
    model.to(device)
    model.eval()
    ADHD_pred = []
    label=[]
    pred = []
    pred2 = []
    true_labels = []
    ADHD_predict=[]
    with torch.no_grad():
        for batch_idx, (data, _,target, pheno,_) in enumerate(test_loader):
            data = data.type(torch.FloatTensor)
            target = target.type(torch.long)
            pheno = pheno.type(torch.FloatTensor)
            data, pheno = data.to(device),pheno.to(device)
            pre_label, _= model(data, pheno)
            for i in range(len(pre_label)):
                label.append(target[i].cpu().item())
                ADHD_predict.append(pre_label[i].cpu())
                if pre_label[i] >= 0.5:
                    ADHD_pred.append(1)
                else:
                    ADHD_pred.append(0)

    for i in range(len(test_loader.dataset)//crops):
        ADHD_subject = ADHD_pred[i * crops:i * crops + crops]
        if np.count_nonzero(ADHD_subject) >= crops / 2:
            pred.append(1)
        else:
            pred.append(0)

        ADHD_subject_score = ADHD_predict[i * crops:i * crops + crops]
        if np.sum(ADHD_subject_score) / crops >= 0.5:
            pred2.append(1)
        else:
            pred2.append(0)

        ADHD_true_subject = label[i * crops:i * crops + crops]
        if np.count_nonzero(ADHD_true_subject) >= crops / 2:
            true_labels.append(1)
        else:
            true_labels.append(0)

    model.train()
    confusion = confusion_matrix( true_labels, pred)
    # print('confusion',confusion)
    TP_num = confusion[1, 1]
    TN_num = confusion[0, 0]
    FP_num = confusion[0, 1]
    FN_num = confusion[1, 0]
    # ACC = (TP_num + TN_num) / float(TP_num + TN_num + FP_num + FN_num)

    if float(TN_num + FP_num) == 0:
        SPE = 0
    else:
        SPE = TN_num / float(TN_num + FP_num)
    if float(TP_num + FN_num) == 0:
        REC = 0
    else:
        REC = TP_num / float(TP_num + FN_num)
    Cor_num = TP_num + TN_num
    # print('target_array', true_labels)
    # print('A_array', pred)
    AUC = roc_auc_score(true_labels, pred)
    return metrics.accuracy_score(label, ADHD_pred), metrics.accuracy_score(true_labels, pred), metrics.accuracy_score(true_labels, pred2), \
           SPE, REC, AUC

def Mutual_LSTM_test_method(model, device, test_loader, crops):
    model.to(device)
    model.eval()
    ADHD_pred = []
    label=[]
    pred = []
    pred2 = []
    true_labels = []
    ADHD_predict=[]
    with torch.no_grad():
        for batch_idx, (data, _,target, pheno,__) in enumerate(test_loader):
            data = data.type(torch.FloatTensor)
            target = target.type(torch.long)
            pheno = pheno.type(torch.FloatTensor)
            data, pheno = data.to(device),pheno.to(device)
            [pre_label, ___], _, __= model(data, pheno)
            pre_label = pre_label.cpu()
            for i in range(len(pre_label)):
                label.append(target[i].item())
                ADHD_predict.append(pre_label[i])
                if pre_label[i] >= 0.5:
                    ADHD_pred.append(1)
                else:
                    ADHD_pred.append(0)

    for i in range(len(test_loader.dataset)//crops):
        ADHD_subject = ADHD_pred[i * crops:i * crops + crops]
        if np.count_nonzero(ADHD_subject) >= crops / 2:
            pred.append(1)
        else:
            pred.append(0)

        ADHD_subject_score = ADHD_predict[i * crops:i * crops + crops]
        if np.sum(ADHD_subject_score) / crops >= 0.5:
            pred2.append(1)
        else:
            pred2.append(0)

        ADHD_true_subject = label[i * crops:i * crops + crops]
        if np.count_nonzero(ADHD_true_subject) >= crops / 2:
            true_labels.append(1)
        else:
            true_labels.append(0)

 
    model.train()
    confusion = confusion_matrix(true_labels,pred)
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
    Cor_num = TP_num + TN_num
    # print('target_array', true_labels)
    # print('A_array', pred)
    AUC = roc_auc_score(true_labels, pred)
    return metrics.accuracy_score(label, ADHD_pred),metrics.accuracy_score(true_labels, pred),\
           metrics.accuracy_score(true_labels, pred2), SPE, REC, AUC

def Mutual_LSTM_test_method_(model, device, test_loader, crops):
    model.to(device)
    model.eval()
    ADHD_pred = []
    label=[]
    pred = []
    pred2 = []
    true_labels = []
    ADHD_predict=[]
    with torch.no_grad():
        for batch_idx, (data, _,target, pheno,__) in enumerate(test_loader):
            data = data.type(torch.FloatTensor)
            target = target.type(torch.long)
            pheno = pheno.type(torch.FloatTensor)
            data, pheno = data.to(device),pheno.to(device)
            [pre_label, ___], _, __= model(data, pheno)
            pre_label = pre_label.cpu()
            for i in range(len(pre_label)):
                label.append(target[i].item())
                ADHD_predict.append(pre_label[i])
                if pre_label[i] >= 0.5:
                    ADHD_pred.append(1)
                else:
                    ADHD_pred.append(0)

    for i in range(len(test_loader.dataset)//crops):
        ADHD_subject = ADHD_pred[i * crops:i * crops + crops]
        if np.count_nonzero(ADHD_subject) >= crops / 2:
            pred.append(1)
        else:
            pred.append(0)

        ADHD_subject_score = ADHD_predict[i * crops:i * crops + crops]
        if np.sum(ADHD_subject_score) / crops >= 0.5:
            pred2.append(1)
        else:
            pred2.append(0)

        ADHD_true_subject = label[i * crops:i * crops + crops]
        if np.count_nonzero(ADHD_true_subject) >= crops / 2:
            true_labels.append(1)
        else:
            true_labels.append(0)

    # print(ADHD_pred)
    # print(pred)
    # print(true_labels)
    # # aaa=y_test==label
    # print('y_test',False in aaa)
    # print("ADHD accuracy: " + str(metrics.accuracy_score(label, ADHD_pred)))
    # print("ADHD sub accuracy: " + str(metrics.accuracy_score(true_labels, pred)))
    # print("ADHD sub2 accuracy: " + str(metrics.accuracy_score(true_labels, pred2)))
    model.train()
    return metrics.accuracy_score(label, ADHD_pred),metrics.accuracy_score(true_labels, pred),\
           metrics.accuracy_score(true_labels, pred2)
