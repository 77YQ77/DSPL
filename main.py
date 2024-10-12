#### This is the implementation of NatIMG_FL ####
import warnings
import argparse
warnings.filterwarnings('ignore')
from collections import OrderedDict
import numpy as np
import copy
import heapq
from tqdm import tqdm, trange
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from resnetv2_temp import resnet18_,resnet50_
from test_model_att import contrastive_global_test_method,train_test

# from test_model_att import contrastive_test_method,contrastive_global_test_method,train_test
from utils import *
from distill import *
data_name="cifar-10"
frozen =True
num_clients=5
def lcoal_training(model,optim,lr_schedule,idx,source_loader,labeled_loader,unlabeled_loader, test_loader, epochs,iterations,comm):
    # print('idx',idx)
    data_name="cifar-10"
    gpus_num = 5
    gpu_id = 1+(idx % gpus_num)
    # if gpu_id==:
    #     gpu_id=1
    # if comm==0:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # gpus_num = 7
    # gpu_id = idx % gpus_num
    # if gpu_id==0 and idx>0:
    #     gpu_id=1
    # if comm==0:
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(7)
    from utils import loop_iterable
    from contrastive_loss import SupConLoss
    import torch
    torch.autograd.set_detect_anomaly(True)
    import models
    import torch.nn as nn
    import torch.nn.functional as F
# from pseudo_labeling_gpu import obtain_cnn_centroid_feature,obtain_cnn_source_centroid_feature,obtain_cnn_confidence_index
    criterion_1 = torch.nn.CrossEntropyLoss()
    criterion_2 = SupConLoss()

    try:
        torch.multiprocessing.set_start_method('fork',force=True)
    except RuntimeError: 
        pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.cpu()
    model = model.to(device)
    if comm>0 and idx ==6:
        AUC_best=0
        model_test = copy.deepcopy(model)
        model_test.load_state_dict(torch.load('global_model/'+str(data_name)+'_att_global_test_'+str(comm-1)+'.pt'))
        test_ACC, test_SPE, test_REC, test_AUC_ = contrastive_global_test_method(model_test, test_loader,device)
        if test_AUC_ >=AUC_best:
            AUC_best=test_AUC_
            torch.save(model_test.state_dict(),'global_model/'+str(data_name)+'_avg_global_best'+'.pt')  
        tqdm.write(f'commu_round {comm-1:01d}: test_ACC_avg={test_ACC:.4f}, ' 
                                f'test_SPE_avg= {test_SPE:.4f}, '
                                f'test_REC_avg= {test_REC:.4f}, '
                                f'test_AUC_avg= {test_AUC_:.4f}')
        del model_test
        torch.cuda.empty_cache()
    
    if comm>0 and idx==7:
        # print('true')
        gems_test_best=0
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_t', default='wrn16x2', type=str)
        parser.add_argument('--model', default='wrn28x2', type=str)
        parser.add_argument('--qk_dim', default=128, type=int)
        args = parser.parse_args()
        # print('args',args)
        if data_name == 'mnist':
            model_global = models.__dict__['wrn28x2'](num_classes=50)
            image_size=28
        if data_name == 'cifar-10':
            model_global = resnet18_(pretrained=False)
            image_size=224
        if data_name == 'imagenet':
            model_global = resnet50_(pretrained=True)
            image_size=224
        # print('model_true')
        args.guide_layers = LAYER[args.model_t]
        args.hint_layers = LAYER[args.model]
        data_ev = torch.randn(2, 3, image_size, image_size).to(device)
        # print('data_ev_done')
        model_global.to(device)
        model.eval()
        model_global.eval()
        with torch.no_grad():
            feat_t, _,_,_ = model(data_ev, is_feat=True)
            feat_s, _,_,_ = model_global(data_ev, is_feat=True)
        # model_global.train()
        model.train()
        model_global.train()
        args.s_shapes = [feat_s[i].size() for i in args.hint_layers]
        args.t_shapes = [feat_t[i].size() for i in args.guide_layers]
        # for i in args.hint_layers:
        #     print('feat_s[i].size()',feat_s[i].size())
        # for i in args.guide_layers:
        #     print('feat_t[i].size()',feat_t[i].size())
        
        args.n_t, args.unique_t_shapes = unique_shape(args.t_shapes)
        criterion_kd = AFD(args)
        # print('criterion_kd_done')
        criterion_kl = DistillKL_(4)
        # print('criterion_kl_done')
        criterion_kd.to(device)
        # print('criterion_kd_device_done')
        model_temp=copy.deepcopy(model)
        model_temp.load_state_dict(torch.load('global_model/'+str(data_name)+'_att_global_test_'+str(comm-1)+'.pt'))
        if frozen==True:
            kk=[list(model_temp.state_dict().keys())[-2],list(model_temp.state_dict().keys())[-1]]
            vv=[model.state_dict()[list(model_temp.state_dict().keys())[-2]],model_temp.state_dict()[list(model_temp.state_dict().keys())[-1]]]
            dictionary = dict(zip(kk, vv))
            model_global.load_state_dict(dictionary,strict=False)
            list(model_global.parameters())[-1].requires_grad = False
            list(model_global.parameters())[-2].requires_grad = False
        # print('global_model_frozen')
        global_epoch=30
        trainable_list = nn.ModuleList([])
        trainable_list.append(model_global)
        trainable_list.append(criterion_kd)
        # model_global=copy.deepcopy(model)
        optim_global = torch.optim.Adam(trainable_list.parameters())
        lr_schedule_global = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_global, patience=1, verbose=True)
        # print('att_training_begin')
        global_model = gems_att_training(comm,model_global,model_temp,optim_global,device,lr_schedule_global,source_loader, global_epoch,criterion_1,criterion_2,criterion_kd,criterion_kl,test_loader)
        test_ACC, test_SPE, test_REC, test_AUC = contrastive_global_test_method(global_model, test_loader,device)
        tqdm.write(f'commu_round {comm-1:01d}: test_ACC_gems={test_ACC:.4f}, ' 
                                f'test_SPE_gems= {test_SPE:.4f}, '
                                f'test_REC_gems= {test_REC:.4f}, '
                                f'test_AUC_gems= {test_AUC:.4f}')

    for epoch in range(1, epochs+1):
        # print('epoch=',epoch,'idx=',idx)
        #     batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))
        batch_iterator = zip(loop_iterable(source_loader),loop_iterable(labeled_loader) ,loop_iterable(unlabeled_loader))
        for ind in range(iterations):
            # print('comm=',comm,'epoch=',epoch,'iterations=',ind,'idx=',idx)
            (x, y_true), (labeled_x,labeled_aug_data, y_target),(unlabeled_x,unlabeled_aug_data, unlabeled_label) = next(batch_iterator)
            # print('eval1',idx)
            x, y_true = x.to(device), y_true.to(device)
            # print('eval2',idx)
            labeled_x, y_target =  labeled_x.to(device), y_target.to(device)
            # print('eval3',idx)
            unlabeled_x, unlabeled_aug_data,unlabeled_label=unlabeled_x.to(device),unlabeled_aug_data.to(device),unlabeled_label.to(device)
            # print('eval4',idx)
            labeled_all = torch.cat([x, labeled_x], dim=0)
            target_all = torch.cat([y_true, y_target], dim=0)
            # print('data_to_device',idx)
            # print('labeled_all',labeled_all.shape)
            # print('target_all',target_all)
            labeled_all = labeled_all.to(torch.float32)
            target_all = target_all.to(torch.long)
            _,y_pred,_,__ = model(labeled_all)
            loss_ce = criterion_1(y_pred, target_all)
            if epoch ==1 and ind<200:
                # print('loss_ce',idx, loss_ce.item())
                loss = loss_ce
                optim.zero_grad()
                loss.backward()
                optim.step()
                del loss
                torch.cuda.empty_cache()
                # print('initial_warm_up_done',idx)
            else:
                ###### confident_supervised_contrastive_loss_calculation ####
                condifent_index, unconfident_index, pseudo_label = obtain_cnn_confidence_index(model, unlabeled_x, unlabeled_label, ind,epoch,device,True)
                # print('unlabeled_x',unlabeled_x.shape,idx)
                # print('condifent_index_done',idx)
                # print('condifent_index',condifent_index,idx)
                # print('unconfident_index',unconfident_index,idx)
                # unlabeled_x_temp = copy.deepcopy(unlabeled_x.cpu())
                # unlabeled_x_=np.array(unlabeled_x_temp)
                # print('unlabeled_x_',unlabeled_x.shape,idx)
                confident_sample = torch.tensor(unlabeled_x[condifent_index])
                # print('confident_sample',confident_sample.shape,idx)
                confident_aug_sample = unlabeled_aug_data[condifent_index]
                # print('confident_aug_sample',confident_aug_sample.shape,idx)
                confident_sample=confident_sample.to(device)
                # print('confident_sample',confident_sample.shape,idx)
                confident_aug_sample=confident_aug_sample.to(device)
                # print('confident_aug_sample',confident_aug_sample.shape,idx)
                confident_sample_more = torch.cat([confident_sample,labeled_x])
                # print('confident_sample_more',confident_sample_more.shape,idx)
                labeled_aug_data=labeled_aug_data.to(device)
                # print('labeled_aug_data',labeled_aug_data.shape,idx)
                confident_sample_aug_more = torch.cat([confident_aug_sample,labeled_aug_data])
                # del confident_aug_sample, labeled_aug_data
                # torch.cuda.empty_cache()
                unconfident_sample = torch.tensor(unlabeled_x[unconfident_index])
                unconfident_aug_sample = unlabeled_aug_data[unconfident_index]
                images = torch.cat([confident_sample_more, confident_sample_aug_more], dim=0)
                # print('images',images.shape,idx)
                # del confident_sample_aug_more
                # torch.cuda.empty_cache()
                if images.shape[0]!=0:
                    pseudo_label = torch.tensor(pseudo_label).to(device)
                    confident_label_more = torch.cat([pseudo_label,y_target])
                    if torch.cuda.is_available():
                        images = images.cuda(non_blocking=True)
                        confident_label_more = confident_label_more.cuda(non_blocking=True)
                    bsz_1 = confident_sample_more.shape[0]
                    _,_, _, features_norm = model(images)
                    
                    f1, f2 = torch.split(features_norm, [bsz_1, bsz_1], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    del _,f1, f2,features_norm
                    torch.cuda.empty_cache()
                    sup_condifent_loss_contrastive = criterion_2(features, confident_label_more)
                    # print('sup_condifent_loss_contrastive',sup_condifent_loss_contrastive,idx)
                ###### unconfident_contrastive_loss_calculation ####

                images_unconfident = torch.cat([unconfident_sample, unconfident_aug_sample], dim=0)
                # print('images_unconfident',images_unconfident.shape,idx)
                if images_unconfident.shape[0]!=0:
                    if torch.cuda.is_available():
                        images_unconfident = images_unconfident.to(device)
                        # print('images_unconfident',images_unconfident.shape,idx)
                        # labels = labels.cuda(non_blocking=True)
                    bsz = unconfident_sample.shape[0]
                    _,_, __, features_norm = model(images_unconfident)
                    f1_unconfident, f2_unconfident = torch.split(features_norm, [bsz, bsz], dim=0)
                    features_unconfident = torch.cat([f1_unconfident.unsqueeze(1), f2_unconfident.unsqueeze(1)], dim=1)
                    # print('features_unconfident',features_unconfident.shape,idx)
                    del _,__,f1_unconfident, f2_unconfident,features_norm
                    torch.cuda.empty_cache()
                    unconfident_unsup_loss_contrastive = criterion_2(features_unconfident)
                    # print('unconfident_unsup_loss_contrastive',unconfident_unsup_loss_contrastive,idx)
                    del features_unconfident
                    torch.cuda.empty_cache()

                ###### supervised_contrastive_centroid_alignment ####
                # print('begin_sor_centroid',idx)
                # x = x.type(torch.FloatTensor)
                sor_centroid, _ = obtain_cnn_source_centroid_feature(model,x,y_true,device)
                # print('sor_centroid',sor_centroid.shape,idx)
                centroid_label = torch.tensor([0,1])
                if confident_sample.shape[0]!=0 and unconfident_sample.shape[0]!=0:
                    confident_unlabeled_centroid,_ = obtain_cnn_centroid_feature(model,confident_sample,device)
                    unconfident_unlabeled_centroid,_ = obtain_cnn_centroid_feature(model,unconfident_sample,device)
                    # print('confident_unlabeled_centroid_done',idx)
                    # print('unconfident_unlabeled_centroid_done',idx)
                # unlabeled_centroid, _ = obtain_cnn_centroid_feature(model,unlabeled_x)
                
                    confident_unlabeled_centroid = torch.tensor(confident_unlabeled_centroid)
                    unconfident_unlabeled_centroid = torch.tensor(unconfident_unlabeled_centroid)
                    sor_centroid = torch.tensor(sor_centroid)
                    centroid_features_1 = torch.cat([confident_unlabeled_centroid.unsqueeze(1),sor_centroid.unsqueeze(1)], dim=1)
                    centroid_features_2 = torch.cat([unconfident_unlabeled_centroid.unsqueeze(1),sor_centroid.unsqueeze(1)], dim=1)
                # print('centroid_features',centroid_features.shape)
                    centroid_contrastive_1= criterion_2(centroid_features_1,centroid_label)
                    centroid_contrastive_2= criterion_2(centroid_features_2,centroid_label)
                    # print('centroid_contrastive_1',centroid_contrastive_1.item())
                    # print('centroid_contrastive_2',centroid_contrastive_2.item())
                    centroid_contrastive = 0.5* centroid_contrastive_1+ 0.5*centroid_contrastive_2

                if confident_sample.shape[0]==0 and unconfident_sample.shape[0]!=0:
                    # confident_unlabeled_centroid,_ = obtain_cnn_centroid_feature(model,confident_sample)
                    unconfident_unlabeled_centroid,_ = obtain_cnn_centroid_feature(model,unconfident_sample,device)
        
                    unconfident_unlabeled_centroid = torch.tensor(unconfident_unlabeled_centroid)
                    sor_centroid = torch.tensor(sor_centroid)
                    centroid_features_2 = torch.cat([unconfident_unlabeled_centroid.unsqueeze(1),sor_centroid.unsqueeze(1)], dim=1)
                    centroid_contrastive_2= criterion_2(centroid_features_2,centroid_label)
                    # print('centroid_contrastive_1',centroid_contrastive_1.item())
                    # print('centroid_contrastive_2',centroid_contrastive_2.item())
                    centroid_contrastive = centroid_contrastive_2
                    # print('centroid_contrastive',centroid_contrastive.shape)
                
                if confident_sample.shape[0]!=0 and unconfident_sample.shape[0]==0:
                    confident_unlabeled_centroid,_ = obtain_cnn_centroid_feature(model,confident_sample,device)
                    confident_unlabeled_centroid = torch.tensor(confident_unlabeled_centroid)
                    sor_centroid = torch.tensor(sor_centroid)
                    centroid_features_1 = torch.cat([confident_unlabeled_centroid.unsqueeze(1),sor_centroid.unsqueeze(1)], dim=1)
                    centroid_contrastive_1= criterion_2(centroid_features_1,centroid_label)
                    centroid_contrastive = centroid_contrastive_1

                if images_unconfident.shape[0]!=0 and images.shape[0]!=0:
                    loss = loss_ce + 0.5*sup_condifent_loss_contrastive +  0.5*unconfident_unsup_loss_contrastive + 0.5*centroid_contrastive
                if images_unconfident.shape[0]==0 and images.shape[0]!=0:
                    loss = loss_ce + 0.5*sup_condifent_loss_contrastive  +  0.5*centroid_contrastive
                    # print('loss_ce',loss_ce.item())
                    # print('condifent_loss_contrastive',sup_condifent_loss_contrastive.item())
                    # print('centroid_contrastive',centroid_contrastive.item())
                if images_unconfident.shape[0]!=0 and images.shape[0]==0:
                    loss = loss_ce + 0.5*unconfident_unsup_loss_contrastive  + 0.5*centroid_contrastive
                    # print('loss_ce',loss_ce.item())
                    # print('unconfident_unsup_loss_contrastive',unconfident_unsup_loss_contrastive.item())
                    # print('centroid_contrastive',centroid_contrastive.item())
                # print('local_loss',loss.item())
                # print('local_training_loss',idx,loss.item())
                if optim is not None:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    del loss
                    torch.cuda.empty_cache()
        # test_AUC = train_test(model, device, test_loader)
        # print('local_test_AUC',idx, test_AUC)
    model.cpu()
    torch.save(model.state_dict(),'local_model/'+'att_local_test_'+str(idx)+'.pt')  
    # print('save_model_done')


def obtain_cnn_confidence_index(exp_1, data_0, target,ind,epoch,device,entro):
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_1.eval()
    features_list = []
    labels_list = []
    pred_list = []
    import torch
    # import torch.nn as nn
    with torch.no_grad():
        data = data_0.type(torch.FloatTensor).to(device)
        del data_0
        torch.cuda.empty_cache()
        # print(data.shape)
        target = target.type(torch.FloatTensor)
        _,outputs, feas,_ = exp_1(data)
        # feas = exp_1.classifier[2](data)
        # print('feas',feas.shape)
        # x = exp_1.feature_extractor(data)
        # x = x.view(data.shape[0], -1)
        # # print('feas_',x.shape)
        # for i in range(len(exp_1.classifier)):
        #     # print('i',i)
        #     x = exp_1.classifier[i](x)
        #     if i == 2:
        #         feas = x
        # print(feas.shape)
        outputs = outputs.float().cpu()
        all_fea = feas.float().cpu()
        for i in range(len(outputs)):
            pred_list.append(outputs[i])
            features_list.append(all_fea[i])
            labels_list.append(target[i])
        # outputs = outputs.float().cpu()
        # all_fea = feas.float().cpu()
        # for i in range(len(outputs)):
        #     pred_list.append(outputs[i])
        #     features_list.append(all_fea[i])
        #     labels_list.append(target[i])
    if entro==True:
        condifent_index, pseudo_label = pseudo_generation_with_entropy_selection(pred_list, features_list, labels_list,ind,epoch)
    else:
        condifent_index, pseudo_label = pseudo_generation_with_weighted_selection(pred_list, features_list, labels_list,ind,epoch)
    # condifent_index, pseudo_label = pseudo_generation_with_entropy_selection(pred_list, features_list, labels_list,ind,epoch)
    # print('condifent_index',len(condifent_index))
    # print('pseudo_label',len(pseudo_label))
    all_index = [i for i in range(len(features_list))]
    confident_idx =[]
    for i in condifent_index:
        confident_idx.append(i)
        all_index.remove(i)
    unconfident_index = all_index
    # print('unconfident_index',unconfident_index)
    
    return  confident_idx, unconfident_index, pseudo_label

def obtain_cnn_centroid_feature(exp_1, data_0,device):
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import torch
    # import torch.nn as nn
    exp_1.eval()
    features_list = []
    pred_list = []
    with torch.no_grad():
        data = data_0.type(torch.FloatTensor).to(device)
        del data_0
        torch.cuda.empty_cache()
        # print(data.shape)
        _,outputs, feas,_ = exp_1(data)
        # feas = exp_1.classifier[2](data)
        # print('feas',feas.shape)
        # x = exp_1.feature_extractor(data)
        # x = x.view(data.shape[0], -1)
        # # print('feas_',x.shape)
        # for i in range(len(exp_1.classifier)):
        #     # print('i',i)
        #     x = exp_1.classifier[i](x)
        #     if i == 2:
        #         feas = x
        # print(feas.shape)
        outputs = outputs.float().cpu()
        all_fea = feas.float().cpu()
        for i in range(len(outputs)):
            pred_list.append(outputs[i])
            features_list.append(all_fea[i])

    centroid, feature = extract_centroid_feature(pred_list, features_list)
    return  centroid, feature

def obtain_cnn_source_centroid_feature(exp_1, data,target,device):
    import torch
    # import torch.nn as nn
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data=data.cpu()
    # data = data.type(torch.FloatTensor).to(device)
    exp_1.eval()
    features_list = []
    pred_list = []
    target_list=[]
    # print('begin_eval_1')
    with torch.no_grad():
        # print('begin_eval_2')
        data = data.to(torch.float32)
        # data = data.type(torch.FloatTensor) ### here
        # print('begin_eval_3')
        # print('data',data.shape)
        # target = target.type(torch.FloatTensor).to(device) ### here
        # print('target',target.shape)
        # del data_0
        # torch.cuda.empty_cache()
        # print(data.shape)
        _,outputs, feas,_ = exp_1(data)
        # print('get_outputs',outputs.shape)
        # feas = exp_1.classifier[2](data)
        # print('feas',feas.shape)
        # x = exp_1.feature_extractor(data)
        # x = x.view(data.shape[0], -1)
        # # print('feas_',x.shape)
        # for i in range(len(exp_1.classifier)):
        #     # print('i',i)
        #     x = exp_1.classifier[i](x)
        #     if i == 2:
        #         feas = x
        # print(feas.shape)
        outputs = outputs.float().cpu() # here
        all_fea = feas.float().cpu() #here
        for i in range(len(outputs)):
            pred_list.append(outputs[i])
            features_list.append(all_fea[i])
            # features_list.append(feas[i])
            target_list.append(target[i].cpu())
    
    # print('begin_extract_source_labeled_centroid_feature')
    centroid, feature = extract_source_labeled_centroid_feature(pred_list, features_list,target_list)
    return  centroid, feature

def extract_centroid_feature(pred_list, feature_list):
    import torch
    import torch.nn as nn
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf_1 = torch.stack(list(feature_list)).cpu()
    all_fea_1 = torch.cat((cf_1, torch.ones(cf_1.size(0), 1)), 1)
    all_fea_1 = (all_fea_1.t() / torch.norm(all_fea_1, p=2, dim=1)).t()
    all_fea_1 = all_fea_1.float().cpu().numpy()
    output_1 = torch.stack(list(pred_list))
    output_1 = nn.Softmax()(output_1)
    _, pred_1 = torch.max(output_1, 1)
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
    # print('pred_label_1',pred_label_1)
    return initc_1, all_fea_1

def extract_source_labeled_centroid_feature(pred_list, feature_list,label_list):
    import torch
    import torch.nn as nn
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_label = torch.stack(list(label_list))
    cf_1 = torch.stack(list(feature_list))
    all_fea_1 = torch.cat((cf_1, torch.ones(cf_1.size(0), 1)), 1)
    all_fea_1 = (all_fea_1.t() / torch.norm(all_fea_1, p=2, dim=1)).t()
    all_fea_1 = all_fea_1.float().cpu().numpy()
    output_1 = torch.stack(list(pred_list))
    output_1 = nn.Softmax()(output_1)
    # _, pred_1 = torch.max(output_1, 1)
    pred_1 = all_label.to(torch.int64)
    # print('pred_1',pred_1)
    # print('pred_2',pred_2)
    aff_1 = output_1.float().cpu().numpy() ####here
    # aff_1 = output_1
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
    # print('pred_label_1',pred_label_1)
    return initc_1, all_fea_1

def pseudo_generation_with_weighted_selection(pred_list, feature_list, label_list,ind,epoch):
    import torch
    import torch.nn as nn
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_label = torch.stack(list(label_list))
    cf_1 = torch.stack(list(feature_list))
    all_fea_1 = torch.cat((cf_1, torch.ones(cf_1.size(0), 1)), 1)
    all_fea_1 = (all_fea_1.t() / torch.norm(all_fea_1, p=2, dim=1)).t()
    all_fea_1 = all_fea_1.float().cpu().numpy()
    output_1 = torch.stack(list(pred_list))
    output_1 = nn.Softmax()(output_1)
    # print('output_1',output_1)
    # print('output_1', output_1.shape)
    _, pred_1 = torch.max(output_1, 1)
    true_label_1 = copy.deepcopy(output_1)
    # print('pred_1',pred_1)
    # print('true_label_1', len(np.unique(all_label)))
    confidence_list_1 = []
    for i in range(output_1.shape[0]):
        if np.where(output_1[i] >= threshold_value)[0].tolist() != []:
            confidence_list_1.append(i)
    if len(confidence_list_1)==0:
        for i in range(output_1.shape[0]):
            if np.where(output_1[i] >= 0.6)[0].tolist() != []:
                confidence_list_1.append(i)
    if len(confidence_list_1)==0:
        for i in range(output_1.shape[0]):
            if np.where(output_1[i] >= 0.55)[0].tolist() != []:
                confidence_list_1.append(i)
    # print('confidence_list_1', confidence_list_1)
    confidence_array = np.array(confidence_list_1)
    condi_label = all_label[np.array(confidence_array)]
    # print('condi_label', len(np.unique(condi_label)))

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
#    print('all_fea_1',all_fea_1.shape)
#    print('initc_1',initc_1.shape)
#    print('dd_1',dd_1)
    # print('pred_label_1',pred_label_1)

    # acc_1 = np.sum(guess_label_1 == all_label.float().numpy()) / len(all_label_idx_1)
    two_correct_index_1 = []
    for i in range(pred_label_1.shape[0]):
        if np.array(true_label_1.argmax(1, keepdim=True)[i][0]) == pred_label_1[i]:
            two_correct_index_1.append(i)
    # print('two_correct_index_1',two_correct_index_1)
    inddd_1 = []
    for i in two_correct_index_1:
        if i in confidence_list_1:
            inddd_1.append(i)

    aug_array_1 = np.array(inddd_1)
    # print('aug_array_1', aug_array_1)
    if epoch==1:
#    aug_label_1 = pred_label_1[aug_array_1]
        aug_label_1 = all_label[aug_array_1]
    if epoch>1:
        if ind<=200:
            aug_label_1 = all_label[aug_array_1]
        else:
            aug_label_1 = pred_label_1[aug_array_1]
    # print('aug_label_1_unique',aug_label_1)
#    ppp_label_1 = all_label[aug_array_1]
    # print('ppp_label_1',ppp_label_1)
#    acc_1 = np.sum(aug_label_1 == ppp_label_1.float().numpy()) / len(ppp_label_1)
#    print('acc_1', acc_1)
    return aug_array_1, aug_label_1


def pseudo_generation_with_entropy_selection(pred_list, feature_list, label_list,ind,epoch):
    import torch
    import torch.nn as nn
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_label = torch.stack(list(label_list))
    cf_1 = torch.stack(list(feature_list))
    all_fea_1 = torch.cat((cf_1, torch.ones(cf_1.size(0), 1)), 1)
    all_fea_1 = (all_fea_1.t() / torch.norm(all_fea_1, p=2, dim=1)).t()
    all_fea_1 = all_fea_1.float().cpu().numpy()
    output_1 = torch.stack(list(pred_list))
    output_1 = nn.Softmax()(output_1)
    # print('output_1',output_1)
    # print('output_1', output_1.shape)
    _, pred_1 = torch.max(output_1, 1)
    true_label_1 = copy.deepcopy(output_1)
    # print('pred_1',pred_1)
    # print('true_label_1', len(np.unique(all_label)))

    ent_1 = torch.sum(-output_1 * torch.log(output_1 + 1e-5), dim=1) / np.log(2)
    ent_1 = ent_1.float().cpu()
    ent_dict = []
    for i in range(len(ent_1)):
        ent_dict.append(ent_1[i])
    ent_dict = np.array(ent_dict)

    kmeans_1 = KMeans(2, random_state=0).fit(ent_1.reshape(-1, 1))
    labels_1 = kmeans_1.predict(ent_1.reshape(-1, 1))
    idx_1 = []
    for i in range(2):
        idx_ = np.where(labels_1 == i)[0]
        idx_1.append(idx_)
    iidx_1 = 0
    temp_1 = [ent_1[idx_1[i]].mean() for i in range(len(idx_1))]
    # print(temp_1)
    temp_array_1 = np.array(temp_1)
    iidx_1 = np.argmax(temp_array_1)
    # print(iidx_1)
    known_idx_1 = np.where(kmeans_1.labels_ != iidx_1)[0]
    uncertain_idx_1 = np.where(kmeans_1.labels_ == iidx_1)[0]

    all_fea_1 = all_fea_1[known_idx_1, :]
    output_1 = output_1[known_idx_1, :]
    pred_1 = pred_1[known_idx_1]
    all_label_idx_1 = all_label[known_idx_1]
    ENT_THRESHOLD_1 = (kmeans_1.cluster_centers_).mean()

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
    
    if epoch==1:
#    aug_label_1 = pred_label_1[aug_array_1]
        aug_label_1 = all_label[known_idx_1]
    if epoch>1:
        if ind<=200:
            aug_label_1 = all_label[known_idx_1]
        else:
            aug_label_1 = pred_label_1
    # print('aug_label_1_unique',aug_label_1)

    # guess_label_1 = 2 * np.ones(len(all_label), )
    # guess_label_1[known_idx_1] = pred_label_1


#    ppp_label_1 = all_label[aug_array_1]
    # print('ppp_label_1',ppp_label_1)
#    acc_1 = np.sum(aug_label_1 == ppp_label_1.float().numpy()) / len(ppp_label_1)
#    print('acc_1', acc_1)
    return known_idx_1, aug_label_1


def client_prediction(data,label,model):
    logit=[]
    preds=[]
    ent=[]
    incor_ind_list=[]
    for i in range(num_clients): ### 全部data
        model.load_state_dict(torch.load('local_model/'+'att_local_test_'+str(idx)+'.pt'))
        with torch.no_grad():
            y_pred,_,__ = model(data)
            y_pred = nn.Softmax()(y_pred)
            y_preds = y_pred.max(1)[1]
#             print('y_pred',y_preds)
#             print('label',label)
            incor_idx = torch.where(y_preds != label)[0].tolist()
#             print('incor_idx',incor_idx)
            ent_1 = torch.sum(-y_pred * torch.log(y_pred + 1e-5), dim=1) / np.log(2)
#             print(ent_1)
            ent_1 = ent_1.float().cpu()
            ent_1[incor_idx]=0
            logit.append(np.array(y_pred.cpu()))
            preds.append(np.array(y_preds.cpu()))
            ent.append(np.array(ent_1))
            incor_ind_list.append(incor_idx)
    logit_ = torch.tensor(logit)
#     print('logit_',logit_.shape)
    ent_ = torch.tensor(ent)
#     print('ent_',ent_.shape)
    xxx=[]
    for i in range(logit_.shape[0]):
        d=logit_[i].T*ent_[i]
        x=d.T
        xxx.append(np.array(x))
    weighted_logit=torch.mean(torch.tensor(xxx),0)
    # preds_ =torch.tensor(preds)
    return weighted_logit

def merge(d1, d2): 
    d = {**d1, **d2}
    return d


def gems_training(comm_round,model,optim,device,lr_schedule,global_loader, global_epoch,criterion_1):
    model.to(device)
    for epoch in range(1, global_epoch+1):
        # print('epoch',epoch)
        cor_idx_list=[]
        logits_list=[]
        #     batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))
        # batch_iterator = zip(loop_iterable(labeled_loader) ,loop_iterable(unlabeled_loader))
        for ind, (data,label) in enumerate(global_loader):
            labeled_all,target_all =  data.to(device), label.to(device)
            bsz = labeled_all.shape[0]
            # print('labeled_all',labeled_all.shape)
            # print('target_all',target_all)
            labeled_all = labeled_all.to(torch.float32)
            target_all = target_all.to(torch.long)
            y_preds,_,__ = model(labeled_all)
            y_pred = y_preds.max(1)[1]
            true_idx = torch.where(torch.eq(y_pred,target_all)==True)[0].tolist()
#             print('true_idx',len(true_idx))
            false_idx = [i for i in range(bsz)]
#             print('len',len(true_idx))
            if len(true_idx)!=0 and len(true_idx)!=bsz:
#                 print('len(true_idx)!=0 and len(true_idx)!=bsz')
                for i in true_idx:
                    false_idx.remove(i)
                    cor_idx_list.append(bsz*ind+i) ### 需要check ind对不对
                    logits_list.append(y_preds[i])
#                 print('cor_idx_list',cor_idx_list)
                true_pred=y_preds[true_idx].clone()
                true_label =target_all[true_idx].clone()
#                 print('true_pred',true_pred.shape)
                loss_true1 =criterion_1(true_pred, true_label) ### 预测对的数据
                if comm_round==0:
                    logits_pool = dict(zip(cor_idx_list,logits_list))
#                     print('logits_pool',logits_pool.keys())
                    np.save('global_model/att_global_test_'+'logit_dict.npy', logits_pool) 
                else:
                    logits_pool_pre = np.load('global_model/att_global_test_'+'logit_dict.npy',allow_pickle=True).item()
                    logits_pool = dict(zip(cor_idx_list,logits_list))
#                     print('logits_pool',logits_pool.keys())
                    logits_pool = merge(logits_pool_pre, logits_pool) 
#                     print('merged_logits_pool',logits_pool.keys())
                    np.save('global_model/att_global_test_'+'logit_dict.npy', logits_pool) 
                false_logit_list=[]
                fale_pred_list=[]
                false_label_list=[]
                length_false_idx=len(false_idx)
#                 print('false_idx_be',len(false_idx))
                for k in false_idx: ### 预测错的数据
                    tem=bsz*ind+k
                    if tem in logits_pool.keys(): ### 存在于logit pool中
#                         print('k',k)
                        false_logit_list.append(np.array(logits_pool[tem].cpu().detach())) ### 需要check ind对不对
                        fale_pred_list.append(np.array(y_preds[k].cpu().detach()))
                        false_label_list.append(target_all[k].item())
                        false_idx.remove(k) ### 不存在于logit pool且预测错的
#                 print('false_idx_af',len(false_idx))
#                 print('false_logit_list',len(false_logit_list))
                if len(false_logit_list)!=0 and len(false_logit_list)!=length_false_idx: # 部分存在于logit pool
                    false_logit=torch.tensor(false_logit_list) ### logit pool中的logit
                    false_pred=torch.tensor(fale_pred_list)
                    false_label=torch.tensor(false_label_list)
#                     print('false_label',false_label)
#                     print('false_pred',false_pred)
                    loss_false_ce = criterion_1(false_pred, false_label) ####这里是空的
                    false_pred_ = F.log_softmax(false_pred, dim=-1)
                    false_logit_= F.softmax(false_logit, dim=-1)
                    loss_klx = F.kl_div(false_pred_, false_logit_, reduction='mean')
                    loss_true2=0.5*loss_false_ce+0.5*loss_klx
                    labeled_all_false = labeled_all[false_idx].clone() #### this
                    label_false = target_all[false_idx].clone()
                    model_temp=copy.deepcopy(model)
                    weighted_logits= client_prediction(labeled_all_false,label_false,model_temp)
                    pred_false=y_preds[false_idx].to(device)
                    pred_false_ = F.log_softmax(pred_false, dim=-1)
                    loss_false = F.kl_div(pred_false_, weighted_logits.to(device), reduction='mean')
                    # print('loss_true1',loss_true1)
                    # print('loss_true2',loss_true2)
                    # print('loss_false',loss_false)
                    loss = loss_true1+loss_true2+loss_false
                if len(false_logit_list)==0: #全部都不在logit pool：
                    labeled_all_false = labeled_all[false_idx].clone() #### this
                    label_false = target_all[false_idx].clone()
                    model_temp=copy.deepcopy(model)
                    weighted_logits= client_prediction(labeled_all_false,label_false,model_temp)
                    pred_false=y_preds[false_idx].clone().to(device)
#                     print('pred_false',pred_false.shape)
                    pred_false_ = F.log_softmax(pred_false, dim=-1)
                    loss_false = F.kl_div(pred_false_, weighted_logits.to(device), reduction='mean')
                    # print('loss_true1',loss_true1)
#                     print('loss_true2',loss_true2)
                    # print('loss_false',loss_false)
#                     loss = loss_true1+loss_false
                    loss = loss_true1+loss_false
                if len(false_logit_list)==length_false_idx: #全部都在logit pool：
                    false_logit=torch.tensor(false_logit_list) ### logit pool中的logit
                    false_pred=torch.tensor(fale_pred_list)
                    false_label=torch.tensor(false_label_list)
                    loss_false_ce = criterion_1(false_pred, false_label) ####这里是空的
                    false_pred_ = F.log_softmax(false_pred, dim=-1)
                    false_logit_= F.softmax(false_logit, dim=-1)
                    loss_kl = F.kl_div(false_pred_, false_logit_, reduction='mean')
                    loss_true2=0.5*loss_false_ce+0.5*loss_kl
                    # print('loss_true1',loss_true1)
                    # print('loss_true2',loss_true2)
#                     print('loss_false',loss_false)
                    loss = loss_true1+loss_true2

            if len(true_idx)==bsz: ##全部预测正确
#                 print('len(true_idx)==bsz')
                for i in true_idx:
                    false_idx.remove(i)
                    cor_idx_list.append(bsz*ind+i) ### 需要check ind对不对
                    logits_list.append(y_preds[i])
                true_pred=y_preds[true_idx].clone()
                true_label =target_all[true_idx].clone()
                loss_true1 =criterion_1(true_pred, true_label) ### 预测对的数据
                if comm_round==1:
                    logits_pool = dict(zip(cor_idx_list,logits_list))
#                     print('logits_pool',logits_pool.keys())
                    np.save('global_model/att_global_test_'+'logit_dict.npy', logits_pool) 
                else:
                    logits_pool_pre = np.load('global_model/att_global_test_'+'logit_dict.npy',allow_pickle=True).item()
                    logits_pool = dict(zip(cor_idx_list,logits_list))
#                     print('logits_pool',logits_pool.keys())
                    logits_pool = merge(logits_pool_pre, logits_pool) 
#                     print('_merged_logits_pool',logits_pool.keys())
                    np.save('global_model/att_global_test_'+'logit_dict.npy', logits_pool)
                loss = loss_true1
            if len(true_idx)==0: ###全部预测错误
                # print('len(true_idx)==0')
                if comm_round!=1:
                    logits_pool = np.load('global_model/att_global_test_'+'logit_dict.npy',allow_pickle=True).item()
                    false_logit_list=[]
                    fale_pred_list=[]
                    false_label_list=[]
                    for k in false_idx: ### 预测错的数据
                        tem=bsz*ind+k
                        if tem in logits_pool.keys(): ### 存在于logit pool中
                            false_logit_list.append(np.array(logits_pool[tem].cpu().detach())) ### 需要check ind对不对
                            fale_pred_list.append(np.array(y_preds[k].cpu().detach()))
                            false_label_list.append(target_all[k].item())
                            false_idx.remove(k) ### 不存在于logit pool且预测错的
                    false_logit=torch.tensor(false_logit_list) ### logit pool中的logit
                    false_pred=torch.tensor(fale_pred_list)
                    false_label=torch.tensor(false_label_list)
                    loss_false_ce = criterion_1(false_pred, false_label)
                    false_pred_ = F.log_softmax(false_pred, dim=-1)
                    false_logit_= F.softmax(false_logit, dim=-1)
                    loss_kl = F.kl_div(false_pred_, false_logit_, reduction='mean')
                    loss_true2=0.5*loss_false_ce+0.5*loss_kl
                    if len(false_idx)!=0:
                        labeled_all_false = labeled_all[false_idx].clone() #### this
                        label_false = target_all[false_idx].clone()
                        model_temp=copy.deepcopy(model)
                        weighted_logits= client_prediction(labeled_all_false,label_false,model_temp)
                        pred_false=y_preds[false_idx].clone()
                        pred_false_ = F.log_softmax(pred_false, dim=-1)
                        loss_false = F.kl_div(pred_false_, weighted_logits, reduction='mean')
                    loss = loss_true2+loss_false
                if comm_round ==1:
                    continue
            optim.zero_grad()
            loss.backward()
            optim.step()
            del loss
            torch.cuda.empty_cache()
    return model

def train_kl_att(data,label,model,feat_s, pred_false, criterion_ce,criterion_kd,criterion_kl,device):
    import torch.nn as nn
    import torch.nn.functional as F
    loss=0
    num=num_clients
    for i in range(num_clients): ### 全部data
        model.load_state_dict(torch.load('local_model/'+'att_local_test_'+str(i)+'.pt'))
        model.to(device)
        # print('feat_s',feat_s[0].shape)
        with torch.no_grad():
            feat_t, y_pred,_,__ = model(data, is_feat=True)
            feat_t = [f.detach() for f in feat_t]
        # print('feat_t_done')
        # print('feat_t',feat_t[0].shape)
        y_pred = nn.Softmax()(y_pred)
        y_preds = y_pred.max(1)[1]
        incor_idx = torch.where(y_preds != label)[0].tolist()
        cor_idx = torch.where(y_preds != label)[0].tolist()
        loss_ce = criterion_ce(pred_false, label)
        # print('generate_loss_ce_done')
        # print('cor_idx',cor_idx)
        if len(cor_idx)==0 or len(cor_idx)==1:
            loss=loss_ce
            num-=1
        else:
            ent_1 = torch.sum(-y_pred * torch.log(y_pred + 1e-5), dim=1) / np.log(2)
            ent_1 = ent_1.float().cpu()
            ent_1[incor_idx]=0
            ent_1.to(device)
            feat_ss=[]
            for i in feat_s:
                feat_ss.append(i[cor_idx].to(device))
            feat_tt=[]
            for i in feat_t:
                feat_tt.append(i[cor_idx].to(device))
            output_t = y_pred[cor_idx]
            output_s = pred_false[cor_idx]
            # feat_s=feat_s[cor_idx]
            # feat_t=feat_t[cor_idx]
            loss_kd = criterion_kd(feat_ss, feat_tt,ent_1)
            # print('loss_kd_done')
            loss_kl = criterion_kl(output_s, output_t)
            # print('loss_kl_done')
            loss += loss_kd+5*loss_kl+5*loss_ce
            # print('generate_kt_loss_done')
        # print('kd_att_loss',loss)
    if num==0:
        return loss
    else:
        return loss/num

class DistillKL_(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL_, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        import torch.nn as nn
        import torch.nn.functional as F
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss

def gems_att_training(comm_round,model,local_model,optim,device,lr_schedule,global_loader, global_epoch,criterion_1,criterion_2,criterion_kd,criterion_kl, test_loader):
    import torch.nn as nn
    import torch.nn.functional as F
    AUC_value = 0
    model.to(device)
    for epoch in range(1, global_epoch+1):
        # print('epoch',epoch)
        cor_idx_list=[]
        logits_list=[]
        #     batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))
        # batch_iterator = zip(loop_iterable(labeled_loader) ,loop_iterable(unlabeled_loader))
        for ind, (data,label) in enumerate(global_loader):
            labeled_all,target_all =  data.to(device), label.to(device)
            bsz = labeled_all.shape[0]
            # print('labeled_all',labeled_all.shape)
            # print('target_all',target_all)
            labeled_all = labeled_all.to(torch.float32)
            # aug_data = aug_data.to(torch.float32)
            # images = torch.cat([labeled_all, aug_data], dim=0)
            
            target_all = target_all.to(torch.long)
            # centroid_label = torch.tensor([0,1])
            feat_s, y_preds,_,__ = model(labeled_all)
            # y_preds=y_preds[:bsz]
            y_pred = y_preds.max(1)[1]
            true_idx = torch.where(torch.eq(y_pred,target_all)==True)[0].tolist()
            # f1, f2 = torch.split(features_norm, [bsz, bsz], dim=0)
            # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            # del _,f1, f2,features_norm
            # torch.cuda.empty_cache()
            # clustering_loss = criterion_2(features, target_all)
#             print('true_idx',len(true_idx))
            false_idx = [i for i in range(bsz)]
            # centroid,_ = obtain_cnn_centroid_feature(model,labeled_all)
            # centroid = torch.tensor(centroid).to(device)
            # weighted_centroid_tensor = weighted_centroid_tensor.to(device)
            # print('centroid',centroid.shape)
            # print('weighted_centroid_tensor',weighted_centroid_tensor.shape)
            # centroid_features_1 = torch.cat([centroid.unsqueeze(1),weighted_centroid_tensor.unsqueeze(1)], dim=1)
            # centroid_contrastive_1= criterion_2(centroid_features_1,centroid_label)
            # centroid_contrastive = centroid_contrastive_1
            # print('centroid_features_1',centroid_features_1.shape)
            # print('weighted_centroid_tensor',weighted_centroid_tensor.shape)


#             print('len',len(true_idx))
            if len(true_idx)!=0 and len(true_idx)!=bsz:
#                 print('len(true_idx)!=0 and len(true_idx)!=bsz')
                for i in true_idx:
                    false_idx.remove(i)
                    cor_idx_list.append(bsz*ind+i) ### 需要check ind对不对
                    logits_list.append(y_preds[i])
#                 print('cor_idx_list',cor_idx_list)
                true_pred=y_preds[true_idx].clone()
                true_label =target_all[true_idx].clone()
#                 print('true_pred',true_pred.shape)
                loss_true1 =criterion_1(true_pred, true_label) ### 预测对的数据
                if comm_round==1:
                    logits_pool = dict(zip(cor_idx_list,logits_list))
                    # print('generate_dict_done')
#                     print('logits_pool',logits_pool.keys())
                    np.save('global_model/att_global_test_'+'logit_dict.npy', logits_pool) 
                else:
                    logits_pool_pre = np.load('global_model/att_global_test_'+'logit_dict.npy',allow_pickle=True).item()
                    logits_pool = dict(zip(cor_idx_list,logits_list))
#                     print('logits_pool',logits_pool.keys())
                    logits_pool = merge(logits_pool_pre, logits_pool) 
#                     print('merged_logits_pool',logits_pool.keys())
                    np.save('global_model/att_global_test_'+'logit_dict.npy', logits_pool) 
                false_logit_list=[]
                fale_pred_list=[]
                false_label_list=[]
                length_false_idx=len(false_idx)
#                 print('false_idx_be',len(false_idx))
                for k in false_idx: ### 预测错的数据
                    tem=bsz*ind+k
                    if tem in logits_pool.keys(): ### 存在于logit pool中
#                         print('k',k)
                        false_logit_list.append(np.array(logits_pool[tem].cpu().detach())) 
                        fale_pred_list.append(np.array(y_preds[k].cpu().detach()))
                        false_label_list.append(target_all[k].item())
                        # false_label_list.append(np.array(target_all[k].item()))
                        false_idx.remove(k) ### 不存在于logit pool且预测错的
#                 print('false_idx_af',len(false_idx))
#                 print('false_logit_list',len(false_logit_list))
                if len(false_logit_list)!=0 and len(false_logit_list)!=length_false_idx: # 部分存在于logit pool
                    false_logit=torch.tensor(false_logit_list).to(device) ### logit pool中的logit
                    false_pred=torch.tensor(fale_pred_list).to(device)
                    # print('false_label_list',false_label_list)
                    false_label=torch.tensor(false_label_list).to(device)
#                     print('false_label',false_label)
#                     print('false_pred',false_pred)
                    loss_false_ce = criterion_1(false_pred, false_label) ####这里是空的
                    false_pred_ = F.log_softmax(false_pred, dim=-1)
                    false_logit_= F.softmax(false_logit, dim=-1)
                    loss_klx = F.kl_div(false_pred_, false_logit_, reduction='mean')
                    loss_true2=0.5*loss_false_ce+0.5*loss_klx
                    # print('loss_true2_done')


                    labeled_all_false = labeled_all[false_idx].clone() #### this
                    label_false = target_all[false_idx].clone()
                    
                    pred_false=y_preds[false_idx].to(device) #### 错误预测的值
                    model_temp=copy.deepcopy(local_model)

                    feat_ss=[]
                    for i in feat_s:
                        feat_ss.append(i[false_idx].to(device))
                    # print('generate_feta_ss_done')
                    loss_false = train_kl_att(labeled_all_false,label_false,model_temp,feat_ss,pred_false, criterion_1,criterion_kd,criterion_kl,device)
                    # print('generate_loss_false_done')
                    # feat_s = feat_s[false_idx]
                    # loss_false = train_kl_att(labeled_all_false,label_false,model_temp,feat_s,criterion_kd)


                    # weighted_logits= client_prediction(labeled_all_false,label_false,model_temp)
                    # pred_false=y_preds[false_idx].to(device)
                    # pred_false_ = F.log_softmax(pred_false, dim=-1)
                    # loss_false = F.kl_div(pred_false_, weighted_logits.to(device), reduction='mean')
                    del model_temp,label_false,labeled_all_false,false_logit_,false_pred_,false_pred,false_label
                    torch.cuda.empty_cache()
                    # print('loss_true1',loss_true1)
                    # print('loss_true2',loss_true2)
                    # print('loss_false',loss_false)
                    loss = 0.5*loss_true1+0.3*loss_true2+0.2*loss_false
                    # loss = clustering_weights*clustering_loss+contrastive_weights*centroid_contrastive+0.5*loss_true1+0.3*loss_true2+0.2*loss_false
                if len(false_logit_list)==0: #全部都不在logit pool：
                    labeled_all_false = labeled_all[false_idx].clone() #### this
                    label_false = target_all[false_idx].clone()
                    model_temp=copy.deepcopy(local_model)
                    # feat_s = feat_s[false_idx]
                    feat_ss=[]
                    for i in feat_s:
                        feat_ss.append(i[false_idx])
                    pred_false=y_preds[false_idx].clone().to(device)
                    loss_false = train_kl_att(labeled_all_false,label_false,model_temp,feat_ss,pred_false, criterion_1, criterion_kd,criterion_kl,device)
                    # loss_false = train_kl_att(labeled_all_false,label_false,model_temp,feat_ss,criterion_kd,device)
#                     weighted_logits= client_prediction(labeled_all_false,label_false,model_temp)
                    # pred_false=y_preds[false_idx].clone().to(device)
# #                     print('pred_false',pred_false.shape)
#                     pred_false_ = F.log_softmax(pred_false, dim=-1)
#                     loss_false = F.kl_div(pred_false_, weighted_logits.to(device), reduction='mean')
                    del label_false,model_temp,labeled_all_false
                    torch.cuda.empty_cache()
                    # print('loss_true1',loss_true1)
#                     print('loss_true2',loss_true2)
                    # print('loss_false',loss_false)
#                     loss = loss_true1+loss_false
                    # loss = clustering_weights*clustering_loss+contrastive_weights*centroid_contrastive+0.7*loss_true1+0.3*loss_false
                    loss = 0.7*loss_true1+0.3*loss_false
                if len(false_logit_list)==length_false_idx: #全部都在logit pool：
                    false_logit=torch.tensor(false_logit_list).to(device) ### logit pool中的logit
                    # print('fale_pred_list',fale_pred_list)
                    false_pred=torch.tensor(fale_pred_list).to(device)
                    false_label=torch.tensor(false_label_list).to(device)
                    loss_false_ce = criterion_1(false_pred, false_label) ####这里是空的
                    false_pred_ = F.log_softmax(false_pred, dim=-1)
                    false_logit_= F.softmax(false_logit, dim=-1)
                    loss_kl = F.kl_div(false_pred_, false_logit_, reduction='mean')
                    loss_true2=0.5*loss_false_ce+0.5*loss_kl
                    del false_pred_,false_logit_,false_label
                    torch.cuda.empty_cache()
                    # print('loss_true1',loss_true1)
                    # print('loss_true2',loss_true2)
#                     print('loss_false',loss_false)
                    loss = 0.6*loss_true1+0.4*loss_true2
                    # loss = clustering_weights*clustering_loss+contrastive_weights*centroid_contrastive+0.6*loss_true1+0.4*loss_true2

            if len(true_idx)==bsz: ##全部预测正确
#                 print('len(true_idx)==bsz')
                for i in true_idx:
                    false_idx.remove(i)
                    cor_idx_list.append(bsz*ind+i) ### 需要check ind对不对
                    logits_list.append(y_preds[i])
                true_pred=y_preds[true_idx].clone()
                true_label =target_all[true_idx].clone()
                loss_true1 =criterion_1(true_pred, true_label) ### 预测对的数据
                if comm_round==1:
                    logits_pool = dict(zip(cor_idx_list,logits_list))
#                     print('logits_pool',logits_pool.keys())
                    np.save('global_model/att_global_test_'+'logit_dict.npy', logits_pool)
                else:
                    logits_pool_pre = np.load('global_model/att_global_test_'+'logit_dict.npy',allow_pickle=True).item()
                    logits_pool = dict(zip(cor_idx_list,logits_list))
#                     print('logits_pool',logits_pool.keys())
                    logits_pool = merge(logits_pool_pre, logits_pool) 
#                     print('_merged_logits_pool',logits_pool.keys())
                    np.save('global_model/att_global_test_'+'logit_dict.npy', logits_pool)
                loss = loss_true1
                # loss = clustering_weights*clustering_loss+contrastive_weights*centroid_contrastive+loss_true1
            if len(true_idx)==0: ###全部预测错误
                # print('len(true_idx)==0')
                if comm_round!=1:
                    logits_pool = np.load('global_model/att_global_test_'+'logit_dict.npy',allow_pickle=True).item()
                    false_logit_list=[]
                    fale_pred_list=[]
                    false_label_list=[]
                    for k in false_idx: ### 预测错的数据
                        tem=bsz*ind+k
                        if tem in logits_pool.keys(): ### 存在于logit pool中
                            false_logit_list.append(np.array(logits_pool[tem].cpu().detach())) ### 需要check ind对不对
                            fale_pred_list.append(np.array(y_preds[k].cpu().detach()))
                            false_label_list.append(target_all[k].item())
                            false_idx.remove(k) ### 不存在于logit pool且预测错的
                    false_logit=torch.tensor(false_logit_list).to(device) ### logit pool中的logit
                    false_pred=torch.tensor(fale_pred_list).to(device)
                    false_label=torch.tensor(false_label_list).to(device)
                    loss_false_ce = criterion_1(false_pred, false_label)
                    false_pred_ = F.log_softmax(false_pred, dim=-1)
                    false_logit_= F.softmax(false_logit, dim=-1)
                    loss_kl = F.kl_div(false_pred_, false_logit_, reduction='mean')
                    loss_true2=0.5*loss_false_ce+0.5*loss_kl
                    if len(false_idx)!=0:
                        labeled_all_false = labeled_all[false_idx].clone() #### this
                        label_false = target_all[false_idx].clone()
                        model_temp=copy.deepcopy(model)
                        weighted_logits= client_prediction(labeled_all_false,label_false,model_temp)
                        pred_false=y_preds[false_idx].clone()
                        pred_false_ = F.log_softmax(pred_false, dim=-1)
                        loss_false = F.kl_div(pred_false_, weighted_logits, reduction='mean')
                    loss = 0.6*loss_true2+0.4*loss_false
                    # loss = clustering_weights*clustering_loss+contrastive_weights*centroid_contrastive+0.6*loss_true2+0.4*loss_false
                if comm_round ==1:
                    print('all_incorrect')
                    continue
            # print('att_loss',loss)
            optim.zero_grad()
            loss.backward()
            optim.step()
            del loss
            torch.cuda.empty_cache()
        test_AUC = train_test(model, device, test_loader)
        if test_AUC>= AUC_value:
            AUC_value = test_AUC
            model_ = copy.deepcopy(model)
    return model_
