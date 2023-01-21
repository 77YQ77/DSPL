import numpy as np
import os
from sklearn.model_selection import train_test_split
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
file_path_ASD = '../Data/ASD/rois_cc200_asd(190 feature)'
file_path_ADHD = '../Data/ADHD/rois_cc200_adhd200(190 feature)'


def obtain_site_feature(site_index, feature, label, pheno, ASD_SITE,site_3_4,site_14_15,site_16_17 ):
    site_feature=[]
    site_label=[]
    site_pheno=[]
    for i in site_index:
        if i==3:
            ind = site_3_4
        elif i==14:
            ind = site_14_15
        elif i ==16:
            ind = site_16_17
        else:
            ind=np.where(ASD_SITE==i)[0]
        # print('ind',ind)
        site_feature.append(feature[ind])
        site_label.append(label[ind])
        site_pheno.append(pheno[ind])
    return site_feature,site_label,site_pheno

def get_tensor_data(feature, label,pheno, client_index, site_index,ASD_SITE,site_3_4,site_14_15,site_16_17):
    site_feature, site_label, site_pheno = obtain_site_feature(site_index, feature, label,pheno, ASD_SITE,site_3_4,site_14_15,site_16_17)
    site_feature_array=np.array(site_feature)
    site_label_array=np.array(site_label)
    site_pheno_array=np.array(site_pheno)
    client_1_feature = site_feature_array[client_index]
    client_1_label = site_label_array[client_index]
    client_1_pheno = site_pheno_array[client_index]
    client_feature, client_label, client_pheno=[],[],[]
    for i in range(client_1_feature.shape[0]):
        for j in range(len(client_1_feature[i])):
            client_feature.append(client_1_feature[i][j])
            client_label.append(client_1_label[i][j])
            client_pheno.append(client_1_pheno[i][j])
    feature_tmp = np.array(client_feature)
    label_tmp = np.array(client_label)
    pheno_tmp = np.array(client_pheno)
    return feature_tmp,label_tmp,pheno_tmp

def obtain_ASD_data(ASD_DATA,ASD_LABEL,ASD_PHENO, ASD_SITE):
#     ASD_SITE= np.load('Data/ASD_site_whole.npy')
#     ASD_DATA= np.load('Data/ASD_data_whole.npy',allow_pickle=True)
#     ASD_LABEL= np.load('Data/ASD_label_whole.npy',allow_pickle=True)
    feature = ASD_DATA
    site_index = np.unique(ASD_SITE).tolist()
    a=np.where(ASD_SITE==3)[0]
    b=np.where(ASD_SITE==4)[0]
    c=np.where(ASD_SITE==14)[0]
    d=np.where(ASD_SITE==15)[0]
    e=np.where(ASD_SITE==16)[0]
    f=np.where(ASD_SITE==17)[0]
    site_3_4= np.concatenate([a,b],axis=0)
    site_14_15= np.concatenate([c,d],axis=0)
    site_16_17= np.concatenate([e,f],axis=0)
    site_index.remove(4)
    site_index.remove(15)
    site_index.remove(17)
    client_1_index =np.array([-3,-4,0,6])
    client_2_index = np.array([-2,-5,4,-6,-7,7])
    client_3_index = np.array([3,8,-1,2,9,1])
    target_index = np.array([5])
    tar_index = np.concatenate([client_1_index, target_index],axis=0)
    sor_index = np.concatenate([client_2_index,client_3_index],axis=0)
    client_index=[tar_index, sor_index]
    sequence_1= client_index[0] ##309
    sequence_2= client_index[1] ##274
    feature_1,label_1,pheno_1 = get_tensor_data(feature, ASD_LABEL,ASD_PHENO,sequence_1, site_index,ASD_SITE,site_3_4,site_14_15,site_16_17)
    feature_2,label_2,pheno_2 = get_tensor_data(feature, ASD_LABEL,ASD_PHENO,sequence_2, site_index,ASD_SITE,site_3_4,site_14_15,site_16_17)
    print('ASD',label_1.shape,label_2.shape)
    return [feature_1,feature_2],[label_1,label_2],[pheno_1,pheno_2]

if __name__ =='__main__':
    site = np.load('../data/augmented_site.npy')
    data = np.load('../data/augment_data.npy', allow_pickle=True)
    label = np.load('../data/augment_label.npy', allow_pickle=True)
    print(len(np.where(label==0)[0]))
    print(len(np.where(label == 1)[0]))
    pheno = np.load('../data/augmented_pheno.npy',allow_pickle=True)
    a,b,c = obtain_ASD_data(data, label, pheno, site)
    source_data = np.array(a[0])
    target_data = np.array(a[1])
    source_label = np.array(b[0])
    target_label = np.array(b[1])
    source_pheno = np.array(c[0])
    target_pheno = np.array(c[1])
    print('source_data',source_data.shape)
    print('target_data', target_data.shape)
    print('source_label', source_label.shape)
    print('target_label', target_label.shape)
    print('source_pheno', source_pheno.shape)
    print('target_pheno', target_pheno.shape)
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(source_data, source_label, random_state=777, test_size=0.75)

    print(X_train_s.shape)
    print(X_test_s.shape)
    print(y_train_s.shape)
    print(y_test_s.shape)
    # source_full_data = np.concatenate([source_data,source_pheno],axis=1)
    # target_full_data = np.concatenate([target_data, target_pheno], axis=1)
    # print(source_full_data.shape)
