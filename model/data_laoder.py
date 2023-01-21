from torch.utils.data import Dataset
import copy
import torch

class unlabelled_client_data(Dataset):
    def  __init__(self,data, data_a,label,pheno, fname, transforms=None):
        self.D_client0 = data
        self.D_client1 = data_a
        self.D_label = torch.tensor(label).type(torch.FloatTensor)
        self.fname=fname
        self.pheno = pheno
        self.transforms = transforms
    def __getitem__(self, index):
        img_0 = self.D_client0[index]
        img_1 = self.D_client1[index]
        label = self.D_label[index]
        file_name = self.fname[index]
        pheno_0 = self.pheno[index]
        # if self.transforms is not None:
        #     data = self.transforms(data)
        return (img_0, img_1, label, pheno_0,file_name)
    def __len__(self):
        return len(self.D_client1)
# In[9]:


class client_data(Dataset):
    def  __init__(self,data_a,label, fname, transforms=None):
        self.D_client2 = data_a
        self.D_label = label
        self.fname = fname
        self.transforms = transforms
    def __getitem__(self, index):
        img_1 = self.D_client2[index]
        img_2 = self.D_client2[index]
        label = self.D_label[index]
        file_name = self.fname[index]
        # if self.transforms is not None:
        #     data = self.transforms(data)
        return (img_1,img_2, label,file_name)
    def __len__(self):
        return len(self.D_client2)