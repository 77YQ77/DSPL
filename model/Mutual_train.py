import torch
from model import config
import torch
from model import supcontra
from model import CE_Smooth
import torch.nn.functional as F
args = config.Arguments()
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
clu_criterion = supcontra.SupConLoss()
criterion_ce = torch.nn.BCELoss()
# criterion_ce = CE_Smooth.CrossEntropyLabelSmooth(2).cuda()
criterion_ce_soft = CE_Smooth.SoftEntropy().cuda()

def Mutual_train(mutual_train_epoch, combined_loader, model_50, mm_optimizer):
    for z in range(mutual_train_epoch):
        model_50.train()
        for batch_idx, (data_1, data_2, target, pheno, _) in enumerate(combined_loader): 
            if len(target.shape) == 1:
                target = target.unsqueeze(1)
                data_1 = data_1.type(torch.FloatTensor)
                data_2 = data_2.type(torch.FloatTensor)
                targets = target.type(torch.FloatTensor)
            bsz = targets.shape[0]
            data = torch.cat([data_1, data_2], dim=0)
            del data_1, data_2
            target = torch.cat([targets, targets], dim=0)
            pheno = torch.cat([pheno, pheno],dim=0)
            data, target, pheno = data.to(device), target.to(device), pheno.to(device)
            targets = targets.to(device)
            mm_optimizer.zero_grad()
            [classifier_,cluster_],_,__ = model_50(data, pheno)  # classifier the same as pred
            f1, f2 = torch.split(cluster_, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            del f1, f2
            loss_clu = clu_criterion(features,targets)
            loss_cls = criterion_ce(classifier_, target)
            loss_ce_soft_1 = criterion_ce_soft(cluster_, classifier_)
            loss_ce_soft_2 = criterion_ce_soft( classifier_,cluster_)
            loss_ce_soft=0.5*(loss_ce_soft_1+loss_ce_soft_2)
            loss = 0.01 * loss_clu + loss_cls + 0.3 * loss_ce_soft
            loss.backward()
            mm_optimizer.step()
    return model_50
