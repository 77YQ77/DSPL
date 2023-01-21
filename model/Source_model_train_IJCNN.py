import torch

def ini_target(args, model, device, federated_train_loader, optimizer,criteria): ## train target model
    model.train()
    for z in range(args.epoch):
        for batch_idx, (data, data_1, target, pheno, __) in enumerate(federated_train_loader):
            if len(target.shape) == 1:
                target=target.unsqueeze(1)
                data = data.type(torch.FloatTensor)
                target = target.type(torch.FloatTensor)

            data, target, pheno = data.to(device), target.to(device),pheno.to(device)

            output, fea = model(data, pheno)
            del fea
            loss = criteria(output, target)
            loss.backward()
            optimizer.step()
    return model

