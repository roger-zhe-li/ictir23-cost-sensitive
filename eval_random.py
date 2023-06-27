# -*- coding: utf-8 -*-
import numpy as np
import torch
import torchsnooper
import torch.nn as nn
# from sklearn.metrics import average_precision_score as ap

def hit(gt):
    for gt_item in gt:
        if gt_item == 1:
            return 1
    return 0

def dcg(gt, k):
    return np.sum((np.power(2, gt[: k]) - 1) / np.log2(np.arange(2, k + 2)))

def ap(gt, k):
    if np.sum(gt[: k]) != 0:
        return 1.0 / np.sum(gt[: k]) * (gt[: k] * np.cumsum(gt[: k]) / (1 + np.arange(k))).sum()
    return 0

# @torchsnooper.snoop()
def evaluation(model, test_loader, device, num_test, user_factors, item_factors):
    model.eval()
    HR, NDCG, NDCG_k, AP_k, AP= [], [], [], [], []
    Loss = []
    
    for users, items, rels in test_loader:
        users_ = torch.LongTensor(users).to(device)
        items_ = torch.LongTensor(items).to(device)
        user_review_vec = user_factors(users_)
        item_review_vec = item_factors(items_)

        pred = model(user_review_vec, item_review_vec)
        rels_ = rels

        with torch.no_grad():
            prediction_i = model(user_review_vec, item_review_vec)

            ratings, indices = torch.topk(prediction_i.t(), num_test)
            loss = (nn.Tanh()(nn.BCEWithLogitsLoss(reduction='none')(prediction_i, rels_))).mean()
            

        Loss.append(loss.cpu().detach())
      
        recommends = torch.take(items_, indices).cpu().numpy().tolist()
        binary_gt = np.squeeze(rels[indices].cpu().numpy())        

        # with cutoff
        idcg_gt = np.sort(rels.cpu().numpy())[::-1]

        NDCG.append(dcg(binary_gt, num_test) / dcg(idcg_gt, num_test))
       
        AP.append(ap(binary_gt, num_test))

    NDCG_avg = np.mean(NDCG)
    AP_avg = np.mean(AP)
    print("NDCG = %.4f" % NDCG_avg)
   
    print("AP = %.4f" % AP_avg)

    return np.mean(Loss), NDCG, AP, NDCG_avg, AP_avg
    


