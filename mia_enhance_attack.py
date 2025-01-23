import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import random
import copy
import pandas as pd
from utils import *
import pickle
from arg_parser import *
from dataloader import *
from attacker import Discriminator
from tqdm import tqdm
from models import MyModel, Shadow_Model

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def normalize_features(features):
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    return (features - mean) / (std + 1e-6)

args = parse_args()
config = vars(args)

seed_all(config['seed'])

path = f'{args.shadow_prefix}/'
# path = f'../raw dataset/{args.dataset}/social_mia/'
config['train_path'] = path + config['shadow_train']
config['test_path'] = path + config['shadow_test']
config['param_path'] = f"./saved/{config['dataset']}/{config['shadow_name']}"
config['rec_result_path'] = f'{args.shadow_prefix}/' + config['rec_result']
dir2 = f'{args.shadow_prefix}/{args.shadow_dataset}_uu_social_adj_mat.npz'
dir3 = f'{args.shadow_prefix}/{args.shadow_dataset}_adj_mat_tr.npz'
config['shadow_dataset'] = f"{args.shadow_prefix}/{args.shadow_dataset}"
# config['rec_len'] = 30
# data_file = open(config['rec_result_path'], 'rb')
# rec_result = pickle.load(data_file)
data_file = open(config['shadow_dataset'], 'rb')
print_config(config)

rec_result, _, _, _, _, _, _, num_of_target_users, num_of_all_users, num_of_nodes = pickle.load(data_file)
for k in rec_result.keys():
    rec_result[k] = [x-num_of_all_users-1 for x in rec_result[k]]

config['user_rating'] = rec_result
config['n_users'] = num_of_all_users
config['n_target_users'] = num_of_target_users
config['n_items'] = num_of_nodes-num_of_all_users

# load adj mat
uu_social_adj_mat = np.load(dir2)     
uu_social_adj_mat = uu_social_adj_mat['all']
A_tr = sp.load_npz(dir3)
config['RRT_tr'] = None # sp
config['S'] = uu_social_adj_mat # np
config['A_tr'] = A_tr # sp

net = Shadow_Model(config=config, args=args, device=args.gpu_id)
net = net.to(args.gpu_id) 
shadow_param = torch.load(config['param_path'])
net.load_state_dict(shadow_param)

# attack_param = net.gen_attack_emb()
# shadow_attack_param = {'user_embs_S.weight': attack_param['user_embs_S.weight'], 'user_embs_A.weight': attack_param['user_embs_A.weight'],
#                 'item_embs.weight': shadow_param['item_embs.weight']}
shadow_attack_param = {'user_embs_S.weight': shadow_param['user1_embs.weight'], 'user_embs_A.weight': shadow_param['user2_embs.weight'],
                'item_embs_S.weight': shadow_param['item1_embs.weight'], 'item_embs_A.weight': shadow_param['item2_embs.weight']}



if config['agg_mtd'] == 'iemb':    
    pri_lrs = [0.03]
else:
    pri_lrs = [0.05]
    
config['pri_epoch'] = 100


for pri_lr in pri_lrs:
    seed_all(config['seed'])
    early_stop = EarlyStopping(30)

    config['pri_lr'] = pri_lr
    print(f'\npri_lr: {pri_lr}')
    # print_config(config)

    train_dataset = atkData(config, 'train', rec_result=rec_result, shadow_param=shadow_attack_param, mtd='my')
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataset = atkData(config, 'test', rec_result=rec_result, shadow_param=shadow_attack_param, mtd='my')
    test_loader = DataLoader(test_dataset, batch_size=4096)
    # exit()
    # model
    model = Discriminator(embed_dim=train_dataset.data_x[0].shape[0])
    model.to(config['gpu_id'])

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['pri_lr'], momentum=0.9, weight_decay=0)

    # train attacker model
    best_record = (0, 0, 0, 0)
    best_model = None
    best_epoch = 0
    record_loss = []
    print('begin train')
    

    for epoch in range(config['pri_epoch']):
        model.train()
        train_loss, train_len = 0, 0
        for batch_x, batch_y in tqdm(train_loader):
            # batch_x = normalize_features(batch_x)

            batch_yhat = model(batch_x)
            batch_loss = criterion(batch_yhat.squeeze(), batch_y)
            
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            train_loss += batch_loss.item()
            train_len += len(batch_x)

        # test
        if (epoch+1) % 2 == 0:
            model.eval()
            test_yhat, test_y = [], []
            test_loss, test_len = 0, 0
            for batch_x, batch_y in tqdm(test_loader):
                yhat, _ = model.predict(batch_x)
                test_yhat.append(yhat)
                test_y.append(batch_y)

                test_loss += criterion(yhat.squeeze(), batch_y).detach().cpu()
                test_len += len(batch_x)
            
            print('[train] loss: {:.4f}, [test] loss: {:.4f}'.format(train_loss*100/train_len, test_loss*100/test_len))
            # print(test_yhat, test_y)
            test_yhat = torch.cat(test_yhat, dim=0).cpu().numpy()
            test_y = torch.cat(test_y, dim=0).long().cpu().numpy()
            record = roc_auc_score(test_y, test_yhat)
            test_yhat_pred = [1 if x > 0.5 else 0 for x in test_yhat]
            recall = recall_score(test_y, test_yhat_pred)
            precision = precision_score(test_y, test_yhat_pred)
            f1 = f1_score(test_y, test_yhat_pred)
            print('epoch: {}, auc: {:.5f}, f1: {:.5f}, recall: {:.5f}, precision: {:.5f}'.format(epoch, record, f1, recall, precision))
            
            if record + f1 > best_record[0] + best_record[1]:
                best_record = (record, f1, recall, precision)
                best_epoch = epoch

            early_stop(val_loss=-(record+f1))
            if early_stop.early_stop:    
                print('Early_stop!')    
                break

    msg = 'best epoch: {}, best auc: {:.4f}, f1: {:.4f}, recall: {:.4f}, precision: {:.4f}'.format(
            best_epoch, best_record[0], best_record[1], best_record[2], best_record[3])
    print(msg)
