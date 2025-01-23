import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import numpy as np
import os
from utils import get_local_time, mkdir_ifnotexist, early_stopping, get_parameter_number, set_seed, random_neq, initLogging
import pickle
import logging
from arg_parser import parse_args
from dataloader import datasetsplit, get_adj_mat, get_train_loader, get_valid_loader, leave_one_out_split, get_RRT, get_spRRT
from models import *
import torch.utils.tensorboard as tb


def train(net, optimizer, trainloader, epoch, device):
    net.train()
    # print(net.RRT)
    # print(net.RRT.todense)
    newRRT = net.edge_dropout(net.RRT)
    newRRT = net.sparse_mx_to_torch_sparse_tensor(newRRT)
    net.RRTdrop = newRRT.to_dense()
    # print(net.RRTdrop)
    # print(torch.isnan(net.RRTdrop.to_dense()).any())
    # assert 0
    
    train_loss = 0
    train_num = 0
    for batch_idx, (user, pos, neg) in enumerate(tqdm(trainloader, file=sys.stdout)):
        user = user.to(device)  # [B]
        pos_item = pos.to(device)  # [B]
        neg = neg.squeeze(1)
        neg_item = neg.to(device)  # [B, neg]
        epoch = torch.LongTensor([epoch])
        epoch = epoch.to(device)
        l = net.calculate_loss(user, pos_item, neg_item, epoch)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_loss += l.item()
        train_num += user.shape[0]
    # writer.add_scalars("BP_4_64_alpha0/Training Loss", {"Social Domain Loss": social_loss, "Item Domain Loss": item_loss}, epoch+1)
    # writer.add_scalar("Training Loss", train_loss, epoch+1)
    # print(f'Training on Epoch {epoch + 1}  [train_loss {float(train_loss):f}]')
    return train_loss / train_num

def validate(net, config, train_data, valid_loader, epoch, device):
    net.eval()
    NDCG_5 = 0.0
    NDCG_10 = 0.0
    NDCG_15 = 0.0
    HT_10 = 0.0
    HT_5 = 0.0
    HT_15 = 0.0
    val_loss = 0
    val_num = 0
    # print(train_data)
    valid_user = config['n_target_users'] # the number of users in validation
    with torch.no_grad():
        for _, (user, pos, negs) in enumerate(tqdm(valid_loader, file=sys.stdout)):
            # print(va_user, va_pos)
            # for vu, vi in zip(va_user, va_pos):
            #     print('val user', vu)
            #     print('val item', vi)
            #     print('train item', train_data[int(vu)+1])
            #     print('val item', valid_loader.dataset.val_data[int(vu)+1])
            user = user.to(device)  # [B]
            pos_item = pos.to(device)  # [B]
            neg_items = negs.to(device)  # [B, 999]
            epoch = torch.LongTensor([epoch])
            epoch = epoch.to(device)
            # indices = net.batch_full_sort_predict(user, pos_item, neg_items) # [B, 1000] [B,1]
            # indices = indices.cpu().numpy() 
            HT_5_B, HT_10_B, HT_15_B, NDCG_5_B, NDCG_10_B, NDCG_15_B = net.batch_full_sort_predict(user, pos_item, neg_items, epoch) # compute_rank(indices)
            HT_5 += HT_5_B.item()
            HT_10 += HT_10_B.item()
            HT_15 += HT_15_B.item()
            NDCG_5 += NDCG_5_B.item()
            NDCG_10 += NDCG_10_B.item()
            NDCG_15 += NDCG_15_B.item()
            l = net.calculate_loss(user, pos_item, neg_items[:,0], epoch)
            val_loss += l.item()
            val_num += user.shape[0]
        # print(
        #     f'Validating on epoch {epoch + 1} [HR@5:{float(HT_5 / valid_user):4f} HR@10:{float(HT_10 / valid_user):4f} HR@15:{float(HT_15 / valid_user):4f} NDCG@5:{float(NDCG_5 / valid_user):4f} NDCG@10:{float(NDCG_10 / valid_user):4f} NDCG@15:{float(NDCG_15 / valid_user):4f}]')
        # print('--------')
        return val_loss / val_num, HT_5 / valid_user, HT_10 / valid_user, HT_15 / valid_user, NDCG_5 / valid_user, NDCG_10 / valid_user, NDCG_15 / valid_user

def get_pos_items_per_user(valid_data, train_data, config):
    # train_data, valid_data
    u_ids = []
    i_ids = []
    train_pos_len_list = []
    for u in valid_data:
        train_items = [iid-config['n_users']-1 for iid in train_data[u]]
        i_len = len(train_items)
        train_pos_len_list.append(i_len)
        u_ids.extend([u-1]*i_len)
        i_ids.extend(train_items)
    return torch.tensor([u_ids, i_ids]).type(torch.LongTensor), train_pos_len_list

if __name__ == '__main__':
    args = parse_args()
    # parser = argparse.ArgumentParser() 
    test_mtd = 0
    # print(args)
    # exit()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    ## start logging.
    output_dir =  f"./log/{args.dataset}"
    initLogging(output_dir)
    
    # f = open(log_dir, 'a')
    logging.info(str(device))
    set_seed(42) #  
    
    # dataset
    data_file = open(args.data_dir, 'rb')
    # uid: 1--num_of_target_users, num_of_target_users+1--num_of_all_users
    # itemid: num_of_all_users+1--num_of_nodes
    history_u_lists, _, social_adj_lists, _, _, avg_interaction, avg_friend, \
                                                        num_of_target_users, num_of_all_users, num_of_nodes = pickle.load(data_file)
    
    config = dict()
    config['user_rating'] = history_u_lists
    config['n_users'] = num_of_all_users
    config['n_target_users'] = num_of_target_users
    config['n_items'] = num_of_nodes-num_of_all_users
    config['user_social'] = social_adj_lists
    
    # dataset split
    train_data, valid_data, test_data = leave_one_out_split(history_u_lists) # need to fix the random seed
    
    # dataloader
    train_loader = get_train_loader(config=config, train_data=train_data, args=args)
    valid_loader = get_valid_loader(config=config, valid_data=valid_data, args=args, num_negs=1) # 
    # load adj mat
    uu_collab_adj_mat_tr, uu_collab_adj_mat_val, uu_social_adj_mat, A_tr, A_val = get_adj_mat(config, args, valid_data, test_data)
    spRRT_tr, spRRT_val = get_spRRT(config, args, valid_data, test_data)

    config['RRT_tr'] = spRRT_tr # sp
    config['S'] = uu_social_adj_mat # np
    config['A_tr'] = A_tr # sp
    
    t = get_local_time()
    # load saved model
    net = create_model(config=config, args=args, device=device)
    print(net)
    print(args)
    net = net.to(device)    
    # pth_dir = f'./saved/flickr/DESIGN-Dec-04-23h21m23s.pth'  
    # pth_dir = f'./saved/ciao/DESIGN-Dec-12-19h10m03s.pth' 
    pth_dir = f'./saved/{args.dataset}/{args.trained_name}' 
    new_model = torch.load(pth_dir)
    net.load_state_dict(new_model)

    users_pos_items, train_pos_len_list = get_pos_items_per_user(valid_data, train_data, config)

    net.eval()
    val_loss = 0
    val_num = 0
    # print(train_data)
    HT_15 = 0
    valid_user = config['n_target_users'] # the number of users in validation
    rec_result = {}
    with torch.no_grad():
        positem_pt, vuser_pt = 0, 0 # (1) positem_pt: item point in users_pos_items; (2) vuser_pt: user point in train_pos_len_list
        for bidx, (user, pos, negs) in enumerate(tqdm(valid_loader, file=sys.stdout)):
            this_bs = user.shape[0]
            num_pos_items = sum(train_pos_len_list[vuser_pt: vuser_pt+this_bs]) # get the num of pos items this batch user interacted
            batch_mask = users_pos_items[:, positem_pt: positem_pt+num_pos_items]
            # print(batch_mask)
            batch_mask[0] -= vuser_pt # 
            positem_pt += num_pos_items
            vuser_pt += this_bs

            user = user.to(device)  # [B]
            # batch_scores = net.full_query(user, method=1) # social only
            batch_scores = net.full_query(user, method=test_mtd) # UI only

            # print(batch_scores.shape)
            # print(batch_mask, batch_mask[0].max(), batch_mask[1].max())
            batch_scores[batch_mask[0], batch_mask[1]] = -1e10 #
            _, topk_idx = torch.topk(batch_scores, args.rec_lens, dim=-1)
            # print(topk_idx, topk_idx.shape)
            # print(pos)
            for idx, u in enumerate(user):
                u_number = int(u)+1
                topk_li = [int(ki) for ki in topk_idx[idx]]
                rec_result[u_number] = topk_li
    
        for uu in range(num_of_target_users+1, num_of_all_users+1):
            # print(uu)
            uu = torch.tensor(uu-1)
            uu = uu.to(device)
            batch_scores = net.full_query(uu, method=test_mtd)
            _, topk_idx = torch.topk(batch_scores, args.rec_lens, dim=-1)
            topk_li = [int(ki) for ki in topk_idx]
            rec_result[int(uu)+1] = topk_li


    # print(sorted(rec_result[1]))
    # with open("ciao_design_result_sicial-top30.pkl", 'wb') as fo:
    with open(f"{args.shadow_prefix}/{args.rec_result}", 'wb') as fo:
        pickle.dump(rec_result, fo)
