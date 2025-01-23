import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import numpy as np
import os
from utils import *
import pickle
import logging
from arg_parser import parse_args
from dataloader import *
from models import * 
import os
import copy
from attacker import Defender, Discriminator
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import pandas as pd
import random
from inference_result import get_pos_items_per_user

 


def pp_train(args, net, defender, train_df, already_u_pair, optimizer, trainloader, epoch, device):
    net.train()
    # 
    pos_df, neg_df = gen_defend_df(train_df, already_u_pair, args.pri_ratio) # 调一调这里的参数；
    pos_data, neg_data = uPairData(pos_df), uPairData(neg_df)
    batch_len = len(trainloader)
    pos_batch = len(pos_data) // batch_len
    pos_dloader = DataLoader(pos_data, batch_size=pos_batch, shuffle=True)
    neg_dloader = DataLoader(neg_data, batch_size=pos_batch, shuffle=True)
    # print(batch_len, pos_batch, len(pos_data), len(pos_dloader))
    
    if args.model_name == 'DESIGN':
        newRRT = net.edge_dropout(net.RRT)
        newRRT = net.sparse_mx_to_torch_sparse_tensor(newRRT)
        net.RRTdrop = newRRT.to_dense()

    train_loss, train_pploss = 0, 0
    train_num = 0
    # 此循环一次运行一个batch的数据  features: torch.Size([B,...]) tqdm是运行batch_idx进度 1，2，3，第几个batch进行过计算了
    # for train_batch, pos_u_batch, neg_u_batch in tqdm(zip(trainloader, pos_dloader, neg_dloader), total=min(len(trainloader), len(pos_dloader))):
    for train_batch, pos_u_batch, neg_u_batch in tqdm(zip(trainloader, pos_dloader, neg_dloader), total=min(len(train_loader), len(pos_dloader))): # zip会在短的dataloader结束后停止
        user, pos, neg = train_batch
        user = user.to(device)  # [B]
        pos_item = pos.to(device)  # [B]
        neg = neg.squeeze(1)
        neg_item = neg.to(device)  # [B, neg]
        epoch = torch.LongTensor([epoch])
        epoch = epoch.to(device)
        l = net.calculate_loss(user, pos_item, neg_item, epoch)
        batch_loss = l

        # if epoch >= args.defend_warm:
        pos_u1, pos_u2 = pos_u_batch
        pos_u1, pos_u2 = pos_u1.to(device), pos_u2.to(device)
        neg_u1, neg_u2 = neg_u_batch
        neg_u1, neg_u2 = neg_u1.to(device), neg_u2.to(device)

        pp_l = net.pp_loss(pos_u1, pos_u2, neg_u1, neg_u2, defender)
        batch_loss += args.pri_coef * pp_l
        train_pploss += pp_l.item()

        # batch_loss = l
        # print(l)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        train_loss += l.item()
        train_num += user.shape[0]
    # writer.add_scalars("BP_4_64_alpha0/Training Loss", {"Social Domain Loss": social_loss, "Item Domain Loss": item_loss}, epoch+1)
    # writer.add_scalar("Training Loss", train_loss, epoch+1)
    # print(f'Training on Epoch {epoch + 1}  [train_loss {float(train_loss):f}]')
    return train_loss / train_num, train_pploss / len(pos_data)

# validate 利用评价指标criterion验证模型效果 (也可以用loss) 应该利用全部的验证集 123
def validate_with_fullrank(net, config, valid_loader, epoch, device, valid_data, train_data):
    net.eval()
    NDCG_5 = 0.0
    NDCG_10 = 0.0
    NDCG_15 = 0.0
    HT_10 = 0.0
    HT_5 = 0.0
    HT_15 = 0.0
    val_loss = 0
    val_num = 0
    valid_user = config['n_target_users'] # the number of users in validation
    
    # save rec result
    rec_result = {}
    users_pos_items, train_pos_len_list = get_pos_items_per_user(valid_data, train_data, config)

    with torch.no_grad():
        positem_pt, vuser_pt = 0, 0 # (1) positem_pt: item point in users_pos_items; (2) vuser_pt: user point in train_pos_len_list
        for _ , (user, pos, negs) in enumerate(tqdm(valid_loader, file=sys.stdout)):
            # calculate 1000 result; 目的是与non-pp train保持对齐
            user = user.to(device)  # [B]
            pos_item = pos.to(device)  # [B]
            neg_items = negs.to(device)  # [B, 999]
            epoch = torch.LongTensor([epoch])
            epoch = epoch.to(device)
            # indices = net.batch_full_sort_predict(user, pos_item, neg_items) # [B, 1000] [B,1]
            # indices = indices.cpu().numpy() 
            # # 计算一个batch的rank情况
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

            # calclulate full rank
            this_bs = user.shape[0]
            num_pos_items = sum(train_pos_len_list[vuser_pt: vuser_pt+this_bs]) # get the num of pos items this batch user interacted
            batch_mask = users_pos_items[:, positem_pt: positem_pt+num_pos_items]
            # print(batch_mask)
            batch_mask[0] -= vuser_pt # 在batch中的相对位置
            positem_pt += num_pos_items
            vuser_pt += this_bs
            batch_scores = net.full_query(user, method=0) # inference from uu+ui graph
            batch_scores[batch_mask[0], batch_mask[1]] = -1e10 # 把batch中的user交互过的pos item给mask掉
            _, topk_idx = torch.topk(batch_scores, 30, dim=-1)
            for idx, u in enumerate(user):
                u_number = int(u)+1
                topk_li = [int(ki) for ki in topk_idx[idx]]
                rec_result[u_number] = topk_li

        for uu in range(num_of_target_users+1, num_of_all_users+1):
            # print(uu)
            uu = torch.tensor(uu-1)
            uu = uu.to(device)
            batch_scores = net.full_query(uu, method=0)
            _, topk_idx = torch.topk(batch_scores, 30, dim=-1)
            topk_li = [int(ki) for ki in topk_idx]
            rec_result[int(uu)+1] = topk_li

        # print(
        #     f'Validating on epoch {epoch + 1} [HR@5:{float(HT_5 / valid_user):4f} HR@10:{float(HT_10 / valid_user):4f} HR@15:{float(HT_15 / valid_user):4f} NDCG@5:{float(NDCG_5 / valid_user):4f} NDCG@10:{float(NDCG_10 / valid_user):4f} NDCG@15:{float(NDCG_15 / valid_user):4f}]')
        # print('--------')
        return val_loss / val_num, HT_5 / valid_user, HT_10 / valid_user, HT_15 / valid_user, NDCG_5 / valid_user, NDCG_10 / valid_user, NDCG_15 / valid_user, rec_result

def init_defend_dataset(social_adj_lists):
    # 每个epoch sample一部分数据当作training defender；
    training_dataset = {
        'user1': [],
        'user2': [],
        'y': []
    }
    already_u_pair = set()
    for u1 in social_adj_lists.keys():
        for u2 in social_adj_lists[u1]:
            already_u_pair.add((u1, u2))
            training_dataset['user1'].append(u1)
            training_dataset['user2'].append(u2)
            training_dataset['y'].append(1)
    train_df = pd.DataFrame(training_dataset)
    return train_df, already_u_pair

def gen_defend_df(train_df, already_u_pair, ratio=0.25):
    # 每次从train_df中sample ratio 的正样本，然后生成同样数量的负样本
    sampled_df = train_df.sample(frac=ratio).reset_index(drop=True)
    neg_dataset = {
        'user1': [],
        'user2': [],
        'y': []
    }

    # num of target users 是参与训练的
    n_neg = 0
    pos_len = len(sampled_df)
    this_already_u_pair = copy.deepcopy(already_u_pair)
    while n_neg < pos_len:
        u1 = random.randint(1, num_of_target_users)
        u2 = random.randint(1, num_of_target_users)
        while u1 == u2:
            u2 = random.randint(1, num_of_target_users)
        if (u1, u2) in this_already_u_pair:
            continue
        this_already_u_pair.add((u1, u2))
        neg_dataset['user1'].append(u1)
        neg_dataset['user2'].append(u2)
        neg_dataset['y'].append(0)
        n_neg += 1
    neg_df = pd.DataFrame(neg_dataset)
    return sampled_df, neg_df

def train_defender(args, defender, train_all_df, already_u_pair, net_param, defdr_optimizer):
    pos_df, neg_df = gen_defend_df(train_all_df, already_u_pair, args.pri_ratio)
    train_df = pd.concat([pos_df, neg_df])

    this_param = copy.deepcopy(net_param)
    train_dataset = defendData(args, train_df, this_param)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    criterion = nn.BCELoss()

    for local_epoch in range(args.pri_epoch):
        defender.train()
        train_loss, train_len = 0, 0
        for batch_x, batch_y in tqdm(train_loader):
            # print(batch_x)
            batch_yhat = defender(batch_x)
            batch_loss = criterion(batch_yhat.squeeze(), batch_y)
            
            defdr_optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(defender.parameters(), 5)
            defdr_optimizer.step()

            train_loss += batch_loss.item()
            train_len += len(batch_x)
    return train_loss / train_len

def mia_attack(args, rec_result, shadow_param):    
    config = vars(args)
    # path = f'../raw dataset/{args.dataset}/social_mia/'
    config['train_path'] = args.ppmain_trn # 这里就是仿真mia attack，所以随便一个范围的shadow dataset即可
    config['test_path'] = args.ppmain_tst
    # config['rec_len'] = 30

    # dataloader
    train_dataset = atkData(config, 'train', rec_result=rec_result, shadow_param=shadow_param, mtd=config['agg_mtd'])
    test_dataset = atkData(config, 'test', rec_result=rec_result, shadow_param=shadow_param, mtd=config['agg_mtd'])
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4096)

    atk_epoch = 25 # 这个只是用来模拟一个攻击，以及判定防御成功
    early_stop = EarlyStopping(10)

    # model
    model = Discriminator(embed_dim=train_dataset.data_x[0].shape[0])
    model.to(config['gpu_id'])

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['atk_lr'], momentum=0.9, weight_decay=0)

    best_record = (0, 0, 0, 0)
    best_model = None
    best_epoch = 0
    record_loss = []
    print('begin train')
    for epoch in tqdm(range(atk_epoch)):
        model.train()
        train_loss, train_len = 0, 0
        for batch_x, batch_y in train_loader:
            batch_yhat = model(batch_x)
            batch_loss = criterion(batch_yhat.squeeze(), batch_y)
            
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            train_loss += batch_loss.item()
            train_len += len(batch_x)

        # test every 2 epochs
        if epoch % 2 == 0:
            model.eval()
            test_yhat, test_y = [], []
            test_loss, test_len = 0, 0
            for batch_x, batch_y in test_loader:
                yhat, _ = model.predict(batch_x)
                test_yhat.append(yhat)
                test_y.append(batch_y)

                test_loss += criterion(yhat.squeeze(), batch_y).detach().cpu()
                test_len += len(batch_x)
            
            # logging('[train] loss: {:.4f}, [test] loss: {:.4f}'.format(train_loss*100/train_len, test_loss*100/test_len))
            # print(test_yhat, test_y)
            test_yhat = torch.cat(test_yhat, dim=0).cpu().numpy()
            test_y = torch.cat(test_y, dim=0).long().cpu().numpy()
            record = roc_auc_score(test_y, test_yhat)
            test_yhat_pred = [1 if x > 0.5 else 0 for x in test_yhat]
            recall = recall_score(test_y, test_yhat_pred)
            precision = precision_score(test_y, test_yhat_pred)
            f1 = f1_score(test_y, test_yhat_pred)

            if record > best_record[0]:
                best_record = (record, f1, recall, precision)
                best_epoch = epoch

            early_stop(val_loss=-record)
            if early_stop.early_stop:    
                print('Early_stop!')    
                break

    msg = 'best epoch: {}, best auc: {:.4f}, f1: {:.4f}, recall: {:.4f}, precision: {:.4f}'.format(
            best_epoch, best_record[0], best_record[1], best_record[2], best_record[3])
    return msg


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    ## start logging.
    output_dir =  f"./log/{args.dataset}"
    initLogging(output_dir)
    
    # f = open(log_dir, 'a')
    logging.info(str(device))
    set_seed(42) # 先声明device再声明seed
    
    # dataset
    data_file = open(args.data_dir, 'rb')
    # 数据集必须处理成这种形式 这样划分训练测试集时才能保证每一个user都有一部分进训练一部分进测试
    # uid: 1--num_of_target_users, num_of_target_users+1--num_of_all_users
    # itemid: num_of_all_users+1--num_of_nodes
    history_u_lists, _, social_adj_lists, _, _, avg_interaction, avg_friend, \
                                                        num_of_target_users, num_of_all_users, num_of_nodes = pickle.load(data_file)
    
        
    # init defend dataset
    train_all_df, already_u_pair = init_defend_dataset(social_adj_lists)

    dataset_stat = f"{args.dataset} statistical information\n"
    dataset_stat += f"num_of_users in u-i and u-u: {num_of_target_users}\n"
    dataset_stat += f"num_of_users only exists in u-u: {num_of_all_users-num_of_target_users}\n"
    dataset_stat += f"num_of_items: {num_of_nodes-num_of_all_users}\n"
    dataset_stat += f"num_of_nodes in the network: {num_of_nodes}\n"
    dataset_stat += f"avg_num_of_interaction: {avg_interaction}\n"
    dataset_stat += f"avg_num_of_friend: {avg_friend}"

    logging.info("----------------------------")
    logging.info(dataset_stat)
    logging.info("----------------------------")  
    
    config = dict()
    config['user_rating'] = history_u_lists
    config['n_users'] = num_of_all_users
    config['n_target_users'] = num_of_target_users
    config['n_items'] = num_of_nodes-num_of_all_users
    config['user_social'] = social_adj_lists
    logging.info('\n'.join([str(k) + ': ' + str(v) for k, v in config.items() if k not in ['user_rating', 'user_social'] ]))
    
    # dataset split
    # train_data, test_data = datasetsplit(history_u_lists, args.split)
    train_data, valid_data, test_data = leave_one_out_split(history_u_lists) # 需要固定随机数种子 否则adj_mat每次都会不一样
    
    # dataloader
    train_loader = get_train_loader(config=config, train_data=train_data, args=args)
    valid_loader = get_valid_loader(config=config, valid_data=valid_data, args=args) # full test的话，改这里；
    # load adj mat
    uu_collab_adj_mat_tr, uu_collab_adj_mat_val, uu_social_adj_mat, A_tr, A_val = get_adj_mat(config, args, valid_data, test_data)
    spRRT_tr, spRRT_val = get_spRRT(config, args, valid_data, test_data)
    # 稀疏矩阵卷积额外写
    config['RRT_tr'] = spRRT_tr # sp
    config['S'] = uu_social_adj_mat # np
    config['A_tr'] = A_tr # sp

    pri_coefs = [0.01]
    pri_ratios = [1] # 每次训练defender和pp loss的比例
    # def_losses = ['cl']
    # pri_epoch 也是一个超参
    # pri_coefs = [0.1, 0.2, 0.3, 0.4, 0.5]

    for pri_coef in pri_coefs:
        for pri_ratio in pri_ratios:    
            args.pri_coef = pri_coef
            args.pri_ratio = pri_ratio
            # early stopping parameter
            test_all_step = 1  # test_all_step=x:每x个epoch在验证集上evaluate一次
            best_valid_score = -100000 # best_valid_score和bigger搭配使用 评价指标越大越好
            bigger = True
            conti_step = 0  # 有连续几次验证效果没有超过best
            stopping_step = 10  # 如果连续有n次的验证效果没有超过best 则将early stop
            
            t = get_local_time()
            # 模型
            net = create_model(config=config, args=args, device=device)
            net = net.to(device)    
            # Learning Algorithm
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.learning_rate)

            # defend model
            defender = Defender(args.hidden*2)
            defender = defender.to(device)
            defdr_optimizer = torch.optim.Adam(defender.parameters(), lr=args.dfd_lr)

            print(get_parameter_number(net))
            logging.info(f'模型总参数：{get_parameter_number(net)}')
            logging.info('-------------------')
            logging.info('超参数如下:')
            logging.info('\n'.join([str(k) + ': ' + str(v) for k, v in vars(args).items()]))
            logging.info('-------------------')
            logging.info('输出记录如下')
            logging.info('-----------------')
            
            bestht = [0.0] * 6
            # 训练 验证
            for epoch in range(args.num_epoch):
                # 在epoch里面，训练defender
                if epoch >= args.defend_warm:
                    train_def_loss = train_defender(args, defender, train_all_df, already_u_pair, net.gen_all_uemb(), defdr_optimizer)
                    logging.info('Defender Trn_Loss={:.5f}'.format(train_def_loss))
                # continue
                train_loss, pp_loss = pp_train(args, net, defender, train_all_df, already_u_pair, optimizer, train_loader, epoch, device)
                logging.info('Trn_Loss={:.5f}, PP_Loss={:.5f}'.format(train_loss, pp_loss))

                val_loss, HT5, HT10, HT15, NDCG5, NDCG10, NDCG15, rec_result = validate_with_fullrank(
                    net, config, valid_loader, epoch, device, valid_data, train_data)
                logging.info('Val_Loss={:.5f}'.format(val_loss))

                if HT5+HT10 > bestht[0]+bestht[1]:
                    bestht = [HT5, HT10, HT15, NDCG5, NDCG10, NDCG15]
                # writer.add_scalar("Testing acc:", HT10, epoch+1)
                a = f'[Epoch {epoch+1}]: HT@5:{HT5:.4f} HT@10:{HT10:.4f} NDCG@5:{NDCG5:.4f} NDCG@10:{NDCG10:.4f}'
                logging.info(a)

                # 早停&模型训练参数保存
                save_name = f'{args.model_name}-{t}.pth' if not args.model_save_name else args.model_save_name
                pth_dir = f'./saved/{args.dataset}/{save_name}'
                valid_result = HT5+HT10
                best_valid_score, conti_step, stop_flag, update_flag = early_stopping(valid_result, best_valid_score,
                                                                                    conti_step, stopping_step, bigger)
                if update_flag:
                    torch.save(net.state_dict(), pth_dir)
                    logging.info(f'Current best epoch is {epoch + 1}, Model saved in: {pth_dir}')
                
                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                (epoch + 1 - conti_step * test_all_step)
                    # logging.info(stop_output)
                    logging.info('\n'+stop_output)
                    break
                
                # 进行随着训练的mia attack；这个是fake mia attack，因为shadow param直接拿rec param模拟的，并没有重新根据rec result来train shadow model；
                # record rec result
                if epoch >= args.defend_warm and epoch % 2 == 0:
                    shadow_param = copy.deepcopy(net.state_dict())
                    mia_result = mia_attack(args, rec_result, shadow_param)
                    logging.info(mia_result)
                logging.info('-------')
                last_rec_result = copy.deepcopy(rec_result)

            HT5, HT10, HT15, NDCG5, NDCG10, NDCG15 = bestht
            logging.info(f"[FINAL] HT@5:{HT5:.4f} HT@10:{HT10:.4f} NDCG@5:{NDCG5:.4f} NDCG@10:{NDCG10:.4f}\n")
