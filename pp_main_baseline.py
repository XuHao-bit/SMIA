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
import os
import copy
import math
 

def ldp_add_noise(model, ep, named_scale, batch_len, device):
    delta = 1e-5 #    
    noise_multiplier = math.sqrt(2 * math.log(1.25 / delta)) / (ep * batch_len)
    for name, param in model.named_parameters():
        if param.grad != None:
            clip_threshold = named_scale[name]
            # print(clip_threshold)
            grad = param.grad.data

            grad_norm = grad.norm(2)
            if grad_norm > clip_threshold:
                grad.mul_(clip_threshold / grad_norm)
            # print(param.grad.data)

            noise_scale = clip_threshold * noise_multiplier
            # print(noise_multiplier, clip_threshold)
            noise = torch.distributions.Normal(0, noise_scale).sample(grad.size()).to(device)  

            # print(noise)
            mask = param.grad != 0
            param.grad.data[mask] += noise[mask]

def print_grad_scale(model, name_grad):
    this_grad = {}
    for name, param in model.named_parameters():
        if param.grad != None:
            grad_norm = param.grad.data.norm(2).item()
            this_grad[name] = grad_norm
            
            if name not in name_grad:
                name_grad[name] = this_grad[name]
            else:
                name_grad[name] += this_grad[name]
    return name_grad, this_grad

def train(args, net, optimizer, trainloader, epoch, device, last_uembs):
    net.train()
    # print(net.RRT)
    # print(net.RRT.todense)
    if args.model_name == 'DESIGN':
        newRRT = net.edge_dropout(net.RRT)
        newRRT = net.sparse_mx_to_torch_sparse_tensor(newRRT)
        net.RRTdrop = newRRT.to_dense()
    # print(net.RRTdrop)
    # print(torch.isnan(net.RRTdrop.to_dense()).any())
    # assert 0
    
    train_loss, train_pploss = 0, 0
    train_num = 0
    named_grad = {}
    for batch_idx, (user, pos, neg) in enumerate(tqdm(trainloader, file=sys.stdout)):
        user = user.to(device)  # [B]
        pos_item = pos.to(device)  # [B]
        neg = neg.squeeze(1)
        neg_item = neg.to(device)  # [B, neg]
        # print(user, pos_item, neg, neg_item)
        epoch = torch.LongTensor([epoch])
        epoch = epoch.to(device)
        l = net.calculate_loss(user, pos_item, neg_item, epoch)
        batch_loss = l
        if args.pp_mtd == 'er':
            pp_l = net.pp_er_loss(user, last_uembs)
            batch_loss += args.pri_coef * pp_l
            if epoch >= 1:
                train_pploss += pp_l.item()

        # print(l)
        optimizer.zero_grad()
        batch_loss.backward()
        batch_len = user.shape[0] #
        if args.pp_mtd == 'dp':
            named_grad, this_grad = print_grad_scale(net, named_grad)
            ldp_add_noise(net, args.dp_ep, this_grad, batch_len, device)
        optimizer.step()
        train_loss += l.item()
        train_num += user.shape[0]
        
    return train_loss / train_num, train_pploss / train_num

def validate(net, config, valid_loader, epoch, device):
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
    with torch.no_grad():
        for _ , (user, pos, negs) in enumerate(tqdm(valid_loader, file=sys.stdout)):
            user = user.to(device)  # [B]
            pos_item = pos.to(device)  # [B]
            neg_items = negs.to(device)  # [B, 999]
            epoch = torch.LongTensor([epoch])
            epoch = epoch.to(device)
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

if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    ## start logging.
    output_dir =  f"./log/{args.dataset}"
    initLogging(output_dir)
    
    # f = open(log_dir, 'a')
    logging.info(str(device))
    set_seed(42) #  
    
    # dataset
    data_file = open(args.data_dir, 'rb')
    #   
    # uid: 1--num_of_target_users, num_of_target_users+1--num_of_all_users
    # itemid: num_of_all_users+1--num_of_nodes
    history_u_lists, _, social_adj_lists, _, _, avg_interaction, avg_friend, \
                                                        num_of_target_users, num_of_all_users, num_of_nodes = pickle.load(data_file)
    
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
    train_data, valid_data, test_data = leave_one_out_split(history_u_lists) #    
    
    # dataloader
    train_loader = get_train_loader(config=config, train_data=train_data, args=args)
    valid_loader = get_valid_loader(config=config, valid_data=valid_data, args=args) #     ；
    # load adj mat
    _, _, uu_social_adj_mat, A_tr, _ = get_adj_mat(config, args, valid_data, test_data)
    spRRT_tr, spRRT_val = get_spRRT(config, args, valid_data, test_data)
    #     
    config['RRT_tr'] = spRRT_tr # sp
    config['S'] = uu_social_adj_mat # np
    config['A_tr'] = A_tr # sp
    
    # decay = [1e-7, 1e-4, 1e-3]
    # kl_reg = [0, 1]
    
    # decay = [1e-3]
    # kl_reg = [0]
    # pri_coefs = [0.1, 0.3, 0.5, 1., 1.5]
    pri_ratios = [1] #     

    # args.pp_mtd = 'dp'
    dp_eps = [2.]
    args.pri_ratio = pri_ratios[0]

    args.pp_mtd = 'er'
    pri_coefs = [3, 4, 5, 7]
    
    for pri_coef in pri_coefs:
        for dp_ep in dp_eps:    
            args.pri_coef = pri_coef
            # args.pri_ratio = pri_ratio
            args.dp_ep = dp_ep
            # args.decay = i
            # args.kl_reg = j
            # early stopping parameter
            test_all_step = 1  # test_all_step=x:    
            best_valid_score = -100000 # best_valid_score    
            bigger = True
            conti_step = 0  #     
            stopping_step = 10  #     
            
            t = get_local_time()
            # dir = f'runs/decay{args.decay}_kl_reg{args.kl_reg}_time{t}'
            # writer = tb.SummaryWriter(log_dir=dir)
            #     
            if args.is_shadow:
                net = Shadow_Model(config=config, args=args, device=device)
            else:
                net = create_model(config=config, args=args, device=device)
            net = net.to(device)    
            # Learning Algorithm
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.learning_rate)

            #     
            # output_dir =  f"./log/{args.dataset}/{t}" #         
            # mkdir_ifnotexist(output_dir)
            # mkdir_ifnotexist('./saved')
            print(get_parameter_number(net))
            
            bestht = [0.0] * 6
            last_uembs = None
            #     
            for epoch in range(args.num_epoch):
                this_uembs = net.gen_all_uemb()['user_embs.weight']
                train_loss, pp_loss = train(args, net, optimizer, train_loader, epoch, device, last_uembs)
                logging.info('Trn_Loss={:.5f}, PP_Loss={:.5f}'.format(train_loss, pp_loss))
                last_uembs = copy.deepcopy(this_uembs)

                if args.is_shadow and epoch % 3 == 0:
                    val_loss, HT5, HT10, HT15, NDCG5, NDCG10, NDCG15 = validate(
                        net, config, valid_loader, epoch, device)
                    logging.info('Val_Loss={:.5f}'.format(val_loss))
                elif not args.is_shadow:
                    val_loss, HT5, HT10, HT15, NDCG5, NDCG10, NDCG15 = validate(
                        net, config, valid_loader, epoch, device)
                    logging.info('Val_Loss={:.5f}'.format(val_loss))

                if HT5+HT10 > bestht[0]+bestht[1]:
                    bestht = [HT5, HT10, HT15, NDCG5, NDCG10, NDCG15]
                # writer.add_scalar("Testing acc:", HT10, epoch+1)
                a = f'[Epoch {epoch+1}]: HT@5:{HT5:.4f} HT@10:{HT10:.4f} NDCG@5:{NDCG5:.4f} NDCG@10:{NDCG10:.4f}'
                logging.info(a)

                #     
                save_name = f'{args.model_name}-{t}.pth' if not args.model_save_name else args.model_save_name
                pth_dir = f'./saved/{args.dataset}/{save_name}'
                valid_result = HT5+HT10
                best_valid_score, conti_step, stop_flag, update_flag = early_stopping(valid_result, best_valid_score,
                                                                                    conti_step, stopping_step, bigger)
                if update_flag:
                    torch.save(net.state_dict(), pth_dir)
                    logging.info(f'Current best epoch is {epoch + 1}, Model saved in: {pth_dir}')
                    logging.info('-------')
                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                (epoch + 1 - conti_step * test_all_step)
                    # logging.info(stop_output)
                    logging.info('\n'+stop_output)
                    break
            
            HT5, HT10, HT15, NDCG5, NDCG10, NDCG15 = bestht
            logging.info(f"[FINAL] HT@5:{HT5:.4f} HT@10:{HT10:.4f} NDCG@5:{NDCG5:.4f} NDCG@10:{NDCG10:.4f}")
