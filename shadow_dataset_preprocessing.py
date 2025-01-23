# import scipy.io as scio
# import networkx as nx
import random
import numpy as np
import torch
import pickle
from collections import defaultdict
import sys
import copy
from arg_parser import *
from utils import seed_all

args = parse_args()
# SHADOW_SOCIAL = 'user'
SHADOW_SOCIAL = 'rand'

# load init datasets
data_dir = args.data_dir
data_file = open(data_dir, 'rb')
_, _, ori_social, _, user_collab, avg_interaction, avg_friend, \
                                                    num_of_target_users, num_of_all_users, num_of_nodes = pickle.load(data_file)
num_item = num_of_nodes-num_of_all_users

# load rec results
# item idx + n_users + 1
data_dir = f'{args.shadow_prefix}/' + args.rec_result
data_file = open(data_dir, 'rb')
history_u_lists = pickle.load(data_file)

for k in history_u_lists.keys():
    # history_u_lists[k] = history_u_lists[k][:1] + [random.randint(0, num_item-1) for _ in range(29)] # 模拟只用top10的推荐结果的攻击效果
    # history_u_lists[k] = [random.randint(0, num_item-1) for _ in range(30)] # 模拟只用top10的推荐结果的攻击效果
    history_u_lists[k] = [x+num_of_all_users+1 for x in history_u_lists[k]]

if SHADOW_SOCIAL == 'user':
    # build shadow social graphs
    u_list = list(ori_social.keys())
    seed_all(42)
    rand_u_list = random.sample(u_list, int(len(u_list)*args.shadow_ratio)) # 固定随机数种子，生成相同的公开关注列表，后续会生成相同的test set
    print('selected shadow users number', len(rand_u_list))

    shadow_edge = 0
    shadow_social = defaultdict(list)
    for uid in rand_u_list:
        shadow_social[uid] = copy.copy(ori_social[uid])
        shadow_edge += len(ori_social[uid])
        for nei in shadow_social[uid]:
            if uid not in shadow_social[nei]:
                shadow_social[nei].append(uid)
                shadow_edge += 1

elif SHADOW_SOCIAL == 'rand':
    sample_ratio = args.shadow_ratio
    seed_all(42)
    
    all_u_pair = list()
    for uid in ori_social.keys():
        for nei in ori_social[uid]:
            all_u_pair.append((uid, nei))
    
    shadow_edge = int(len(all_u_pair)*sample_ratio)
    shadow_social_pair = random.sample(all_u_pair, shadow_edge)
    shadow_social = defaultdict(list)
    for u_pair in shadow_social_pair:
        u1, u2 = u_pair
        shadow_social[u1].append(u2)
    rand_u_list = None
                
print('social pairs: ', shadow_edge)
all_social = 0
for u in ori_social.keys():
    all_social += len(ori_social[u])
print('all social pairs: ', all_social)

valid_user = 0
avg_interaction = 0
for user in history_u_lists.keys():
    valid_user += 1
    avg_interaction += len(history_u_lists[user])
avg_interaction = avg_interaction / valid_user
print("avg_num_of_interaction:", avg_interaction)

valid_user = 0
avg_friend = 0
for user in shadow_social.keys():
    valid_user += 1
    avg_friend += len(shadow_social[user])
avg_friend = avg_friend / valid_user
print("avg_num_of_friend:", avg_friend)

pickle_data = [history_u_lists, 1, shadow_social, 1, rand_u_list, int(avg_interaction), int(avg_friend), num_of_target_users,
            num_of_all_users, num_of_nodes]
with open(f"{args.shadow_prefix}/{args.shadow_dataset}", 'wb') as fo:
    pickle.dump(pickle_data, fo)
print('shadow dataset saved in: ', f"{args.shadow_prefix}/{args.shadow_dataset}")
