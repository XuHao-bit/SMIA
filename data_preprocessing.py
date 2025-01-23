import scipy.io as scio
import networkx as nx
import random
import numpy as np
import torch
import pickle
from collections import defaultdict
import sys

def build_item_adj_lists(history_v_lists):
    adj_lists = {}
    for key in history_v_lists.keys():
        adj_lists[key] = []
    for key in history_v_lists.keys():
        for key_temp in history_v_lists.keys():
            if key != key_temp and len(set(history_v_lists[key]) | set(history_v_lists[key_temp])) != 0:
                if len(set(history_v_lists[key]) & set(history_v_lists[key_temp])) / len(set(history_v_lists[key]) | set(history_v_lists[key_temp])) > 0.1:
                    adj_lists[key].append(key_temp)
    return adj_lists

def build_user_adj_lists(history_v_lists):
    adj_lists = {}
    for key in history_v_lists.keys():
        adj_lists[key] = []
    for key in history_v_lists.keys():
        for key_temp in history_v_lists.keys():
            if key != key_temp and len(set(history_v_lists[key]) | set(history_v_lists[key_temp])) != 0:
                if len(set(history_v_lists[key]) & set(history_v_lists[key_temp])) / len(set(history_v_lists[key]) | set(history_v_lists[key_temp])) > 0.1:
                    adj_lists[key].append(key_temp)
    return adj_lists

def cold_start_interac_del(rating, n):
    # del users with interaction < n
    user_interaction_dict = defaultdict()
    for row in rating:
        if row[0] not in user_interaction_dict.keys():
            user_interaction_dict[row[0]] = []
        user_interaction_dict[row[0]].append(row[1])

    delete = []
    for i in range(len(rating)):
        user_id = rating[i, 0]
        if len(user_interaction_dict[user_id]) < n:  
            delete.append(i)
    rating = np.delete(rating, delete, 0)
    return rating 

def cold_start_social_del(trust, n):
    user_social_dict = defaultdict()
    for row in trust:
        if row[0] not in user_social_dict.keys():
            user_social_dict[row[0]] = []
        user_social_dict[row[0]].append(row[1])

    delete = []
    for i in range(len(trust)):
        user_id = trust[i, 0]
        if len(user_social_dict[user_id]) < n:
            delete.append(i)
    trust = np.delete(trust, delete, 0)
    return trust 

def bad_interac_del(rating, n):
    delete = []
    for i in range(len(rating)): # rating<=n delete
        if rating[i, 2] <= n:
            delete.append(i)
    rating = np.delete(rating, delete, 0)
    print(f"rating<={n} deletion complete")
    return rating 

def check_if_trust_bi(trust):
    user_social_dict = defaultdict()
    for row in trust:
        if row[0] not in user_social_dict.keys():
            user_social_dict[row[0]] = []
        user_social_dict[row[0]].append(row[1])
    
    for u in user_social_dict.keys():
        for f in user_social_dict[u]:
            if f not in user_social_dict: # 
                return False
            if u not in user_social_dict[f]:
                return False
    return True
            
def make_trust_bidirec(trust):
    # if (u1, u2) in trust, then put (u2,u1) in trust
    trust_pair_set = set()
    for i in range(len(trust)):
        u1_id, u2_id = trust[i,0], trust[i,1]
        trust_pair_set.add((u1_id, u2_id)) # adds a given element to a set if the element is not present in the set 
    reverse_trust_pair_set = set()
    for (id1, id2) in trust_pair_set:
        reverse_trust_pair_set.add((id2, id1))
    all_user_pair_set = trust_pair_set | reverse_trust_pair_set
    l = list(all_user_pair_set)
    return np.array(l)

def load_data_v2():
    # dir_path = os.path.join(conf.data_dir, conf.data_name)
    # social_filename = os.path.join(dir_path, conf.data_name + '.links')
    # rating_filename = os.path.join(dir_path, conf.data_name + '.rating')
    social_filename = 'flickr.links'
    rating_filename = 'flickr.rating'
    user_pair_set = read_social_data_v2(social_filename)
    reverse_user_pair_set = reverse_pair_set(user_pair_set)
    all_user_pair_set = user_pair_set | reverse_user_pair_set
    all_ui_pair_dict = read_rating_data_v2(rating_filename)

    return all_user_pair_set, all_ui_pair_dict

def reverse_pair_set(pair_set):
    # minid, maxid = 99999, 0
    rev_pair_set = set()
    for (id_1, id_2) in pair_set:
        # minid = min(min(id_1, id_2), minid)
        # maxid = max(max(id_1, id_2), maxid)
        rev_pair_set.add((id_2, id_1))
    # print(f'user_id in social network from {minid} to {maxid}')  # 1-7375
    return rev_pair_set

def reverse_pair_list(pair_list):
    rev_pair_list = list()
    for (id_1, id_2) in pair_list:
        rev_pair_list.append((id_2, id_1))
    return rev_pair_list

def read_social_data_v2(filename):
    with open(filename) as f:
        user_pair_set = set()
        for line in f:
            arr = line.split("\t")
            u1_id, u2_id = int(arr[0]), int(arr[1])
            user_pair_set.add((u1_id+1, u2_id+1))
        return user_pair_set

def read_rating_data_v2(filename):
    with open(filename) as f:
        # ui_pair_dict = defaultdict(set)
        ui_pair_dict = set()
        for line in f:
            arr = line.split("\t")
            user_id, item_id = int(arr[0]), int(arr[1])
            # ui_pair_dict[user_id].add(item_id)
            ui_pair_dict.add((user_id+1, item_id+1, 1))
        return ui_pair_dict

trust, rating = load_data_v2()
trust = np.array(list(trust))
# print(trust)
# trust = scio.loadmat("trustnetwork.mat")
# trust = trust['trustnetwork']
# if not check_if_trust_bi(trust):
#     trust = make_trust_bidirec(trust)
print("---------------")
print(f"num_of_trust: {len(trust)}")
print(f'user_id in social network from {np.min(trust)} to {np.max(trust)}')  # 1-7375
print("---------------")

# exit()
rating = np.array(list(rating))
# print(rating[:10])

# exit()
# rating = scio.loadmat("rating.mat")
# rating = rating['rating']
# rating = rating[:, [0, 1, 3]] # user item rating
rating[:, 1] = rating[:, 1] + np.max(trust) # item index increase
print(f"num_of_rating: {len(rating)}")
print(f'user_id in interaction from {np.min(rating[:, 0])} to {np.max(rating[:,0])}') # 0 to 8357
print(f'item_id from {np.min(rating[:,1])} to {np.max(rating[:,1])}')  # 0 to 82119
print(f'rating from {np.min(rating[:,2])} to {np.max(rating[:,2])}')  # 1
print("item index increase(+num_user)")
print(f'item_id from {np.min(rating[:,1])} to {np.max(rating[:,1])}') # 8358 to 90477
print("------------------------")

print("preprocessing: deleting bad interactions and cold start users/items")
# delete cold start users: interaction number <5
rating = cold_start_interac_del(rating, 3) 
print("cold-start deletion complete")
print(f"num_of_rating: {len(rating)}")
# exit()

print("------------------")
print("users and items reindex:")
# build user index: 1---n1, n1+1----n2+n1
dic_node_o2i = {}
i = 1 # index
for user in rating[:, 0]:
    if user in dic_node_o2i.keys():
        continue
    elif user in trust.reshape(-1, ):
        dic_node_o2i[user] = i
        i += 1
num_of_target_users = len(dic_node_o2i)
print("The num of users in both u-i and u-u:", num_of_target_users) 

for user in trust.reshape(-1, ):
    if user in dic_node_o2i.keys():
        continue
    else:
        dic_node_o2i[user] = i
        i += 1
num_of_all_users = len(dic_node_o2i)
print('The number of all users: ', num_of_all_users) 

# delete users only appear in UI graph
delete = []
for j in range(len(rating)):
    if rating[j, 0] not in dic_node_o2i.keys():
        delete.append(j)
rating = np.delete(rating, delete, 0)

# build item index
for item in rating[:, 1]:
    if item in dic_node_o2i.keys():
        continue
    else:
        dic_node_o2i[item] = i
        i += 1
num_of_nodes = len(dic_node_o2i)
print('The number of items ', num_of_nodes-num_of_all_users)
# node_index: 1--num_of_u num_of_u+1--num_of_friend num_of_friend+1--num_of_nodes

history_u_lists = defaultdict()
history_v_lists = defaultdict()
social_adj_lists = defaultdict()

print('building social_adj_lists')
for line in trust:
    if dic_node_o2i[line[0]] not in social_adj_lists.keys():
        social_adj_lists[dic_node_o2i[line[0]]] = []
    social_adj_lists[dic_node_o2i[line[0]]].append(dic_node_o2i[line[1]])
for user in social_adj_lists.keys():
    social_adj_lists[user] = set(social_adj_lists[user])  
    social_adj_lists[user] = list(social_adj_lists[user])

print('building other dicts')
for line in rating:
    user = line[0]
    item = line[1]
    # rate = ratings_list[line[2]]
    if dic_node_o2i[user] not in history_u_lists.keys():
        history_u_lists[dic_node_o2i[user]] = []
    if dic_node_o2i[item] not in history_v_lists.keys():
        history_v_lists[dic_node_o2i[item]] = []
    history_u_lists[dic_node_o2i[user]].append(dic_node_o2i[item])
    history_v_lists[dic_node_o2i[item]].append(dic_node_o2i[user])

valid_user = 0
avg_interaction = 0
for user in history_u_lists.keys():
    valid_user += 1
    avg_interaction += len(history_u_lists[user])
avg_interaction = avg_interaction / valid_user
print("avg_num_of_interaction:", avg_interaction)

valid_user = 0
avg_friend = 0
for user in social_adj_lists.keys():
    valid_user += 1
    avg_friend += len(social_adj_lists[user])
avg_friend = avg_friend / valid_user
print("avg_num_of_friend:", avg_friend)

# # item_adj_lists = build_item_adj_lists(history_v_lists)
# # user_collab = build_user_adj_lists(history_u_lists)
item_adj_lists = 1
user_collab = 1

pickle_data = [history_u_lists, 1, social_adj_lists, 1, 1, int(avg_interaction), int(avg_friend), num_of_target_users,
               num_of_all_users, num_of_nodes]
with open("flickr20241204.pkl", 'wb') as fo:
    pickle.dump(pickle_data, fo)