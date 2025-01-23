import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import random
from utils import seed_all
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, roc_curve, auc, f1_score
# from sklearn.metrics import
from arg_parser import parse_args

args = parse_args()
seed_all(42)
if args.dataset == 'ciao':
    target_name = 'ciao20230314.pkl'
elif args.dataset == 'flickr':
    target_name = 'flickr20241204.pkl'
elif args.dataset == 'yelp':
    data_name = 'yelp_small.pkl'

data_path = f'../raw dataset/{args.dataset}/'
data_dir = f'{args.shadow_prefix}/{args.shadow_dataset}'
data_file = open(data_dir, 'rb')
_, _, social_adj_lists, _, rand_u_list, _, _, num_of_target_users, num_of_all_users, _ = pickle.load(data_file)
data_file.close()

ori_data_dir = data_path + target_name
data_file = open(ori_data_dir, 'rb')
_, _, gt_social, _, _, _, _, _, _, _ = pickle.load(data_file)
data_file.close()

print('-'*10)
print('[TRAIN] generate training dataset')
# The attack training dataset consists of all the interactions in the shadow training graph as members.
# We also randomly sample the same number of non-interacted user-item pairs from the shadow training graph, 
# and include them in the attack training dataset as non-members
training_dataset = {
    'user1': [],
    'user2': [],
    'y': []
}

already_u_pair = set()
train_pos_u_pair = set()
train_neg_u_pair = set()
for u1 in social_adj_lists.keys():
    for u2 in social_adj_lists[u1]:
        already_u_pair.add((u1, u2))
        train_pos_u_pair.add((u1, u2))
        training_dataset['user1'].append(u1)
        training_dataset['user2'].append(u2)
        training_dataset['y'].append(1)
# print(pos_u_pair)

pos_len = len(training_dataset['user1'])
print('pos sample num', pos_len, 'now dataset num', len(already_u_pair))

# num of target users 
n_neg = 0
while n_neg < pos_len:
    u1 = random.randint(1, num_of_target_users)
    u2 = random.randint(1, num_of_target_users)
    while u1 == u2:
       u2 = random.randint(1, num_of_target_users)
    if (u1, u2) in already_u_pair:
        continue
    train_neg_u_pair.add((u1, u2))
    already_u_pair.add((u1, u2))
    training_dataset['user1'].append(u1)
    training_dataset['user2'].append(u2)
    training_dataset['y'].append(0)
    n_neg += 1
     
print('neg sample num', n_neg, 'now dataset num', len(already_u_pair))

train_df = pd.DataFrame(training_dataset)
shuffled_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
train_data_path = f'{args.shadow_prefix}/{args.shadow_train}'
shuffled_df.to_csv(train_data_path, index=False)
print(f'Training data saved in {train_data_path}')
print('-'*10)

print('\n')
print('[TEST] generate test dataset')
all_pos_u_pair = set()
cnt_all_pos_pairs = 0
for u1 in gt_social.keys():
    cnt_all_pos_pairs += len(gt_social[u1])
    for u2 in gt_social[u1]:
        all_pos_u_pair.add((u1, u2))

# gen test pos u pair; 
test_pos_u_all = all_pos_u_pair - train_pos_u_pair
test_pos_u_all = list(test_pos_u_all)
print(f'test all pos pairs: {len(test_pos_u_all)}, train pos pairs: {len(train_pos_u_pair)}')
print(f'target graph pairs: {cnt_all_pos_pairs}, sum of train and test: {len(test_pos_u_all)+len(train_pos_u_pair)}')

# random.seed(43)
len_all = len(all_pos_u_pair)
random.shuffle(test_pos_u_all)
test_upair = test_pos_u_all[:len_all//3]
print(len(test_upair), test_upair[:5])

test_dataset = {
    'user1': [],
    'user2': [],
    'y': []
}

for upair in test_upair:
    u1, u2 = upair
    test_dataset['user1'].append(u1)
    test_dataset['user2'].append(u2)
    test_dataset['y'].append(1)

pos_len = len(test_dataset['user1'])
print(pos_len, len(test_upair))

gt_pos_pair = set()
for u1 in gt_social.keys():
    for u2 in gt_social[u1]:
        gt_pos_pair.add((u1, u2))

n_neg = 0
while n_neg < pos_len:
    u1 = random.randint(1, num_of_target_users)
    u2 = random.randint(1, num_of_target_users)
    while u1 == u2:
       u2 = random.randint(1, num_of_target_users)
    if (u1, u2) in gt_pos_pair or (u1, u2) in train_neg_u_pair:
        continue
    gt_pos_pair.add((u1, u2))
    test_dataset['user1'].append(u1)
    test_dataset['user2'].append(u2)
    test_dataset['y'].append(0)
    n_neg += 1
    
print(n_neg)

test_df = pd.DataFrame(test_dataset)
shuffled_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
print(test_df.head())
print(shuffled_df.head())
test_data_path = f'{args.shadow_prefix}/{args.shadow_test}'
shuffled_df.to_csv(test_data_path, index=False)
print(f'Test data saved in {test_data_path}')
print('-'*10)