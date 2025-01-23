import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import scipy.sparse as sp
import torch
from utils import normalize_dense, normalize_sp
import pandas as pd
import pickle

def get_spRRT(config, args, rating_valid, rating_test):
    n_users = config['n_users']
    n_items = config['n_items']
    all_ratings = config['user_rating']
    dir5 = f'../raw dataset/{args.dataset}/sp_uu_collab_adj_mat_tr0613.npz'
    dir6 = f'../raw dataset/{args.dataset}/sp_uu_collab_adj_mat_val0613.npz'
    try:
        sp_uu_collab_adj_mat_tr0613 = sp.load_npz(dir5)
        sp_uu_collab_adj_mat_val0613 = sp.load_npz(dir6)
        print("already load sparse RRT")
    except Exception:
        sp_uu_collab_adj_mat_tr0613, sp_uu_collab_adj_mat_val0613 = create_spRRT(n_users, n_items, all_ratings, rating_valid, rating_test)
        sp.save_npz(dir5, sp_uu_collab_adj_mat_tr0613)
        sp.save_npz(dir6, sp_uu_collab_adj_mat_val0613)
    return sp_uu_collab_adj_mat_tr0613, sp_uu_collab_adj_mat_val0613

def create_spRRT(n_users, n_items, all_ratings, rating_valid, rating_test):
    R_tr = np.zeros((n_users, n_items))
    R_val = np.zeros((n_users, n_items))
    for uid in all_ratings.keys():
        for item in all_ratings[uid]:
            if item not in rating_test[uid]: # 不是 not item == rating_test[uid] 因为我们的dict values是list
                R_val[uid-1, item-n_users-1] = 1
                if item not in rating_valid[uid]:
                    R_tr[uid-1, item-n_users-1] = 1 # idx convert uid:0---n_users-1 itemid:0---n_items-1-n_users 这样可以两个nn.Embedding(n_users) (n_items)
    uu_collab_adj_mat_tr = np.dot(R_tr, R_tr.T) # 得到的矩阵对角元素巨大ri*ri 其他位置稀疏 options:1.对角元素处理 2.每个item至少n个交互才能稠密
    row, col = np.diag_indices_from(uu_collab_adj_mat_tr)
    uu_collab_adj_mat_tr[row, col] = 1
    uu_collab_adj_mat_tr = sp.dok_matrix(uu_collab_adj_mat_tr)

    uu_collab_adj_mat_val = np.dot(R_val, R_val.T)
    uu_collab_adj_mat_val[row, col] = 1
    uu_collab_adj_mat_val = sp.dok_matrix(uu_collab_adj_mat_val)
    
    return uu_collab_adj_mat_tr.tocsr(), uu_collab_adj_mat_val.tocsr()

def get_RRT(config, args, rating_valid, rating_test):
    n_users = config['n_users']
    n_items = config['n_items']
    all_ratings = config['user_rating']
    dir5 = f'../raw dataset/{args.dataset}/uu_collab_adj_mat0613.npz'
    try:
        uu_collab_adj_mat0613 = np.load(dir5)
        uu_collab_adj_mat_tr0613 = uu_collab_adj_mat0613['tr']
        uu_collab_adj_mat_val0613 = uu_collab_adj_mat0613['val']
        print("already load RRT")
    except Exception:
        uu_collab_adj_mat_tr0613, uu_collab_adj_mat_val0613 = create_RRT(n_users, n_items, all_ratings, rating_valid, rating_test)
        np.savez(dir5, tr=uu_collab_adj_mat_tr0613, val=uu_collab_adj_mat_val0613)
    return uu_collab_adj_mat_tr0613, uu_collab_adj_mat_val0613

def create_RRT(n_users, n_items, all_ratings, rating_valid, rating_test):
    R_tr = np.zeros((n_users, n_items))
    R_val = np.zeros((n_users, n_items))
    for uid in all_ratings.keys():
        for item in all_ratings[uid]:
            if item not in rating_test[uid]: # 不是 not item == rating_test[uid] 因为我们的dict values是list
                R_val[uid-1, item-n_users-1] = 1
                if item not in rating_valid[uid]:
                    R_tr[uid-1, item-n_users-1] = 1 # idx convert uid:0---n_users-1 itemid:0---n_items-1-n_users 这样可以两个nn.Embedding(n_users) (n_items)
    uu_collab_adj_mat_tr = np.dot(R_tr, R_tr.T) # 得到的矩阵对角元素巨大ri*ri 其他位置稀疏 options:1.对角元素处理 2.每个item至少n个交互才能稠密
    row, col = np.diag_indices_from(uu_collab_adj_mat_tr)
    uu_collab_adj_mat_tr[row, col] = 1
    uu_collab_adj_mat_tr = normalize_dense(uu_collab_adj_mat_tr)

    uu_collab_adj_mat_val = np.dot(R_val, R_val.T)
    uu_collab_adj_mat_val[row, col] = 1
    uu_collab_adj_mat_val = normalize_dense(uu_collab_adj_mat_val)
    
    return uu_collab_adj_mat_tr, uu_collab_adj_mat_val
    

def get_adj_mat(config, args, rating_valid, rating_test):
    n_users = config['n_users']
    n_target_users = config['n_target_users']
    n_items = config['n_items']
    social_network = config['user_social']
    all_ratings = config['user_rating']

    # dir1 = f'../raw dataset/{args.dataset}/uu_collab_adj_mat.npz'
    dir2 = f'../raw dataset/{args.dataset}/uu_social_adj_mat.npz'
    dir3 = f'../raw dataset/{args.dataset}/adj_mat_tr.npz'
    # dir4 = f'../raw dataset/{args.dataset}/adj_mat_val.npz'
    
    try:
        # uu_collab_adj_mat = np.load(dir1)
        uu_social_adj_mat = np.load(dir2)
        # uu_collab_adj_mat_tr = uu_collab_adj_mat['tr']
        # uu_collab_adj_mat_val = uu_collab_adj_mat['val']        
        uu_social_adj_mat = uu_social_adj_mat['all']
        A_tr = sp.load_npz(dir3)
        # A_val = sp.load_npz(dir4)
        print('already load')
    
    except Exception:
        _, _,  uu_social_adj_mat, A_tr, _  = min_create_adj_mat(n_users, n_items, all_ratings, rating_valid, rating_test, social_network)
        # np.savez(dir1, tr=uu_collab_adj_mat_tr, val=uu_collab_adj_mat_val)
        np.savez(dir2, all=uu_social_adj_mat)
        sp.save_npz(dir3, A_tr)
        # sp.save_npz(dir4, A_val)

    return None, None,  uu_social_adj_mat, A_tr, None

def get_adj_mat_shadow(config, args, rating_valid, rating_test):
    n_users = config['n_users']
    n_target_users = config['n_target_users']
    n_items = config['n_items']
    social_network = config['user_social']
    all_ratings = config['user_rating']

    # dir1 = f'{args.shadow_prefix}/{args.shadow_dataset}_uu_collab_adj_mat.npz'
    dir2 = f'{args.shadow_prefix}/{args.shadow_dataset}_uu_social_adj_mat.npz'
    dir3 = f'{args.shadow_prefix}/{args.shadow_dataset}_adj_mat_tr.npz'
    # dir4 = f'{args.shadow_prefix}/{args.shadow_dataset}_adj_mat_val.npz'
    
    try:
        # uu_collab_adj_mat = np.load(dir1)
        uu_social_adj_mat = np.load(dir2)
        # uu_collab_adj_mat_tr = uu_collab_adj_mat['tr']
        # uu_collab_adj_mat_val = uu_collab_adj_mat['val']        
        uu_social_adj_mat = uu_social_adj_mat['all']
        A_tr = sp.load_npz(dir3)
        # A_val = sp.load_npz(dir4)
        print('already load')
    
    except Exception:
        _, _,  uu_social_adj_mat, A_tr, _  = min_create_adj_mat(n_users, n_items, all_ratings, rating_valid, rating_test, social_network)
        # np.savez(dir1, tr=uu_collab_adj_mat_tr, val=uu_collab_adj_mat_val)
        np.savez(dir2, all=uu_social_adj_mat)
        sp.save_npz(dir3, A_tr)
        # sp.save_npz(dir4, A_val)

    return None, None,  uu_social_adj_mat, A_tr, None


def create_adj_mat(n_users, n_items, all_ratings, rating_valid, rating_test, social_network):
    """
      预处理的数据中item_idx += n_users (item_idx from 1)
      user中只有rating无social的部分已经被去除 所以all_ratings里存下的user全都是1--n_target_users  user_idx:1--n_target_users, n_target_users--n_users
      social_adj: [n_users, n_users]    collab_adj: [n_users, n_users]  # collab必须设置为n_users  n_target_users---n_users部分的RRT为0就好了 不影响其他用户
      上述情况在构建邻接矩阵的时候都要考虑
    """
    R_tr = np.zeros((n_users, n_items))
    R_val = np.zeros((n_users, n_items))
    for uid in all_ratings.keys():
        for item in all_ratings[uid]:
            if item not in rating_test[uid]: # 不是 not item == rating_test[uid] 因为我们的dict values是list
                R_val[uid-1, item-n_users-1] = 1
                if item not in rating_valid[uid]:
                    R_tr[uid-1, item-n_users-1] = 1 # idx convert uid:0---n_users-1 itemid:0---n_items-1-n_users 这样可以两个nn.Embedding(n_users) (n_items)
    uu_collab_adj_mat_tr = np.dot(R_tr, R_tr.T) # 得到的矩阵对角元素巨大ri*ri 其他位置稀疏 options:1.对角元素处理 2.每个item至少n个交互才能稠密
    uu_collab_adj_mat_tr = normalize_dense(uu_collab_adj_mat_tr)
    uu_collab_adj_mat_val = np.dot(R_val, R_val.T)
    uu_collab_adj_mat_val = normalize_dense(uu_collab_adj_mat_val)
    
    S = np.zeros((n_users, n_users)) 
    for uid in social_network.keys(): 
        for fid in social_network[uid]: 
            S[uid-1, fid-1] = 1 # 得到的矩阵非常稀疏 options: 迭代删除social用户等 
    uu_social_adj_mat = S
    uu_social_adj_mat = normalize_dense(uu_social_adj_mat)
    
    spR_tr = sp.dok_matrix(R_tr)
    spR_tr = spR_tr.tolil()
    adj_mat_tr = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32) 
    adj_mat_tr = adj_mat_tr.tolil() # convert it to list of lists format 
    adj_mat_tr[:n_users, n_users:] = spR_tr
    adj_mat_tr[n_users:, :n_users] = spR_tr.T
    adj_mat_tr = adj_mat_tr.todok()
    adj_mat_tr = normalize_sp(adj_mat_tr)
    
    spR_val = sp.dok_matrix(R_val)
    spR_val = spR_val.tolil()
    adj_mat_val = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32) 
    adj_mat_val = adj_mat_val.tolil() # convert it to list of lists format 
    adj_mat_val[:n_users, n_users:] = spR_val
    adj_mat_val[n_users:, :n_users] = spR_val.T
    adj_mat_val = adj_mat_val.todok()
    adj_mat_val = normalize_sp(adj_mat_val)
    
    return uu_collab_adj_mat_tr, uu_collab_adj_mat_val, uu_social_adj_mat, adj_mat_tr.tocsr(), adj_mat_val.tocsr()


def min_create_adj_mat(n_users, n_items, all_ratings, rating_valid, rating_test, social_network):
    """
      预处理的数据中item_idx += n_users (item_idx from 1)
      user中只有rating无social的部分已经被去除 所以all_ratings里存下的user全都是1--n_target_users  user_idx:1--n_target_users, n_target_users--n_users
      social_adj: [n_users, n_users]    collab_adj: [n_users, n_users]  # collab必须设置为n_users  n_target_users---n_users部分的RRT为0就好了 不影响其他用户
      上述情况在构建邻接矩阵的时候都要考虑
    """
    R_tr = np.zeros((n_users, n_items))
    R_val = np.zeros((n_users, n_items))
    for uid in all_ratings.keys():
        for item in all_ratings[uid]:
            if item not in rating_test[uid]: # 不是 not item == rating_test[uid] 因为我们的dict values是list
                R_val[uid-1, item-n_users-1] = 1
                if item not in rating_valid[uid]:
                    R_tr[uid-1, item-n_users-1] = 1 # idx convert uid:0---n_users-1 itemid:0---n_items-1-n_users 这样可以两个nn.Embedding(n_users) (n_items)
    # uu_collab_adj_mat_tr = np.dot(R_tr, R_tr.T) # 得到的矩阵对角元素巨大ri*ri 其他位置稀疏 options:1.对角元素处理 2.每个item至少n个交互才能稠密
    # uu_collab_adj_mat_tr = normalize_dense(uu_collab_adj_mat_tr)
    # uu_collab_adj_mat_val = np.dot(R_val, R_val.T)
    # uu_collab_adj_mat_val = normalize_dense(uu_collab_adj_mat_val)
    
    S = np.zeros((n_users, n_users)) 
    for uid in social_network.keys(): 
        for fid in social_network[uid]: 
            S[uid-1, fid-1] = 1 # 得到的矩阵非常稀疏 options: 迭代删除social用户等 
    uu_social_adj_mat = S
    uu_social_adj_mat = normalize_dense(uu_social_adj_mat)
    
    spR_tr = sp.dok_matrix(R_tr)
    spR_tr = spR_tr.tolil()
    adj_mat_tr = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32) 
    adj_mat_tr = adj_mat_tr.tolil() # convert it to list of lists format 
    adj_mat_tr[:n_users, n_users:] = spR_tr
    adj_mat_tr[n_users:, :n_users] = spR_tr.T
    adj_mat_tr = adj_mat_tr.todok()
    adj_mat_tr = normalize_sp(adj_mat_tr)
    
    # spR_val = sp.dok_matrix(R_val)
    # spR_val = spR_val.tolil()
    # adj_mat_val = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32) 
    # adj_mat_val = adj_mat_val.tolil() # convert it to list of lists format 
    # adj_mat_val[:n_users, n_users:] = spR_val
    # adj_mat_val[n_users:, :n_users] = spR_val.T
    # adj_mat_val = adj_mat_val.todok()
    # adj_mat_val = normalize_sp(adj_mat_val)
    
    return None, None, uu_social_adj_mat, adj_mat_tr.tocsr(), None

def datasetsplit(user_ratings, split):
    train_ratings = {}
    test_ratings = {}
    for user in user_ratings:
        size = len(user_ratings[user])
        train_ratings[user] = user_ratings[user][:int(split*size)]   
        test_ratings[user] = user_ratings[user][int(split*size):size]
    return train_ratings, test_ratings

def leave_one_out_split(user_ratings):
    train_ratings = {}
    valid_ratings = {}
    test_ratings = {}
    for user in user_ratings:
        # random.shuffle(user_ratings[user]) # 合理的吗？？
        size = len(user_ratings[user])
        # if size >= 3:
        train_ratings[user] = user_ratings[user][:size-2]
        valid_ratings[user] = user_ratings[user][size-2:size-1] # 返回单元素的list
        test_ratings[user] = user_ratings[user][size-1:size]
        # else:
        
    return train_ratings, valid_ratings, test_ratings
        
"""
dataloader正确做法: 1.if model 输入 [u,i] pair 保证所有的[u,i]pair被遍历 __len__=数据集所有正样本边个数 2. if model 输入 u and 所有i 保证u被遍历 __len__=数据集所有user个数
总结：总归是训练集所有的正样本边都被训练过
本次我们采用第一种
"""
class myTrainset(Dataset):
    """
    注意idx 
    """
    def __init__(self, config, train_data, neg):
        self.n_items = config['n_items']
        self.n_users = config['n_users']
        self.n_target_users = config['n_target_users']
        self.all_ratings = config['user_rating'] # dict
        self.neg = neg
        train_data_npy = self.get_numpy(train_data) 
        self.train_data_npy = train_data_npy # numpy
    
    def get_numpy(self, train_data):
        train_data_npy = []
        for uid in train_data:
            for item in train_data[uid]:
                train_data_npy.append([uid, item])
        train_data_npy = np.array(train_data_npy)
        return train_data_npy
    
    def __getitem__(self, index):
        """ 
        返回对应index的训练数据 (u,i,[neg个负样本])
        """
        user, pos_item = self.train_data_npy[index][0], self.train_data_npy[index][1] 
        neg_item = np.empty(self.neg, dtype=np.int32)
        for idx in range(self.neg):   
            t = np.random.randint(self.n_users+1, self.n_items+self.n_users+1) # [low, high) itemid: num_of_all_users+1--num_of_nodes
            while t in self.all_ratings[user]: # 不考虑二次负采样
                t = np.random.randint(self.n_users+1, self.n_items+self.n_users+1)
            neg_item[idx] = t-self.n_users-1 # 0开始
        return user-1, pos_item-self.n_users-1, neg_item
    
    def __len__(self): # all u,i pair
        return len(self.train_data_npy)

class myValidset(Dataset):
    
    def __init__(self, config, valid_data, candidate=999):
        self.n_items = config['n_items']
        self.n_users = config['n_users']
        self.n_target_users = config['n_target_users']
        self.all_ratings = config['user_rating'] # dict
        self.n_cnddt = candidate
        self.valid_data = valid_data # dict
    
    def __getitem__(self, user_idx):
        """
        返回对应index的验证数据 (u,i, 999*neg_i)
        """
        [pos_item] = self.valid_data[user_idx+1]
        neg_items = np.empty(self.n_cnddt, dtype=np.int32)
        for idx in range(self.n_cnddt):
            t = np.random.randint(self.n_users+1, self.n_items+self.n_users+1) # [low, high) itemid: num_of_all_users+1--num_of_nodes
            while t in self.all_ratings[user_idx+1]: 
                t = np.random.randint(self.n_users+1, self.n_items+self.n_users+1)
            neg_items[idx] = t-self.n_users-1 # 0开始
        return user_idx, pos_item-self.n_users-1, neg_items
        
    def __len__(self): # all target users
        return len(self.valid_data)

def get_train_loader(config, train_data, args):
    dataset = myTrainset(config, train_data, args.neg)
    # 每次都是随机打乱，然后分成大小为n的若干个mini-batch
    # droplast默认False https://www.cnblogs.com/vvzhang/p/15636814.html
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True) # 训练必shuffle防止训练集顺序影响模型训练 测试验证不用 
    return dataloader

def get_valid_loader(config, valid_data, args, num_negs=999):
    dataset = myValidset(config, valid_data, num_negs) # 如果是full test，就改这里；
    dataloader = DataLoader(dataset, batch_size=args.batch_size*2, shuffle=False)
    return dataloader

# 验证dataloader是否正确
# for batch_idx, (user, pos_item, neg_items) in enumerate(tqdm(train_loader, file=sys.stdout)):
    # print(user.size()) # [B]
    # print(pos_item.size()) # [B]
    # print(neg_items.size()) # [B,neg]
#     if int(pos_item)+config['n_users']+1 not in train_data[int(user)+1]:
#         print("wrong")
#     for i in neg_items[0]:       
#         if int(i)+config['n_users']+1 in train_data[int(user)+1]:
#             print("wrong")


class atkData(Dataset):
    def __init__(self, config, fname, shadow_param, rec_result, mtd='uemb'):
        self.config = config
        self.data_df = pd.read_csv(config[fname+'_path'])
        self.shadow_param = shadow_param
        self.rec_result = rec_result
        # if 'user_embs_A.weight' in self.shadow_param.keys():
        #     self.shadow_param['user_embs.weight'] = 
        # self.shadow_param = torch.load(config['param_path'])
        # self.rec_result = pickle.load(data_file)
        # print('data_df', self.data_df.head())
        # print('rec_result', self.rec_result[1])
        self._get_attack_feature(mtd)

    def _get_attack_feature(self, mtd):
        self.data_x = []
        self.data_y = []
        # if mtd == 'my':
        #     self.shadow_param[]
        for row in self.data_df.itertuples():
            # dataset 里面的user id对应到emb应该-1；item应该-1-u_len
            user1, user2, y = row.user1, row.user2, row.y
            y = torch.tensor(row.y, dtype=torch.float).to(self.config['gpu_id'])
            if mtd == 'uemb':
                x = self._get_uemb_feat(user1, user2)
            # 4 baselines
            elif mtd == 'miner':
                x = self._get_miner_feat(user1, user2)
            elif mtd == 'enminer':
                x = self._get_enminer_feat(user1, user2)
            elif mtd == 'miars':
                x = self._get_miars_feat(user1, user2)
            elif mtd == 'enmiars':
                x = self._get_enmiars_feat(user1, user2)
            elif mtd == 'uiemb':
                x = self._get_user_item_feat(user1, user2)
            elif mtd == 'my':
                x = self._get_my_feat(user1, user2)
                # x = self._ablation_get_my_feat(user1, user2)

            self.data_x.append(x)
            self.data_y.append(y)
    
    def _get_my_feat(self, user1, user2):
        user_embs_A = self.shadow_param['user_embs_A.weight']
        user_embs_S = self.shadow_param['user_embs_S.weight']
        if 'item_embs.weight' not in self.shadow_param.keys():
            self.shadow_param['item_embs.weight'] = (self.shadow_param['item_embs_S.weight'] + self.shadow_param['item_embs_A.weight']) / 2
            
        u1_emb_A = user_embs_A[user1-1]
        u1_emb_S = user_embs_S[user1-1]
        u1_item_embs = self._get_iemb_by_user(user1)
        u1_item_embs_mean = u1_item_embs.mean(dim=0)
        u1_item_embs_var = u1_item_embs.var(dim=0)

        u2_emb_A = user_embs_A[user2-1]
        u2_emb_S = user_embs_S[user2-1]
        u2_item_embs = self._get_iemb_by_user(user2)
        u2_item_embs_mean = u2_item_embs.mean(dim=0)
        u2_item_embs_var = u2_item_embs.var(dim=0)

        # u1_emb_A = torch.abs(u1_emb_A - u1_emb_S)
        # u2_emb_A = torch.abs(u2_emb_A - u2_emb_S)
        u1_emb_A = torch.cat([u1_emb_A, u1_emb_S], dim=0)
        u2_emb_A = torch.cat([u2_emb_A, u2_emb_S], dim=0)
        u1_feat = torch.cat([u1_emb_A, u1_item_embs_mean, u1_item_embs_var], dim=0)
        u2_feat = torch.cat([u2_emb_A, u2_item_embs_mean, u2_item_embs_var], dim=0)
        # u1_feat = torch.cat([u1_emb_A, u1_emb_S, u1_item_embs_mean, u1_item_embs_var], dim=0)
        # u2_feat = torch.cat([u2_emb_A, u2_emb_S, u2_item_embs_mean, u2_item_embs_var], dim=0)

        # pairwise_feat = torch.cat([u1_feat, u2_feat], dim=0)
        diff_feat = torch.abs(u1_feat - u2_feat)
        interaction_feat = u1_feat * u2_feat
        pairwise_feat = torch.cat([u1_feat, u2_feat, diff_feat, interaction_feat], dim=0)

        return pairwise_feat

    def _ablation_get_my_feat(self, user1, user2):
        # w/o social; only interaction;
        # 去掉behavioral, 只有social的, 得到fused social information
        # user_embs_A = self.shadow_param['user_embs_A.weight']
        user_embs_S = self.shadow_param['user_embs_S.weight']
        if 'item_embs.weight' not in self.shadow_param.keys():
            # self.shadow_param['item_embs.weight'] = (self.shadow_param['item_embs_S.weight'] + self.shadow_param['item_embs_A.weight']) / 2
            self.shadow_param['item_embs.weight'] = self.shadow_param['item_embs_S.weight']

        # u1_emb_A = user_embs_A[user1-1]
        u1_emb_S = user_embs_S[user1-1]
        u1_item_embs = self._get_iemb_by_user(user1)
        u1_item_embs_mean = u1_item_embs.mean(dim=0)
        u1_item_embs_var = u1_item_embs.var(dim=0)

        # u2_emb_A = user_embs_A[user2-1]
        u2_emb_S = user_embs_S[user2-1]
        u2_item_embs = self._get_iemb_by_user(user2)
        u2_item_embs_mean = u2_item_embs.mean(dim=0)
        u2_item_embs_var = u2_item_embs.var(dim=0)

        # u1_emb_A = torch.abs(u1_emb_A - u1_emb_S)
        # u2_emb_A = torch.abs(u2_emb_A - u2_emb_S)
        # u1_emb_A = torch.cat([u1_emb_A, u1_emb_S], dim=0)
        # u2_emb_A = torch.cat([u2_emb_A, u2_emb_S], dim=0)
        u1_emb_A = torch.abs(u1_emb_S)
        u2_emb_A = torch.abs(u2_emb_S)
        # u1_feat = torch.cat([u1_emb_A], dim=0)
        # u2_feat = torch.cat([u1_emb_A], dim=0)
        u1_feat = torch.cat([u1_emb_A, u1_item_embs_mean, u1_item_embs_var], dim=0)
        u2_feat = torch.cat([u2_emb_A, u2_item_embs_mean, u2_item_embs_var], dim=0)
        # u1_feat = torch.cat([u1_emb_A, u1_item_embs_mean, u1_item_embs_var], dim=0)
        # u2_feat = torch.cat([u2_emb_A, u2_item_embs_mean, u2_item_embs_var], dim=0)

        # pairwise_feat = torch.cat([u1_feat, u2_feat], dim=0)
        diff_feat = torch.abs(u1_feat - u2_feat)
        interaction_feat = u1_feat * u2_feat
        # pairwise_feat = torch.cat([u1_feat, u2_feat], dim=0)
        pairwise_feat = torch.cat([u1_feat, u2_feat, diff_feat, interaction_feat], dim=0)

        return pairwise_feat

    def _get_uemb_by_user(self, user):
        return self.shadow_param['user_embs.weight'][user-1]
    
    def _get_iemb_by_user(self, user):
        # will return (30, 64)
        u1_items = torch.tensor(self.rec_result[user]).to(self.config['gpu_id']) # 不用-1，rec result 就是从1开始村的;
        return self.shadow_param['item_embs.weight'][u1_items]

    def _get_uemb_feat(self, user1, user2):
        u1_emb = self._get_uemb_by_user(user1)
        u2_emb = self._get_uemb_by_user(user2)
        x_feat = torch.cat([u1_emb, u2_emb], dim=0)
        return x_feat
    
    # CCS 21 Membership Inference Attacks Against Recommender Systems
    def _get_miars_feat(self, user1, user2):
        u1_item_embs = self._get_iemb_by_user(user1).mean(dim=0)
        u2_item_embs = self._get_iemb_by_user(user2).mean(dim=0)
        return u1_item_embs - u2_item_embs
        # x_feat = torch.cat([u1_item_embs, u2_item_embs], dim=0)
        # return x_feat
    
    # CCS 21 Membership Inference Attacks Against Recommender Systems
    def _get_enmiars_feat(self, user1, user2):
        u1_item_embs = self._get_iemb_by_user(user1).mean(dim=0)
        u2_item_embs = self._get_iemb_by_user(user2).mean(dim=0)
        # return u1_item_embs - u2_item_embs
        x_feat = torch.cat([u1_item_embs, u2_item_embs], dim=0)
        return x_feat

    def _get_user_item_feat(self, user1, user2):
        u1_emb = self._get_uemb_by_user(user1)
        u2_emb = self._get_uemb_by_user(user2)
        u1_item_embs = self._get_iemb_by_user(user1).mean(dim=0)
        u2_item_embs = self._get_iemb_by_user(user2).mean(dim=0)
        x_feat = torch.cat([u1_emb, u2_emb, u1_item_embs, u2_item_embs], dim=0)
        return x_feat

    # CIKM 24 Interaction-level Membership Inference Attack against Recommender Systems with Long-tailed Distribution
    def _get_miner_feat(self, user1, user2):
        user_embs = self.shadow_param['user_embs.weight']
        item_embs = self.shadow_param['item_embs.weight']
        # rec result 的item id是item emb对应的idx
        # rec result 的user id是user emb对应idx+1
        u1_items = torch.tensor(self.rec_result[user1]).to(self.config['gpu_id']) 
        u1_item_embs = item_embs[u1_items]
        u2_emb = user_embs[user2-1]

        # single u
        l1_dis = torch.sum(torch.abs(u2_emb - u1_item_embs), dim=1)
        l2_dis = torch.sqrt(torch.sum((u2_emb - u1_item_embs)**2, dim=1))
        cos_dis = 1 - torch.matmul(u1_item_embs,u2_emb) / (torch.norm(u2_emb)*torch.norm(u1_item_embs, dim=1))
        # bray-curtis distance
        bc_dis = torch.sum(torch.abs(u2_emb - u1_item_embs), dim=1) / torch.sum(torch.abs(u2_emb + u1_item_embs), dim=1)
        x_feat = torch.cat([l1_dis, l2_dis, cos_dis, bc_dis])
        # single u end

        # u1 item & u2
        # u2_items = torch.tensor(self.rec_result[user2]).to(self.config['gpu_id'])
        # u2_item_embs = item_embs[u2_items]
        # u1_emb = user_embs[user1-1]

        # l1_dis = torch.sum(torch.abs(u2_emb - u1_item_embs), dim=1)
        # l2_dis = torch.sqrt(torch.sum((u2_emb - u1_item_embs)**2, dim=1))
        # cos_dis = 1 - torch.matmul(u1_item_embs,u2_emb) / (torch.norm(u2_emb)*torch.norm(u1_item_embs, dim=1))
        # bc_dis = torch.sum(torch.abs(u2_emb - u1_item_embs), dim=1) / torch.sum(torch.abs(u2_emb + u1_item_embs), dim=1) # bray-curtis distance
        
        # l1_dis2 = torch.sum(torch.abs(u1_emb - u2_item_embs), dim=1)
        # l2_dis2 = torch.sqrt(torch.sum((u1_emb - u2_item_embs)**2, dim=1))
        # cos_dis2 = 1 - torch.matmul(u2_item_embs, u1_emb) / (torch.norm(u1_emb)*torch.norm(u2_item_embs, dim=1))
        # bc_dis2 = torch.sum(torch.abs(u1_emb - u2_item_embs), dim=1) / torch.sum(torch.abs(u1_emb + u2_item_embs), dim=1) # bray-curtis distance
        # x_feat = torch.cat([l1_dis, l2_dis, cos_dis, bc_dis, l1_dis2, l2_dis2, cos_dis2, bc_dis2])
        
        return x_feat
        
    # CIKM 24 Interaction-level Membership Inference Attack against Recommender Systems with Long-tailed Distribution
    def _get_enminer_feat(self, user1, user2):
        user_embs = self.shadow_param['user_embs.weight']
        item_embs = self.shadow_param['item_embs.weight']
        # rec result 的item id是item emb对应的idx
        # rec result 的user id是user emb对应idx+1
        u1_items = torch.tensor(self.rec_result[user1]).to(self.config['gpu_id']) 
        u1_item_embs = item_embs[u1_items]
        u2_emb = user_embs[user2-1]

        # # single u
        # l1_dis = torch.sum(torch.abs(u2_emb - u1_item_embs), dim=1)
        # l2_dis = torch.sqrt(torch.sum((u2_emb - u1_item_embs)**2, dim=1))
        # cos_dis = 1 - torch.matmul(u1_item_embs,u2_emb) / (torch.norm(u2_emb)*torch.norm(u1_item_embs, dim=1))
        # # bray-curtis distance
        # bc_dis = torch.sum(torch.abs(u2_emb - u1_item_embs), dim=1) / torch.sum(torch.abs(u2_emb + u1_item_embs), dim=1)
        # x_feat = torch.cat([l1_dis, l2_dis, cos_dis, bc_dis])
        # single u end

        # u1 item & u2
        u2_items = torch.tensor(self.rec_result[user2]).to(self.config['gpu_id'])
        u2_item_embs = item_embs[u2_items]
        u1_emb = user_embs[user1-1]

        l1_dis = torch.sum(torch.abs(u2_emb - u1_item_embs), dim=1)
        l2_dis = torch.sqrt(torch.sum((u2_emb - u1_item_embs)**2, dim=1))
        cos_dis = 1 - torch.matmul(u1_item_embs,u2_emb) / (torch.norm(u2_emb)*torch.norm(u1_item_embs, dim=1))
        bc_dis = torch.sum(torch.abs(u2_emb - u1_item_embs), dim=1) / torch.sum(torch.abs(u2_emb + u1_item_embs), dim=1) # bray-curtis distance
        
        l1_dis2 = torch.sum(torch.abs(u1_emb - u2_item_embs), dim=1)
        l2_dis2 = torch.sqrt(torch.sum((u1_emb - u2_item_embs)**2, dim=1))
        cos_dis2 = 1 - torch.matmul(u2_item_embs, u1_emb) / (torch.norm(u1_emb)*torch.norm(u2_item_embs, dim=1))
        bc_dis2 = torch.sum(torch.abs(u1_emb - u2_item_embs), dim=1) / torch.sum(torch.abs(u1_emb + u2_item_embs), dim=1) # bray-curtis distance
        x_feat = torch.cat([l1_dis, l2_dis, cos_dis, bc_dis, l1_dis2, l2_dis2, cos_dis2, bc_dis2])
        
        return x_feat
        
    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]
    
    def __len__(self):
        return len(self.data_x)

class atkData2(Dataset):
    def __init__(self, config, fname):
        self.config = config
        self.data_df = pd.read_csv(config[fname+'_path'])

    def __getitem__(self, idx):
        torch_y = torch.tensor(self.data_df['y'].iloc[idx], dtype=torch.float).to(self.config['gpu_id'])
        return self.data_df['user1'].iloc[idx]-1, self.data_df['user2'].iloc[idx]-1, torch_y
    
    def __len__(self):
        return len(self.data_df)

class defendData(Dataset):
    def __init__(self, args, train_df, all_param):
        self.data_df = train_df
        self.shadow_param = all_param
        self.args = args
        self._get_attack_feature()

    def _get_attack_feature(self):
        self.data_x = []
        self.data_y = []
        for row in self.data_df.itertuples():
            # dataset 里面的user id对应到emb应该-1；item应该-1-u_len
            user1, user2, y = row.user1, row.user2, row.y
            y = torch.tensor(row.y, dtype=torch.float).to(self.args.gpu_id)
            x = self._get_uemb_feat(user1, user2)

            self.data_x.append(x)
            self.data_y.append(y)
    
    def _get_uemb_by_user(self, user):
        return self.shadow_param['user_embs.weight'][user-1]
    
    # def _get_iemb_by_user(self, user):
    #     # will return (30, 64)
    #     u1_items = torch.tensor(self.rec_result[user]).to(self.config['gpu_id']) # 不用-1，rec result 就是从1开始村的;
    #     return self.shadow_param['item_embs.weight'][u1_items]

    def _get_uemb_feat(self, user1, user2):
        u1_emb = self._get_uemb_by_user(user1)
        u2_emb = self._get_uemb_by_user(user2)
        x_feat = torch.cat([u1_emb, u2_emb], dim=0)
        return x_feat
      
    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]
    
    def __len__(self):
        return len(self.data_x)

class uPairData(Dataset):
    def __init__(self, train_df):
        self.data_df = train_df

    def __getitem__(self, idx):
        return self.data_df['user1'].iloc[idx]-1, self.data_df['user2'].iloc[idx]-1
    
    def __len__(self):
        return len(self.data_df)

