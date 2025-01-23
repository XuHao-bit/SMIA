import argparse
from utils import mkdir_ifnotexist

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def parse_args():
    # 用parser的话可以命令行写入参数  python main.py --hop 2 --hidden 32
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dataset', default='ciao')
    # parser.add_argument('--data_dir', default='../raw dataset/ciao/ciao20230314.pkl')
    # parser.add_argument('--dataset', default='flickr')
    # parser.add_argument('--data_dir', default='../raw dataset/flickr/flickr20230314.pkl')
    # parser.add_argument('--dataset', default='yelp')
    # parser.add_argument('--data_dir', default='../raw dataset/yelp/yelp_small.pkl')
    parser.add_argument('--model_name', default="DESIGN")  
    
    parser.add_argument('--gpu_id', default=0, type=int)
    # training hyper_parameter
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--num_epoch', default=200, type=int)
    parser.add_argument('--hop', default=2, type=int) # 3
    parser.add_argument('--hidden', default=64, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--neg', default=1, type=int) # 训练时的neg sample
    # parser.add_argument('--split', default=0.8, type=float)
    parser.add_argument('--std', default=0.1, type=float) # 无用
    parser.add_argument('--decay', default=1e-3, type=float)
    
    # IDGL hyper_parameter 
    parser.add_argument('--graph_learn_hidden_size', default=70, type=int)
    parser.add_argument('--graph_learn_top_k_S', default=30, type=int) # 无用
    parser.add_argument('--graph_learn_epsilon', default=0, type=float)
    parser.add_argument('--graph_skip_conn', default=0.8, type=float)
    parser.add_argument('--graph_learn_num_pers', default=4, type=int)
    parser.add_argument('--metric_type', default='weighted_cosine', type=str)
    
    # ssl hyper_parameter
    parser.add_argument('--ssl_temp', default=0.2, type=float)
    parser.add_argument('--ssl_reg', default=1e-6, type=float) # 0.1/0.2
    parser.add_argument('--ssl_ratio', default=0.1, type=float) # 无用
    parser.add_argument('--ssl_aug_type', default='ed', type=str) # 无用
    
    # recon hyper_parameter
    parser.add_argument('--recon_reg', default=0.2, type=float)
    parser.add_argument('--recon_drop', default=0.8, type=float)
    
    # kl hyper_parameter
    parser.add_argument('--kl_reg', default=0, type=float)
    
    # test 
    parser.add_argument('--mtd', default='UI+social', type=str) 
    parser.add_argument('--is_shadow', type=boolean_string, default=False) 
    parser.add_argument('--seed', default=42, type=int) 
    
    # parser.add_argument('--shadow_name', default='DESIGN-shadow-e100.pth', type=str) 
    parser.add_argument('--trained_name', default='DESIGN-ciao-final.pth', type=str)  # 正常训练好的推荐模型，只开放推荐接口
    parser.add_argument('--shadow_train', default='shadow_ciao_train.csv', type=str) 
    parser.add_argument('--shadow_test', default='shadow_ciao_test.csv', type=str)  
    parser.add_argument('--model_save_name', default=None, type=str)  
    # parser.add_argument('--rec_result', default='ciao_design_result_UI+social-top30-pp.pkl', type=str) 
    parser.add_argument('--agg_mtd', default='uiemb', type=str)
    parser.add_argument('--shadow_ratio', default=0.1, type=float) # ratio of shadow users
    parser.add_argument('--rec_lens', default=30, type=int) # ratio of shadow users

    # defend 
    parser.add_argument('--defend_warm', default=5, type=int) 
    parser.add_argument('--pri_epoch', default=3, type=int) # local epoch for training defender
    parser.add_argument('--pri_coef', default=0.1, type=float)
    parser.add_argument('--dfd_lr', default=0.01, type=float) # defender lr in pp training
    parser.add_argument('--atk_lr', default=0.01, type=float) # attack lr in mia

    # defend baseline
    parser.add_argument('--pp_mtd', default='er', type=str) # attack lr in mia

    args = parser.parse_args()

    pref = '../raw dataset/'
    # train target rec model
    # or train mia attack
    args.shadow_name = f'shadow_{args.shadow_ratio}_{args.trained_name}' # shadow model name
    args.rec_result = f'{args.dataset}_{args.trained_name}-top{args.rec_lens}-pp.pkl' 
    args.shadow_dataset = f'shadow_{args.shadow_ratio}_{args.rec_result}'
    args.shadow_train = f'mia_train_{args.shadow_dataset}.csv' # mia train and test
    args.shadow_test = f'mia_test_{args.shadow_dataset}.csv' 
    args.shadow_prefix = f'./social_mia/{args.trained_name}'
    mkdir_ifnotexist(args.shadow_prefix)

    if not args.is_shadow:
        if args.dataset == 'ciao':
            data_name = 'ciao20230314.pkl' 
            # 是训练时候简单测试用的，真实测试用不到
            args.ppmain_trn = f'../raw dataset/ciao/social_mia/mia_train_shadow_0.1_ciao.csv'
            args.ppmain_tst = f'../raw dataset/ciao/social_mia/mia_test_shadow_0.1_ciao.csv'
        elif args.dataset == 'flickr':
            data_name = 'flickr20241204.pkl' 
            args.ppmain_trn = f'../raw dataset/flickr/social_mia/mia_train_shadow_0.1_flickr.csv'
            args.ppmain_tst = f'../raw dataset/flickr/social_mia/mia_test_shadow_0.1_flickr.csv'
        elif args.dataset == 'yelp':
            data_name = 'yelp_small.pkl'
        args.data_dir = pref + args.dataset + '/' + data_name
    else: # train shadow rec model
        data_name = f'{args.shadow_prefix}/{args.shadow_dataset}'
        args.model_save_name = f'shadow_{args.shadow_ratio}_{args.trained_name}'
        # 因为每个model的shadow dataset不一样，所以在model自己的文件夹下找shadow dataset
        args.data_dir = data_name

    return args