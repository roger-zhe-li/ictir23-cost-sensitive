import copy
import json
import math
import os
import random
import argparse

import msgpack
from tqdm import tqdm
import lmdb
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn
from torch.utils.data import DataLoader, Dataset
import logging

from util.utils import *
from util.parser import parse_args

import torch.nn.utils.rnn as rnn_utils
import torchsnooper
import scipy.stats
import shutil

from data_loader_random import DeepCoNNDataLoader
from model.core_random import *
from eval_random import evaluation

np.set_printoptions(suppress=True)
logger = logging.getLogger('RATE.DeepCoNN.train_test')
dir_path = os.path.dirname((os.path.abspath(os.path.join(os.path.realpath(__file__), './.'))))


args = parse_args()  
args.cuda = not args.no_cuda and torch.cuda.is_available()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def set_random_seed(state=1):
	gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
	for set_state in gens:
		set_state(state)

# general parameters
dataset = args.dataset
fold = args.fold 
dataset_path = args.ds_path
db_path = args.db_path
feature_path = args.feature_path
feature_model_path = args.feature_model_path

dir_exists(feature_path)
dir_exists(feature_model_path)
dir_exists(db_path)

# model parameters
batch_size = args.batch_size 
reg = args.weight_decay
lr = args.lr 
epochs = args.epochs 
review_length = args.review_length
word_vec_dim = args.word_vec_dim
conv_length = args.conv_length 
conv_kernel_num = args.conv_kernel_num 
latent_factor_num = args.latent_factor_num
sample_ratio = args.sample_ratio
train_ratio = args.train_ratio

# util parameters
rebuild = args.rebuild
retrain = args.retrain
tower_type = args.tower_type
data_type = args.data_type
train_low = args.train_low
valid_low = args.valid_low

qcut = args.qcut

seed = args.seed
set_random_seed(seed)
# print(l)

if data_type == 'review':
	data_file = 'train.csv'
else:
	pass

train_data_path = os.path.join(dataset_path, dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low), 'fold_'+str(fold), data_file)
meta_data_path = os.path.join(dataset_path, dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low), 'dataset_meta_info.json')

# train_data_path = os.path.join(dataset_path, dataset, 'fold_'+str(fold), data_file)
# meta_data_path = os.path.join(dataset_path, dataset, 'dataset_meta_info.json')

data_pd = pd.read_csv(train_data_path, header=0, index_col=False)
with open(meta_data_path) as f:
	dataset_meta_info = json.load(f)
	user_size = dataset_meta_info['user_size']
	item_size = dataset_meta_info['item_size']

# df_ind = pd.read_csv(os.path.join(dir_path, 'dataset', dataset+'.csv'), header=None, index_col=None)
# ind_selected = df_ind[0].tolist()
mask = np.ones(user_size)
# mask[ind_selected] = 1


# train_path = './dataset/movielens/fold_0/train.csv'
# valid_path = './dataset/movielens/fold_0/valid.csv'
# test_path = './dataset/movielens/fold_0/test.csv'

train_path = os.path.join(os.path.dirname(train_data_path), 'train.csv')
valid_path = os.path.join(os.path.dirname(train_data_path), 'valid.csv')
test_path = os.path.join(os.path.dirname(train_data_path), 'test.csv')

 
train_loader = DeepCoNNDataLoader(train_path=train_path,
							valid_path=valid_path,
							test_path=test_path,
							batch_size=512, 
							device=device,
							n_user=user_size,
							n_item=item_size,
							num_ng=4,
							num_test=500,
							# num_test=int(500*sample_ratio),
							subset='train',
							train_ratio=train_ratio,
							sample_ratio=sample_ratio,
							seed=seed,
							shuffle=True)

valid_loader = DeepCoNNDataLoader(train_path=train_path,
							valid_path=valid_path,
							test_path=test_path,
							batch_size=500,
							# batch_size=int(500*sample_ratio), 
							device=device,
							n_user=user_size,
							n_item=item_size,
							num_ng=4,
							num_test=500,
							# num_test=int(500*sample_ratio),
							subset='valid',
							train_ratio=train_ratio,
							sample_ratio=sample_ratio,
							seed=seed,
							shuffle=False)

test_loader = DeepCoNNDataLoader(train_path=train_path,
							valid_path=valid_path,
							test_path=test_path,
							batch_size=500,
							# batch_size=int(500*sample_ratio), 
							device=device,
							n_user=user_size,
							n_item=item_size,
							num_ng=4,
							num_test=500,
							# num_test=int(500*sample_ratio),
							subset='test',
							train_ratio=train_ratio,
							sample_ratio=sample_ratio,
							seed=seed,
							shuffle=False)

model = DeepCoNN(fm_k=32,
			     conv_length=3,
			     conv_kernel_num=100,
			     latent_factor_num=latent_factor_num,
			     ).to(device)

loss_func = nn.BCEWithLogitsLoss(reduction='none')
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
										   lr=3e-4)
# min_valid_loss = float('inf')
NDCG_max = 0
AP_max = 0

user_factors = torch.nn.Embedding(user_size, latent_factor_num).requires_grad_(requires_grad=False).to(device)
item_factors = torch.nn.Embedding(item_size, latent_factor_num).requires_grad_(requires_grad=False).to(device)
                                              

for e in tqdm(range(300)):
	for users, items, rels in train_loader:
		users_ = torch.LongTensor(users).to(device)
		items_ = torch.LongTensor(items).to(device)
		user_review_vec = user_factors(users_)
		item_review_vec = item_factors(items_)

		pred = model(user_review_vec, item_review_vec)
		user_freq = torch.bincount(torch.LongTensor(users), minlength=user_size).requires_grad_(False)
		item_freq = torch.bincount(torch.LongTensor(items), minlength=item_size).requires_grad_(False)
		user_count = user_freq[users].to(device)
		item_count = item_freq[items].to(device)

		user_weight_all = torch.Tensor(mask).to(device).requires_grad_(False)
		user_weight = user_weight_all[users].to(device)

		# train_loss = loss_func(pred, rels.flatten())
		train_loss = user_weight * loss_func(pred, rels.flatten())
		batch_loss = train_loss.sum() / user_weight.sum()
		optimizer.zero_grad()
		# train_loss.mean().backward()
		batch_loss.mean().backward()
		optimizer.step()
		# print(train_loss.mean().detach().cpu().numpy())

	model_file = 'test.pth.tar'
	# model_save_path = os.path.join('./saved_models_random', 'fold_0', 'seed_'+str(seed), 'ratio_'+str(train_ratio)+'_'+str(sample_ratio), model_file)
	model_save_path = os.path.join('./saved_models_random', dataset, 'fold_0', 'seed_'+str(seed), 'train_'+str(train_low)+'_'+'valid_'+str(valid_low)+'_sample_'+str(sample_ratio), 'fold_0', 'seed_'+str(seed), model_file)
	if ((e + 1) % 5 == 0) and (e >= 9):
		valid_loss, _, _, NDCG, AP = evaluation(model, valid_loader, device, num_test=500, user_factors=user_factors, item_factors=item_factors, weight=mask)

		if NDCG > NDCG_max:
			NDCG_max = NDCG
			dir_exists(os.path.dirname(model_save_path))
			save_model(e, model, valid_loss, optimizer, model_save_path)


# test for valid
model_file = 'test.pth.tar'
# model_save_path = os.path.join('./saved_models_random', 'fold_0', 'seed_'+str(seed), 'ratio_'+str(train_ratio)+'_'+str(sample_ratio), model_file)
model_save_path = os.path.join('./saved_models_random', dataset, 'fold_0', 'seed_'+str(seed), 'train_'+str(train_low)+'_'+'valid_'+str(valid_low)+'_sample_'+str(sample_ratio), 'fold_0', 'seed_'+str(seed), model_file)

checkpoint = torch.load(model_save_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

columns = ['nDCG', 'AP']
columns_indi = ['nDCG', 'AP']

_, NDCG, AP, ndcg, ap = evaluation(model, valid_loader, device, num_test=500, user_factors=user_factors, item_factors=item_factors, weight=mask)

result = pd.DataFrame([[ndcg, ap]], columns=columns).round(4)
result_indi = list(zip(*[NDCG, AP]))
result_indi = pd.DataFrame(result_indi, columns=columns_indi).round(4)

name = ''
# res_path_folder = os.path.join('./results_random', dataset, 'fold_'+str(fold), 'seed_'+str(seed), 'ratio_'+str(train_ratio)+'_'+str(sample_ratio), 'overall')
# res_path_folder_indi = os.path.join('./results_random', dataset, 'fold_'+str(fold), 'seed_'+str(seed), 'ratio_'+str(train_ratio)+'_'+str(sample_ratio), 'individual')
res_path_folder = os.path.join('./results_random', dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low)+'_sample_'+str(sample_ratio), 'fold_'+str(fold), 'seed_'+str(seed), 'overall')
res_path_folder_indi = os.path.join('./results_random', dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low)+'_sample_'+str(sample_ratio), 'fold_'+str(fold), 'seed_'+str(seed), 'individual')

dir_exists(res_path_folder)
dir_exists(res_path_folder_indi)
result.to_csv(os.path.join(res_path_folder, name+'valid.csv'), index=False)
result_indi.to_csv(os.path.join(res_path_folder_indi,  name+'valid.csv'), index=False)

# test for test

_, NDCG, AP, ndcg, ap = evaluation(model, test_loader, device, num_test=500, user_factors=user_factors, item_factors=item_factors, weight=mask)

result = pd.DataFrame([[ndcg, ap]], columns=columns).round(4)
result_indi = list(zip(*[NDCG, AP]))
result_indi = pd.DataFrame(result_indi, columns=columns_indi).round(4)

name = ''
# res_path_folder = os.path.join('./results_random', dataset, 'fold_'+str(fold), 'seed_'+str(seed), 'ratio_'+str(train_ratio)+'_'+str(sample_ratio), 'overall')
# res_path_folder_indi = os.path.join('./results_random', dataset, 'fold_'+str(fold), 'seed_'+str(seed), 'ratio_'+str(train_ratio)+'_'+str(sample_ratio), 'individual')
res_path_folder = os.path.join('./results_random', dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low)+'_sample_'+str(sample_ratio), 'fold_'+str(fold), 'seed_'+str(seed), 'overall')
res_path_folder_indi = os.path.join('./results_random', dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low)+'_sample_'+str(sample_ratio), 'fold_'+str(fold), 'seed_'+str(seed), 'individual')

result.to_csv(os.path.join(res_path_folder, name+'test.csv'), index=False)
result_indi.to_csv(os.path.join(res_path_folder_indi, name+'test.csv'), index=False)
