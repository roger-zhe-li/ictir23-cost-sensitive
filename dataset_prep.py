import torch
import pandas as pd
import numpy as np
import lenskit.crossfold as xf
import os
import sys
import argparse
import json
from tqdm import tqdm
import time

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

parser = argparse.ArgumentParser(description='Parameters for analyzing Amazon dataset')

parser.add_argument('--dataset', nargs='?', default='Musical_Instruments',
	help='dataset for analysis')
parser.add_argument('--ds_path', nargs='?', default='./dataset/',
	help='Input dataset path.')

args = parser.parse_args()

dataset = args.dataset
ds_path = args.ds_path

n_train = [3, 4, 5, 10]
n_valid = [1, 2, 3, 4, 5]

seeds = [1]
eps = 1e-3

for train_low in n_train:
	for valid_low in n_valid:
		data_path = os.path.join(ds_path, dataset, dataset+'_train_'+str(train_low)+'_valid_'+str(valid_low)+'_raw.csv')
		df = pd.read_csv(data_path, header=0, index_col=False)
		n_user = df.user_id.nunique()
		n_item = df.item_id.nunique()

		dataset_meta_info = {'dataset_size': len(df),
		                     'user_size': n_user,
		                     'item_size': n_item
		                     } 
		data_split_path = os.path.join(ds_path, dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low))
		os.makedirs(data_split_path, exist_ok=True)               
		with open(os.path.join(data_split_path, 'dataset_meta_info.json'), 'w') as f:
			json.dump(dataset_meta_info, f)

		for j in tqdm(range(len(seeds))):
			print(j)
			print(time.time())
			sample_frac = train_low / (train_low + 2 * valid_low) + eps
			for i, tp in enumerate(xf.partition_users(df, partitions=1, method=xf.SampleFrac(sample_frac), rng_spec=seeds[j])):
				save_path = os.path.join(data_split_path, 'fold_'+str(j))
				print(time.time())
				if not os.path.exists(save_path):
					os.makedirs(save_path)
				train = tp.test
				test = tp.train
				print(time.time())
				for k, td in enumerate(xf.partition_users(test, partitions=1, method=xf.SampleFrac(0.5), rng_spec=seeds[j])):
					valid = td.train
					test = td.test
				print(time.time())
			train.to_csv(os.path.join(save_path, 'train.csv'), index=False)
			valid.to_csv(os.path.join(save_path, 'valid.csv'), index=False)
			test.to_csv(os.path.join(save_path, 'test.csv'), index=False)








