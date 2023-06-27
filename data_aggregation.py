import copy
import json
import math
import os
import random
import argparse
from util.parser import parse_args

import pandas as pd
import numpy as np

from plotnine import *
from util.quanti import F

np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', None)
dir_path = os.path.dirname((os.path.abspath(os.path.join(os.path.realpath(__file__), './.'))))
args = parse_args()  


# general parameters
dataset = args.dataset
fold = args.fold 
dataset_path = args.ds_path

NORM = ['quantile']
DISTRIBUTION = ['tn0']
# FREQ = ['neg', 'ori', 'pos']
FREQ = ['ori']
# OPTION= [5.0, 10.0, 20.0, 40.0, 80.0]
OPTION= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SAMPLE_RATIO = [0.4, 1.0]
TRAIN_RATIO = [1.0]
TRAIN_LOW = [5]
VALID_LOW = [5]
theta = 0.01
# DROPOUT_RATIO = [0.1, 0.2, 0.25]

# df_ind = pd.read_csv(os.path.join(dir_path, 'dataset', dataset+'.csv'), header=None, index_col=None)
# ind_selected = df_ind[0].tolist()

results = []
seeds = [1, 777, 8964]


df = pd.DataFrame()
for train_low in TRAIN_LOW:
	for valid_low in VALID_LOW:
		df_train = pd.read_csv(os.path.join(dataset_path, dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low), 'fold_0', 'train.csv'), header=0, index_col=False)
		df_valid = pd.read_csv(os.path.join(dataset_path, dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low), 'fold_0', 'valid.csv'), header=0, index_col=False)
		df_test = pd.read_csv(os.path.join(dataset_path, dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low), 'fold_0', 'test.csv'), header=0, index_col=False)

		freq_train = df_train.groupby('user_id')['user_id'].count().tolist()
		freq_valid = df_valid.groupby('user_id')['user_id'].count().tolist()
		freq_test = df_test.groupby('user_id')['user_id'].count().tolist()

		user_id = list(range(1+max(df_train['user_id'])))
		for seed in seeds:
			for sample_ratio in SAMPLE_RATIO:
				df_valid = pd.read_csv(os.path.join(dir_path, 'results_random', dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low)+'_'+'sample_'+str(sample_ratio), 'fold_'+str(fold), 'seed_'+str(seed), 'individual', 'valid.csv'), header=0, index_col=False)	
				df_test = pd.read_csv(os.path.join(dir_path, 'results_random', dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low)+'_'+'sample_'+str(sample_ratio), 'fold_'+str(fold), 'seed_'+str(seed), 'individual', 'test.csv'), header=0, index_col=False)
				df_valid['nDCG_test'] = df_test['nDCG']
				df_valid['valid_qt_nDCG'] = F(df_valid['nDCG'].tolist()).quantile()
				df_valid['test_qt_nDCG'] = F(df_test['nDCG'].tolist()).quantile()
				df_valid['freq_train'] = freq_train
				df_valid['freq_valid'] = freq_valid
				df_valid['freq_test'] = freq_test
				df_valid = df_valid.rename(columns={"nDCG": "nDCG_valid", "AP": "AP_valid"})

				for norm in NORM:
					for distribution in DISTRIBUTION:
						for freq in FREQ:
							for option in OPTION:
								# for dropout_ratio in DROPOUT_RATIO:
								test_path = os.path.join(dir_path, 'results_random', dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low)+'_'+'sample_'+str(sample_ratio), 'fold_0', 'seed_'+str(seed), 'individual', norm+'_'+distribution+'_'+freq+'_'+str(option)+'_test_nov_v2.csv')
								valid_path = os.path.join(dir_path, 'results_random', dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low)+'_'+'sample_'+str(sample_ratio), 'fold_0', 'seed_'+str(seed), 'individual', norm+'_'+distribution+'_'+freq+'_'+str(option)+'_valid_nov_v2.csv')
								# test_path = os.path.join(dir_path, 'results_random', dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low)+'_'+'sample_'+str(sample_ratio), 'fold_0', 'seed_'+str(seed), 'individual', norm+'_'+distribution+'_'+freq+'_'+str(option)+'_nov_v2.csv')
								# valid_path = os.path.join(dir_path, 'results_random', dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low)+'_'+'sample_'+str(sample_ratio), 'fold_0', 'seed_'+str(seed), 'individual', norm+'_'+distribution+'_'+freq+'_'+str(option)+'_nov_v2.csv')
								df_obj = pd.read_csv(test_path, header=0, index_col=False)
								df_valid_cut = pd.read_csv(valid_path, header=0, index_col=False)
								df_obj = df_obj.drop(['AP'], axis=1)

								df_obj['train_cut'] = norm+'_'+distribution+'_'+freq
								df_obj['seed'] = str(seed)
								df_obj['weight'] = str(option)
								df_obj['sample_ratio'] = str(sample_ratio)
								df_obj['user_id'] = user_id
								df_obj['base_valid'] = df_valid['nDCG_valid']
								df_obj['base_test'] = df_valid['nDCG_test']
								df_obj['exp_valid'] = df_valid_cut['nDCG']
								df_obj['exp_test'] = df_obj['nDCG']
								df_obj['qbase_valid'] = df_valid['valid_qt_nDCG']
								df_obj['qbase_test'] = df_valid['test_qt_nDCG']
								df_obj['qexp_valid'] = F(df_valid_cut['nDCG'].tolist()).quantile()
								df_obj['qexp_test'] = F(df_obj['nDCG'].tolist()).quantile()    
								
								df_obj['ntrain'] = df_valid['freq_train']
								df_obj['nvalid'] = df_valid['freq_valid']
								df_obj['ntest'] = df_valid['freq_test']
								df_obj['diff'] = df_obj['nDCG'] - df_valid['nDCG_test']
								df_obj = df_obj.drop(['nDCG'], axis=1)
								df_obj['th_train'] = str(train_low)
								df_obj['th_valid'] = str(valid_low)
								# df_obj['dropout_ratio'] = str(dropout_ratio)

								# df_obj = df_obj[df_obj['qbase_valid'] > dropout_ratio]
								# df_obj = df_obj[df_obj['user_id'].isin(ind_selected)]
								df_obj['qbase_valid'] = F(df_obj['qbase_valid'].tolist()).quantile()
								df = pd.concat([df, df_obj])


print(df.shape)
print(df.head())

# option_mapping = {'5.0': '5x', '10.0': '10x', '20.0': '20x', '40.0': '40x', '80.0': '80x'}
option_mapping = {'0': '5x', '1': '10x', '2': '20x', '3': '40x', '4': '80x', '5': '25x', '6': '30x', '7': '35x', '8': '50x', '9': '60x', '10': '70x'}

df['weight'].replace(option_mapping, inplace=True)


diff = []
for label, group_data in df.groupby(['th_train', 'th_valid', 'train_cut', 'seed', 'weight', 'sample_ratio']):
	t = list(label)
	absdiff = (group_data['exp_test'] - group_data['base_test']).mean()
	reldiff = ((group_data['exp_test'] - group_data['base_test']) / group_data['base_test']).mean()
	t += [absdiff, reldiff]
	diff.append(t)

df_diff = pd.DataFrame(diff, columns=[['th_train', 'th_valid', 'train_cut', 'seed', 'weight', 'sample_ratio', 'absdiff', 'reldiff']])
print(df.shape)
print(df.head())

df.to_csv(dataset+'_csl_with_sample_ratio'+'.csv', index=False)




