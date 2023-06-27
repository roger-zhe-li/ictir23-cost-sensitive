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
np.set_printoptions(precision=4)
dir_path = os.path.dirname((os.path.abspath(os.path.join(os.path.realpath(__file__), './.'))))
args = parse_args()  



# general parameters
dataset = args.dataset
fold = args.fold 
dataset_path = args.ds_path

NORM = ['quantile']
DISTRIBUTION = ['tn0', 'tn50', 'tn100']
# FREQ = ['neg', 'ori', 'pos']
FREQ = ['ori']
OPTION= [0, 1, 2, 3, 4]
SAMPLE_RATIO = [1.0]
TRAIN_RATIO = [1.0]
TRAIN_LOW = [3, 4, 5, 10]
VALID_LOW = [1, 2, 3, 4, 5]

results = []
seeds = [8964]

df = pd.DataFrame()
for train_low in TRAIN_LOW:
	for valid_low in VALID_LOW:
		df_train = pd.read_csv(os.path.join(dataset_path, dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low), 'fold_0', 'train.csv'), header=0, index_col=False)
		df_valid = pd.read_csv(os.path.join(dataset_path, dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low), 'fold_0', 'valid.csv'), header=0, index_col=False)
		df_test = pd.read_csv(os.path.join(dataset_path, dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low), 'fold_0', 'test.csv'), header=0, index_col=False)

		freq_train = df_train.groupby('user_id')['user_id'].count().tolist()
		freq_valid = df_valid.groupby('user_id')['user_id'].count().tolist()
		freq_test = df_test.groupby('user_id')['user_id'].count().tolist()

		for seed in seeds:
			for sample_ratio in SAMPLE_RATIO:
				df_valid = pd.read_csv(os.path.join(dir_path, 'results_random', dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low)+'_'+'sample_'+str(sample_ratio), 'fold_'+str(fold), 'seed_'+str(seed), 'individual', 'valid.csv'), header=0, index_col=False)	
				df_test = pd.read_csv(os.path.join(dir_path, 'results_random', dataset, 'train_'+str(train_low)+'_'+'valid_'+str(valid_low)+'_'+'sample_'+str(sample_ratio), 'fold_'+str(fold), 'seed_'+str(seed), 'individual', 'test.csv'), header=0, index_col=False)

				df_valid['nDCG_test'] = df_test['nDCG']
				df_valid['AP_test'] = df_test['AP']
				df_valid['seed'] = str(seed)
				df_valid['sample_ratio'] = str(sample_ratio)
				df_valid['valid_qt_nDCG'] = F(df_valid['nDCG'].tolist()).quantile()
				df_valid['valid_qt_AP'] = F(df_valid['AP'].tolist()).quantile()
				df_valid['test_qt_nDCG'] = F(df_test['nDCG'].tolist()).quantile()
				df_valid['test_qt_AP'] = F(df_test['AP'].tolist()).quantile()
				df_valid['freq_train'] = freq_train
				df_valid['freq_valid'] = freq_valid
				df_valid['freq_test'] = freq_test
				df_valid['th_train'] = str(train_low)
				df_valid['th_valid'] = str(valid_low)

				df_valid = df_valid.rename(columns={"nDCG": "nDCG_valid", "AP": "AP_valid"})
				df = pd.concat([df, df_valid])

print(df.shape)
print(df.head())
# df = df[df['freq_valid'] >= 20]

corr = []

for label, group_data in df.groupby(['th_train', 'th_valid', 'seed', 'sample_ratio']):
	t = list(label)
	df_diff = group_data[['valid_qt_nDCG', 'test_qt_nDCG']]
	spearman = np.round(df_diff.corr(method='spearman').values[0][1], 4)
	pearson = np.round(np.square(df_diff.corr(method='pearson').values[0][1]), 4)
	rmse = np.round(((df_diff['valid_qt_nDCG'] - df_diff['test_qt_nDCG']) ** 2).mean() ** 0.5, 4)
	t += [spearman, pearson, rmse]
	corr.append(t)

df_stat = pd.DataFrame(corr, columns=[['th_train', 'th_valid', 'seed', 'sample_ratio', 'spearman', 'pearson', 'rmse']])
df_stat.to_csv('stat_'+dataset+'_vanilla_extended.csv', index=False)


def facet_pages(df, column):
	for label, group_data in df.groupby(column):
		# print(group_data.head(10))
		yield (
					ggplot(group_data, aes(x='nDCG_valid'))
					+ geom_histogram()
					+ facet_wrap('sample_ratio')
					+ ggtitle('Seed: ' + str(label))
					+ coord_cartesian(xlim=[0, 1])
					+ theme(axis_text_x=element_text(angle = 90, hjust = 1))
				)

# save_as_pdf_pages(facet_pages(df, 'seed'), 'hist_'+dataset+'_train_sampled_nDCG_valid.pdf')



def facet_pages_(df, column):
	for label, group_data in df.groupby(column):
		# print(group_data.head(10))
		yield (
					ggplot(group_data, aes(x='valid_qt_nDCG', y='test_qt_nDCG'))
					+ geom_point(fill='white', size=.1, alpha=.2)
					+ geom_smooth(method='lowess', size=1)
					+ facet_wrap('sample_ratio')
					+ ggtitle('Seed: ' + label)
					+ theme(axis_text_x=element_text(angle = 90, hjust = 1))
				)

p = (
	ggplot(df, aes(x='valid_qt_nDCG', y='test_qt_nDCG'))
	+ geom_point(fill='white', size=.1, alpha=.2)
	+ geom_smooth(method='lowess', size=1)
	# + facet_grid('sample_ratio ~ seed')
	+ facet_grid('th_train ~ th_valid')
	+ theme(axis_text_x=element_text(angle = 90, hjust = 1))
	)

# fig = p.draw()
# # points = fig.axes[0].collections[0]
# for i in range(len(fig.axes)):
# 	points = fig.axes[i].collections[0]
# 	points.set_rasterized(True)
# fig.savefig('corr_valid_vs_test_'+dataset+'_train_sampled_nDCG_extended.pdf')

# save_as_pdf_pages(facet_pages_(df, 'seed'), 'corr_valid_vs_test_'+dataset+'_train_sampled_nDCG.pdf')

fig.savefig('corr_valid_vs_test_'+dataset+'_train_sampled_nDCG.pdf')
df.to_csv('vanilla_'+dataset+'_with_ratio.csv', index=False)



