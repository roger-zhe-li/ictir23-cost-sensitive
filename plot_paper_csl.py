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
# dataset = args.dataset
fold = args.fold 
dataset_path = args.ds_path

NORM = ['quantile']
DISTRIBUTION = ['ttn']
# FREQ = ['neg', 'ori', 'pos']
FREQ = ['ori']
OPTION= [5.0, 10.0, 20.0, 40.0, 80.0]
SAMPLE_RATIO = [0.4, 1.0]
TRAIN_RATIO = [1.0]
TRAIN_LOW = [5]
VALID_LOW = [5]
# DROPOUT_RATIO = [0.1, 0.2, 0.25]

results = []
seeds = [8964]

df_new = pd.DataFrame()
for dataset in ['movielens', 'beer', 'Digital_Music', 'Musical_Instruments']:
	df = pd.read_csv(dataset+'_csl_with_sample_ratio_nov_v2.csv', header=0, index_col=None)
	df = df[(df['seed'] == 8964) & (df['sample_ratio'] == 1.0) & (df['weight'] != '25x') & (df['weight'] != '35x') & (df['weight'] != '30x') & (df['weight'] != '40x') & (df['weight'] != '60x') & (df['weight'] != '70x')]

	df['diff_nDCG'] = df['diff']
	if dataset == 'movielens':
		df['dataset'] = 'MovieLens 1M'
	elif dataset == 'beer':
		df['dataset'] = 'BeerAdvocate'
	elif dataset == 'Digital_Music':
		df['dataset'] = 'Amazon Digital Music'
	elif dataset == 'Musical_Instruments':
		df['dataset'] = 'Amazon Musical Instruments'
		
	option_mapping = {'5x': 5, '10x': 10, '20x': 20, '30x': 30, '40x': 40, '50x': 50, '60x': 60, '70x': 70, '80x': 80}
	df['weight'].replace(option_mapping, inplace=True)
	df['diff_valid'] = df['exp_valid'] - df['base_valid']
	df['contrast'] = df['weight']
	df['relative nDCG difference (%)'] = 100 * ((df['exp_test'] - df['base_test']) / df['base_test'])
	
	df_new = df_new.append(df, ignore_index=True)
print(df_new.head())
df_new['ecdf(nDCG) in FM'] = df_new['qbase_test']
df_new['weight_type'] = 'util'


df_new['dataset'] = pd.Categorical(df_new['dataset'], 
                             ordered=True,
                             categories=['MovieLens 1M', 'BeerAdvocate', 'Amazon Digital Music', 'Amazon Musical Instruments'])
# df_new = df_new[df_new['dataset'] == 'MovieLens 1M']
palette=('#a50026','#d73027','#f46d43','#fdae61','#fee090','#abd9e9','#74add1','#4575b4','#313695')
palette=('#a50026','#d73027','#74add1','#4575b4','#313695')

df_new.to_csv('csl_ndcg_all_roger.csv', index=None)


# # df = df[(df['nvalid'] >= 12.5) & (df['ntest']>= 12.5)]
def facet_pages(df, column):
	for label, group_data in df.groupby(column):
		# print(group_data.head(10))
		yield (
					ggplot(group_data, aes(x='ecdf(nDCG)', y='relative nDCG difference (%)', fill='factor(contrast)', color='factor(contrast)'))
					+ geom_smooth(method='loess', size=0.5, se=True, alpha=.2)
					# + geom_smooth(alpha=.3, size=0, span=.5, se=True)
					+ facet_wrap('dataset', scales='free', nrow=1)
					+ scale_x_continuous(breaks=(0, .2, .4, .6, .8, 1))
					# + ggtitle(dataset + ': test_v1_' + str(label))
					# + scale_color_brewer(palette='YlOrRd')
					# + ggtitle(dataset + ': valid_v1')
					+ guides(color=guide_legend(nrow=1, byrow=False))
					+ theme_bw()
					+ scale_color_manual(values=palette)
					+ scale_fill_manual(values=palette)
					+ labs(color='contrast', fill='contrast') 
					+ geom_hline(yintercept=0)
					+ theme(
							
							# legend_box='horizontal',
							# legend_position='bottom',
							axis_text_y=element_text(angle=90),
							figure_size=(16, 3),
							subplots_adjust={'wspace':0.1},
							legend_box_spacing=0.25,
							legend_title=element_text(size=7), 
    						legend_text=element_text(size=7),
    						# plot_margin=0
    						plot_margin=0.1,
    						legend_margin=-1,
    						legend_position=(.5, -.1), 
    						legend_direction='horizontal', 
    						# legend_title=element_blank()
    						) 

							)   
					
					
# save_as_pdf_pages(facet_pages(df_new, 'seed'), 'csl_test_v6.pdf')
