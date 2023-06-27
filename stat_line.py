import numpy as np 
import pandas as pd 
from plotnine import *

file_1 = './results_overall/stat_movielens_vanilla_extended.csv'
file_2 = './results_overall/stat_Musical_Instruments_vanilla_extended.csv'
file_3 = './results_overall/stat_beer_vanilla_extended.csv'
file_4 = './results_overall/stat_Digital_Music_vanilla_extended.csv'

df_1 = pd.read_csv(file_1, header=0, index_col=False)
df_2 = pd.read_csv(file_2, header=0, index_col=False)
df_3 = pd.read_csv(file_3, header=0, index_col=False)
df_4 = pd.read_csv(file_4, header=0, index_col=False)

df_1 = df_1.drop('pearson', axis=1)
df_2 = df_2.drop('pearson', axis=1)
df_3 = df_3.drop('pearson', axis=1)
df_4 = df_4.drop('pearson', axis=1)

df_1 = pd.melt(df_1, id_vars=['th_train', 'th_valid', 'seed'], value_vars=['spearman', 'rmse'], var_name ='stat_method', value_name ='corr')
df_1['dataset'] = 'MovieLens 1M'
df_2 = pd.melt(df_2, id_vars=['th_train', 'th_valid', 'seed'], value_vars=['spearman', 'rmse'], var_name ='stat_method', value_name ='corr')
df_2['dataset'] = 'Amazon Musical Instruments'
df_3 = pd.melt(df_3, id_vars=['th_train', 'th_valid', 'seed'], value_vars=['spearman', 'rmse'], var_name ='stat_method', value_name ='corr')
df_3['dataset'] = 'beer'
df_4 = pd.melt(df_4, id_vars=['th_train', 'th_valid', 'seed'], value_vars=['spearman', 'rmse'], var_name ='stat_method', value_name ='corr')
df_4['dataset'] = 'Amazon Digital Music'

df = pd.DataFrame()
df = df.append(df_1, ignore_index=True)
df = df.append(df_2, ignore_index=True)
df['dataset'] = pd.Categorical(df['dataset'], 
                             ordered=True,
                             categories=['MovieLens 1M', 'beer', 'Amazon Musical Instruments', 'Amazon Digital Music'])

df['th_valid'] = df['th_valid'].astype(str)
df['training threshold'] = df['th_train'].astype(str)
df['validation/test threshold'] = df['th_valid'].astype(str)
df['value'] = df['corr']
order = ['3', '4', '5', '10']

df['stat_method'] = df['stat_method'].replace(['spearman', 'rmse'], ['Spearman', 'RMSE'])

df['stat_method'] = pd.Categorical(df['stat_method'], 
                             ordered=True,
                             categories=['RMSE', 'Spearman'])

print(df.head())
# palette=('#a50026','#d73027','#f46d43','#fdae61','#fee090','#abd9e9','#74add1','#4575b4','#313695')
palette=('#a50026','#d73027','#74add1','#4575b4','#313695')


p = (
	ggplot(df, aes(x='training threshold', y='value', color='validation/test threshold', group='th_valid'))
	+ geom_line()
	+ geom_point(size=1)
	+ scale_x_discrete(limits=order)
	# + scale_color_brewer(palette = "YlOrRd")
	+ scale_color_manual(values=palette)
	+ facet_wrap('~dataset+stat_method', scales='free_y')
	+ guides(color=guide_legend(nrow=1, byrow=False))
	+ theme_bw()
	+ theme(legend_box='horizontal',
			# legend_position='bottom',
			axis_text_y=element_text(angle=90),
			subplots_adjust={'wspace':0.1},
			figure_size=(8, 6),
			legend_box_spacing=0.2,
			strip_text_y=element_text(size=8),
			legend_direction='horizontal',
			legend_title=element_text(size=7), 
    		legend_text=element_text(size=7),
    		plot_margin=0.1,
    		legend_margin=-1,
    		legend_position=(.5, 0), 
    		)
			 
	)

p.save('corr.pdf')

