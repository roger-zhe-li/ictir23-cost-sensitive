import os
import wget
import argparse
import sys
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(dir_path)
from util import parser
import pandas as pd 
import numpy as np
import json

args = parser.parse_args()

dataset = args.dataset
ds_path = os.path.join(dir_path, 'dataset')

print(dataset)
print(ds_path)



def parseData(fname):
  for l in open(fname):
    yield eval(l)

def load_data(data_path):
	data = list(parseData(data_path))
	df = pd.DataFrame.from_records(data)

	data_pd = df[['user/profileName', 'beer/beerId', 'review/overall', 'review/text']]
	data_pd = data_pd.rename(index=int, columns={'user/profileName': 'user',
												'beer/beerId': 'item',
												'review/overall': 'rating',
												'review/text': 'review_text'})
	return data_pd

def get_unique_id(data_pd: pd.DataFrame, column: str) -> (dict, pd.DataFrame):
    """
    :param data_pd: pd.DataFrame
    :param column
    :return: dict: {value: id}
    """
    new_column = '{}_id'.format(column)
    assert new_column not in data_pd.columns
    temp = data_pd.loc[:, [column]].drop_duplicates().reset_index(drop=True)
    temp[new_column] = temp.index
    temp.index = temp[column]
    del temp[column]
    # data_pd.merge()
    data_pd = pd.merge(left=data_pd,
                       right=temp,
                       left_on=column,
                       right_index=True,
                       how='left')

    return temp[new_column].to_dict(), data_pd


def get_put_idx_value(data, user_size, item_size):
    user_id = data['user_id']
    item_id = data['item_id']
    rating = data['rating']
    month = data['month']
    rated_items = data['rated_item']
    x_length = user_size + 3 * item_size + 1

    x = np.zeros(x_length)

    # user_id, item_id one-hot
    x_idx = [user_id, user_id + item_id]
    x_value = [1, 1]

    if len(rated_items):
        idx_temp = user_size + item_size
        x_idx += [i + idx_temp for i in rated_items]
        x_value += [1 / len(rated_items)] * len(rated_items)

    idx_temp = user_size + item_size * 2 + 1
    x_idx.append(idx_temp)
    x_value.append(month)

    # last rated movie
    if len(rated_items):
        x_idx.append(rated_items[-1])
        x_value.append(1)

    data['put_idx'] = x_idx
    data['put_value'] = x_value

    return data

df = pd.read_csv(os.path.join(ds_path, dataset, 'beer_ratings.csv'), header=0, index_col=False)


# print(df.groupby('user').filter(lambda x : (x['user'].count() >= 5) and (x['user'].count() <= 100)).shape)

train_low = [3, 4, 5, 10]
valid_low = [1, 2, 3, 4, 5]
eps = 1e-3

for train in train_low:
    for valid in valid_low:
        n_low = train + 2 * valid
        min_freq = min(df.groupby('user')['user'].count().values)
        print(min_freq)
        if min_freq <= n_low:
            df_data = df.groupby('user').filter(lambda x : (x['user'].count() >= n_low).any())
        else:
            df_data = pd.DataFrame(columns=df.columns)
            ratio = n_low / min_freq + eps
            for label, group_data in df.groupby('user'):
                df_data_user = group_data.sample(frac=ratio, random_state=8964)
                df_data = df_data.append(df_data_user, ignore_index=True)

        df_train_valid = pd.DataFrame(columns=df_data.columns)
        for label, group_data_new in df_data.groupby('user'):
            item_num = group_data_new.shape[0]
            sample_num = item_num if item_num < 200 else 200
            df_user = group_data_new.sample(n=sample_num, random_state=8964)
            df_train_valid = df_train_valid.append(df_user, ignore_index=True)
        n_user = df_train_valid.user.nunique()
        n_item = df_train_valid.item.nunique()
        user_ids, df_train_valid = get_unique_id(df_train_valid, 'user')
        item_ids, df_train_valid = get_unique_id(df_train_valid, 'item')
        file_path = os.path.join(ds_path, dataset, dataset+'_train_'+str(train)+'_valid_'+str(valid)+'_raw.csv')
        df_train_valid.to_csv(file_path, index=False)
        print(f'Finished: train {train}, val {valid}')




