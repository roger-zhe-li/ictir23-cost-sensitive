import copy
import json
import math
import os
import random

import msgpack
from tqdm import tqdm
import lmdb
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn
from torch.utils.data import DataLoader, Dataset
import logging
# from util import data_split_pandas
from util.npl_util import WordVector
import torch.nn.utils.rnn as rnn_utils
import torchsnooper
import scipy.stats

np.set_printoptions(suppress=True)
logger = logging.getLogger('RATE.DeepCoNN.train_test')

class Flatten(nn.Module):
	"""
	squeeze layer for Sequential structure
	"""
	def forward(self, x):
		return x.squeeze()


class UnFlatten(nn.Module):
	def forward(self, x):
		return x.view(x.size()[0], x.size()[1], 1, 1)


def collate_fn(data):
	data.sort(key=lambda x: len(x), reverse=True)
	data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
	return data


def get_train_pointwise(df_train, num_ng, n_user, n_item, train_ratio, seed):

	users, items, rels = [], [], []
	all_items = set(np.arange(n_item))
	random.seed(seed)

	for user_id in range(n_user):
		pos_items_train = df_train.loc[df_train.user_id == user_id].item_id.tolist()
		# num_valid = num_test - len(pos_items_valid)]
		num_sampled = int(len(pos_items_train) * train_ratio)

		pos_items = random.sample(pos_items_train, num_sampled)
		neg_pool = list(all_items - set(pos_items_train))
		for i in range(len(pos_items)):
			users.append(int(user_id))
			items.append(int(pos_items[i]))
			rels.append(float(1))

			neg_items = list(np.random.choice(neg_pool, num_ng, replace=False))
			for j in range(num_ng):
				users.append(int(user_id))
				items.append(int(neg_items[j]))
				rels.append(float(0))

	return users, items, rels


def get_valid(df_train, df_valid, n_user, n_item, num_test, sample_ratio, seed):

	users, items, rels = [], [], []
	all_items = set(np.arange(n_item))
	random.seed(seed)

	for user_id in range(n_user):
		pos_items_valid = df_valid.loc[df_valid.user_id == user_id].item_id.tolist()
		pos_items_train = df_train.loc[df_train.user_id == user_id].item_id.tolist()
		# num_valid = num_test - len(pos_items_valid)]
		num_sampled = int(len(pos_items_valid) * sample_ratio)
		num_valid = num_test - num_sampled

		neg_pool = list(all_items - set(pos_items_valid) - set(pos_items_train))
		neg_items = list(np.random.choice(neg_pool, num_valid, replace=False))

		neg_i = random.sample(pos_items_valid, num_sampled) + neg_items

		rel = [1] * num_sampled + [0] * num_valid

		for i in range(num_test):
			users.append(user_id)
			items.append(neg_i[i])
			rels.append(float(rel[i]))

		# users.append(user_id)
		# items.append(neg_i)
		# rels.append(rel)

	return users, items, rels


def get_test(df_train, df_valid, df_test, n_user, n_item, num_test, sample_ratio, seed):

	users, items, rels = [], [], []
	all_items = set(np.arange(n_item))
	random.seed(seed)

	for user_id in range(n_user):
		pos_items_valid = df_valid.loc[df_valid.user_id == user_id].item_id.tolist()
		pos_items_train = df_train.loc[df_train.user_id == user_id].item_id.tolist()
		pos_items_test = df_test.loc[df_test.user_id == user_id].item_id.tolist()

		num_sampled = int(len(pos_items_test) * sample_ratio)
		num_ng_test = num_test - num_sampled

		neg_pool = list(all_items - set(pos_items_valid) - set(pos_items_train) - set(pos_items_test))
		neg_items = list(np.random.choice(neg_pool, num_ng_test, replace=False))
		neg_i = random.sample(pos_items_test, num_sampled) + neg_items

		rel = [1] * num_sampled + [0] * num_ng_test

		for i in range(num_test):
			users.append(user_id)
			items.append(neg_i[i])
			rels.append(float(rel[i]))

		# users.append(user_id)
		# items.append(neg_i)
		# rels.append(rel)

	return users, items, rels



class DeepCoNNDataLoader:

	def __init__(self, train_path: str, valid_path:str, test_path: str, batch_size,
				 device: torch.device, n_user, n_item, num_ng, num_test, subset, train_ratio, sample_ratio, seed, shuffle=False):
		# self._data = pd.read_csv(data_path)\
		# 				 .reset_index(drop=True) \
						 # .loc[:, ['user_id', 'item_id', 'rel']].to_numpy()
		self._df_train = pd.read_csv(train_path, header=0, index_col=False).reset_index(drop=True)
		self._df_valid = pd.read_csv(valid_path, header=0, index_col=False).reset_index(drop=True)
		self._df_test = pd.read_csv(test_path, header=0, index_col=False).reset_index(drop=True)
		if subset == 'train':
			users, items, rels = get_train_pointwise(self._df_train, num_ng, n_user, n_item, train_ratio, seed)
		elif subset == 'valid':
			users, items, rels = get_valid(self._df_train, self._df_valid, n_user, n_item, num_test, sample_ratio, seed)
		elif subset == 'test':
			users, items, rels = get_test(self._df_train, self._df_valid, self._df_test, n_user, n_item, num_test, sample_ratio, seed)
		# self._data = np.array(list(zip(users, items, rels)))
		self._data = np.stack((users, items, rels), axis=1)

		self._device = device
		self._shuffle = shuffle
		self._batch_size = batch_size
		self._index = 0
		self._index_list = list(range(self._data.shape[0]))
		if shuffle:
			random.shuffle(self._index_list)

	def __len__(self):
		return math.ceil(len(self._index_list) // self._batch_size)
		# return len(self._data)

	def __iter__(self):
		return self

	def __next__(self):
		if self._index < len(self._index_list):
			# data = self._data.loc[self._index: self._index+self._batch_size-1]
			idx_ls = self._index_list[self._index: self._index+self._batch_size]
			self._index += self._batch_size

			data = self._data[idx_ls, :]
			users = data[:, 0].tolist()
			items = data[:, 1].tolist()
			rels = data[:, 2].astype(np.float32)
			rels = torch.from_numpy(rels).to(self._device)
				
			return users, items, rels

		else:
			self._index = 0
			raise StopIteration
