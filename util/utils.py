# -*- coding: utf-8 -*-
import numpy as np
import logging
import re
import os
import torch

util_logger = logging.getLogger('RATE.npl')

def dir_exists(path):
	if not os.path.exists(path):
		os.makedirs(path)


def save_model(epoch, model, best_result, optimizer, save_path):
	torch.save({
		'epoch': epoch + 1,
		'state_dict': model.state_dict(),
		'best_performance': best_result,
		'optimizer': optimizer.state_dict(),
		}, save_path)


def del_db(path):
	lmdb_path = path
	if os.path.exists(os.path.join(lmdb_path, 'data.mdb')):
		os.remove(os.path.join(lmdb_path, 'data.mdb'))
		os.remove(os.path.join(lmdb_path, 'lock.mdb'))


def collate_fn(data):
	data.sort(key=lambda x: len(x), reverse=True)
	data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
	return data
