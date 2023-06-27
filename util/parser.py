import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='hyperparameters')
	parser.add_argument('--save_path', 
						nargs='?', 
						default='./saved_models/',
                        help='Save data path.')
	parser.add_argument('--res_path', 
						nargs='?', 
						default='./results_ndcg/',
						help='Results path.')
	parser.add_argument('--feature_path', 
						nargs='?', 
						default='./features',
	                    help='Feature path')
	parser.add_argument('--learner', 
						nargs='?', 
						default='adam', 
						help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
	parser.add_argument('--weight_decay', 
						type=float, 
						default=1e-3, 
						help='L2 regularization')
	parser.add_argument('--mode', 
						nargs='?', 
						choices=['pairwise', 'pointwise'], 
						default='pointwise', 
						help='ranking architecture')
	parser.add_argument('--weight', 
						type=float, 
						default=0.0, 
						help='L2 regularization')
	parser.add_argument('--batch_size', 
						type=int, 
						default=512, 
						help='input batch size for training (default: 512)')
	parser.add_argument('--no-cuda', 
						action='store_true', 
						default=False,
						help='enables CUDA training') 
	parser.add_argument('--num_ng', 
						type=int, 
						default=4, 
						help='negative sampling size for training')
	parser.add_argument('--dataset', 
						nargs='?', 
						default='Musical_Instruments',
						help='dataset for analysis')
	parser.add_argument('--ds_path', 
						nargs='?', 
						default='./dataset/',
						help='Input dataset path.')	
	parser.add_argument('--fold', 
						type=int, 
						default=0, 
						help='fold for cross validation')
	parser.add_argument('--seed', 
						type=int, 
						default=8964, 
						help='random seed')	
	parser.add_argument('--num_test', 
						type=int, 
						default=1000, 
						help='total number of items per user for validation and test')	
	parser.add_argument('--fc', 
						type=int, 
						default=0,
						help='reconstruction mode: 1 for concatenate and 0 for slice')	
	parser.add_argument('--epochs', 
						type=int, 
						default=10,
						help='num of epochs')
	parser.add_argument('--latent_factor_num', 
						type=int, 
						default=4096,
						help='number of user/item low-rank representation for recommendation')	
	parser.add_argument('--layers', 
						nargs='?', 
						default=[64, 32, 16, 8],
						help='Size of each layer')
	parser.add_argument('--rdm', 
						action='store_true', 
						default=False,
						help='whether using pretrained embeddings')
	parser.add_argument('--init_range', 
						type=float, 
						default=0.01,
						help='range for initial user/item embeddings')
	parser.add_argument('--dropout', 
						type=float, 
						default=0.0,
						help='dropout ratio for training')	
	parser.add_argument('--lr', 
						type=float, 
						default=1e-7,
						help='learning rate')			
	parser.add_argument('--k', 
						type=int, default=10,
						help='cutoff for evaluation')
	parser.add_argument('--baseline', 
						nargs='?', choices=['fm', 'ncf', 'mf'], 
						default='fm', 
						help='choose the implicit model to run')
	parser.add_argument('--fm_k', 
						type=int, default=32,
						help='number of interaction latent factors for FM')
	parser.add_argument('--paradigm', 
						type=str, choices=['random', 'text', 'multi'], 
						default='text', 
						help='choose the information source')
	parser.add_argument('--tower_type', 
						type=str, default='both',
						choices=['random', 'user', 'item', 'both'],
						help='extract features for users or items') 
	parser.add_argument('--data_type', 
						type=str, default='review',
						choices=['review', 'description', 'visual'],
						help='train for reviews or item descriptions or item visual features')
	parser.add_argument('--representation_factor_num', 
						type=int, default=100,
						help='number of latent factors used for representing users/items for interaction')
	parser.add_argument('--model_type', 
						type=str, 
						choices=['prediction', 'hybrid'], 
						default='hybrid', 
						help='choose the information source')
	parser.add_argument('--user_weight',
						type=float,
						default=0.0,
						help='weight for user vector reconstruction loss')
	parser.add_argument('--item_weight',
						type=float,
						default=0.0,
						help='weight for item vector reconstruction loss')
	parser.add_argument('--pred_weight',
						type=float,
						default=1.0,
						help='weight for recommendation loss')
	parser.add_argument('--db_path',
						type=str,
						default='../db',
						help='lmdb files root folder')
	parser.add_argument('--feature_model_path', 
						type=str, 
						default='./saved_models/features/',
						help='folder for saving feature training models') 
	parser.add_argument('--rebuild', 
						action='store_true', 
						default=False,
						help='whether rebuilding the db file')
	parser.add_argument('--retrain', 
						action='store_true', 
						default=False,
						help='whether retrain the feature extraction model')
	parser.add_argument('--review_length', 
						type=int, 
						default=512,
						help='length for cutting off reviews')
	parser.add_argument('--word_vec_dim', 
						type=int, 
						default=300,
						help='dimensionality of each word vector')
	parser.add_argument('--conv_length', 
						type=int, 
						default=3,
						help='n-grams used for TextCNN')
	parser.add_argument('--epsilon', 
						type=float, 
						default=1e-4,
						help='epsilon for quantification')
	parser.add_argument('--conv_kernel_num', 
						type=int, 
						default=100,
						help='number of channels for CNN feature extraction')
	parser.add_argument('--training_type',
						nargs='?',
						default='unweighted',
						choices=['unweighted', 'weighted'],
						help='whether incorporate the individual weights for training'
						)
	parser.add_argument('--weight_combi',
						type=str,
						default='norm_b15_origin',
						help='choice of the f,g,h combination'
						)
	parser.add_argument('--ae_on', 
						action='store_true', 
						default=False,
						help='whether activate AEs for training') 
	parser.add_argument('--qcut', 
						type=int, 
						default=0,
						help='select the subset to train')
	parser.add_argument('--th_cut', 
						type=float, 
						default=50.0,
						help='select the subset to train')
	parser.add_argument('--data_incl',
						nargs='?',
						default='included',
						choices=['included', 'excluded'],
						help='whether include the indicated part of users for training'
						)
	parser.add_argument('--norm',
						type=str,
						default='quantile',
						help='method for normalization')
	parser.add_argument('--distribution',
						type=str,
						default='tn0',
						help='method for adapting distributions')
	parser.add_argument('--freq',
						type=str,
						default='ori',
						help='method for normalization')
	parser.add_argument('--sigma_type',
						type=int,
						default=1,
						help='sigma value check')
	parser.add_argument('--sample_ratio',
						type=float,
						default=1.0,
						help='sample ratio for validation/test')
	parser.add_argument('--train_ratio',
						type=float,
						default=1.0,
						help='sample ratio for training')
	parser.add_argument('--train_low', 
						type=int, 
						default=3,
						help='select the minimum number of training samples per user')
	parser.add_argument('--valid_low', 
						type=int, 
						default=1,
						help='select the minimum number of valid/test samples per user')
	parser.add_argument('--dropout_ratio',
						type=float,
						default=0,
						help='sigma value check')
	parser.add_argument('--theta',
						type=float,
						default=0.01,
						help='truncated normal peak value reached on this quantile value')
	parser.add_argument('--contrast',
						type=float,
						default=80,
						help='contrast of max and min')
	parser.add_argument('--sigma1',
						type=float,
						default=0.015,
						help='controls the shape of the left part')
	parser.add_argument('--mu1',
						type=float,
						default=0.05,
						help='controls the position of quantile value on the left part')
	args = parser.parse_args()

	return args