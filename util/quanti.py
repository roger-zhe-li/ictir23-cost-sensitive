import numpy as np 
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import beta, truncnorm
from scipy.optimize import minimize_scalar

class F:
	def __init__(self, x):
		self.x = x

	def origin(self):
		x = np.array(self.x) 
		return x

	def norm(self):
		x = np.array(self.x)
		x_min = min(x)
		if x_min == 0:
			t = [i for i in x if i != 0]
			x_min = min(t)
		x_max = max(x)
		norm_x = (x - x_min) / (x_max - x_min)
		return norm_x

	def quantile(self):
		x = np.array(self.x)
		ecdf_x = ECDF(x)
		quanti_x = ecdf_x(x)
		return quanti_x
#Note: check the max and min values in the corresponding validation results
#Gain and ndcg plot or visualization


class G:
	def __init__(self, x):
		self.x = x

	def origin(self):
		x = np.array(self.x) 
		return x

	def Beta(self, a, b):
		x = np.array(self.x)
		beta_trans = beta(a, b)
		beta_dis = beta_trans.pdf(x)
		beta_dis = beta_dis / max(beta_dis)
		return beta_dis

	# def TN(self, lower_bound, upper_bound, loc, scale):
	#   x = np.array(self.x)
	#   tn_trans = truncnorm(lower_bound, upper_bound, loc, scale)
	#   tn_dis = tn_trans.pdf(x)
	#   tn_dis = tn_dis / max(tn_dis)
	#   return tn_dis

	def TN(self, lower, upper, mu, sigma):
		x = np.array(self.x)
		shape_0, shape_1 = (lower - mu) / sigma, (upper - mu) / sigma
		transform = truncnorm(shape_0, shape_1, loc=mu, scale=sigma)
		return transform.pdf(x)


	def TTN(self, lower, upper, theta, contrast, sigma1, mu1):
		x = np.array(self.x)
		user_size = len(x)
		def f2_ratio(sigma2, theta=theta, w=contrast):
		    f2 = truncnorm(0, (1-theta)/sigma2, theta, sigma2)
		    m = f2.pdf(theta)
		    n = f2.pdf(1)
		    return (m/n-w)**2
		sigma2 = minimize_scalar(f2_ratio, bounds=(1.0/user_size, 1), method='bounded').x
		shape_0, shape_1 = (theta - theta) / sigma2, (upper - theta) / sigma2
		transform = truncnorm(shape_0, shape_1, loc=theta, scale=sigma2)
		m, n = transform.pdf(theta), transform.pdf(1)

		shape_2, shape_3 = (lower - theta + mu1) / sigma1, (theta - theta + mu1) / sigma1
		transform_y = truncnorm(shape_2, shape_3, loc=theta-mu1, scale=sigma1)
		m1, n1 = transform_y.cdf(0), transform_y.cdf(theta)
		return [(transform_y.cdf(i) - m1) / (n1 - m1) * (m - n) + n if i <= theta else transform.pdf(i) for i in x]



class H:
	def __init__(self, x):
		self.x = x

	def origin(self):
		x = np.array(self.x)
		freq = [1 if i != 0 else 0 for i in x]
		return np.array(freq)

	def freq(self):
		x = np.array(self.x)
		return np.log1p(x) / np.log1p(max(x))

	def freq_reciprocal(self):
		x = np.array(self.x)
		t = [i for i in x if i != 0]
		min_x = min(t)
		freq = [np.log1p(min_x) / np.log1p(i) if i != 0 else 0 for i in x]
		return np.array(freq)







