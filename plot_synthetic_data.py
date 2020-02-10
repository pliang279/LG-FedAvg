import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt

d = 20
trainN = 2000
testN = 1000
M = 100

def gen_data(sigma, rho):
	v = np.random.random((d,))
	mean = np.zeros((d,))
	cov = rho**2 * np.eye(d)
	all_data = []
	for m in range(M):
		r_m = np.random.multivariate_normal(mean, cov)
		u_m = v + r_m

		x_m = np.random.uniform(-1.0, 1.0, (trainN+testN,d))
		y_m = np.dot(x_m, u_m) + np.random.normal(0, sigma**2, (trainN+testN,))

		train_x_m = x_m[:trainN]
		train_y_m = y_m[:trainN]
		test_x_m = x_m[trainN:]
		test_y_m = y_m[trainN:]

		#print (train_x_m)[:10]
		#print (train_y_m)[:10]
		#print (test_x_m)[:10]
		#print (test_y_m)[:10]
		#assert False
		all_data.append((train_x_m, train_y_m, test_x_m, test_y_m))
	return all_data

def local_model(all_data):
	train_errors = []
	test_errors = []
	for (train_x_m, train_y_m, test_x_m, test_y_m) in all_data:
		u_m_hat1 = np.linalg.inv(np.dot(np.transpose(train_x_m), train_x_m))
		u_m_hat2 = np.dot(np.transpose(train_x_m), train_y_m)
		u_m_hat = np.dot(u_m_hat1, u_m_hat2)

		train_pred = np.dot(train_x_m, u_m_hat)
		test_pred = np.dot(test_x_m, u_m_hat)
		train_error = np.mean((train_pred - train_y_m)**2)
		test_error = np.mean((test_pred - test_y_m)**2)

		#print (train_pred)[:10]
		#print (train_y_m)[:10]
		#print (test_pred)[:10]
		#print (test_y_m)[:10]
		#assert False

		train_errors.append(train_error)
		test_errors.append(test_error)
	return np.mean(train_errors), np.mean(test_errors)

def global_model(all_data):
	all_train_x = []
	all_train_y = []
	all_test_x = []
	all_test_y = []
	for (train_x_m, train_y_m, test_x_m, test_y_m) in all_data:
		all_train_x.append(train_x_m)
		all_train_y.append(train_y_m)
		all_test_x.append(test_x_m)
		all_test_y.append(test_y_m)
	all_train_x = np.concatenate(all_train_x, axis=0)
	all_train_y = np.concatenate(all_train_y, axis=0)
	all_test_x = np.concatenate(all_test_x, axis=0)
	all_test_y = np.concatenate(all_test_y, axis=0)

	v_hat1 = np.linalg.inv(np.dot(np.transpose(all_train_x), all_train_x))
	v_hat2 = np.dot(np.transpose(all_train_x), all_train_y)
	v_hat = np.dot(v_hat1, v_hat2)
	train_pred = np.dot(all_train_x, v_hat)
	test_pred = np.dot(all_test_x, v_hat)
	train_error = np.mean((train_pred - all_train_y)**2)
	test_error = np.mean((test_pred - all_test_y)**2)
	return train_error, test_error

def local_global(all_data, alpha):
	all_train_x = []
	all_train_y = []
	all_test_x = []
	all_test_y = []
	for (train_x_m, train_y_m, test_x_m, test_y_m) in all_data:
		all_train_x.append(train_x_m)
		all_train_y.append(train_y_m)
		all_test_x.append(test_x_m)
		all_test_y.append(test_y_m)
	all_train_x = np.concatenate(all_train_x, axis=0)
	all_train_y = np.concatenate(all_train_y, axis=0)
	all_test_x = np.concatenate(all_test_x, axis=0)
	all_test_y = np.concatenate(all_test_y, axis=0)

	v_hat1 = np.linalg.inv(np.dot(np.transpose(all_train_x), all_train_x))
	v_hat2 = np.dot(np.transpose(all_train_x), all_train_y)
	v_hat = np.dot(v_hat1, v_hat2)

	train_errors = []
	test_errors = []
	for (train_x_m, train_y_m, test_x_m, test_y_m) in all_data:
		u_m_hat1 = np.linalg.inv(np.dot(np.transpose(train_x_m), train_x_m))
		u_m_hat2 = np.dot(np.transpose(train_x_m), train_y_m)
		u_m_hat = np.dot(u_m_hat1, u_m_hat2)

		ensemble = alpha*u_m_hat + (1.0-alpha)*v_hat
		train_pred = np.dot(train_x_m, ensemble)
		test_pred = np.dot(test_x_m, ensemble)
		train_error = np.mean((train_pred - train_y_m)**2)
		test_error = np.mean((test_pred - test_y_m)**2)

		train_errors.append(train_error)
		test_errors.append(test_error)
	return np.mean(train_errors), np.mean(test_errors)

# local better
# rho = 0.1
# sigma = 1.5
# plt.ylim(5.07, 5.14)

# local too good
# rho = 0.5
# sigma = 1.5
# plt.ylim(5.0, 8.0)

# global better
# rho = 0.06
# sigma = 1.5
# plt.ylim(5.06, 5.12)

# global too good
# rho = 0.02
# sigma = 1.5
# plt.ylim(5.04, 5.14)

sigmas = [i/10.0 for i in range(11)]
rhos = [i/10.0 for i in range(11)]
alphas = [i/10.0 for i in range(11)]
ls = []
gs = []
lgs = []
rho = 0.5
sigma = 1.5
all_data = gen_data(sigma, rho)
for alpha in alphas:
	local_train_err, local_test_err = local_model(all_data)
	global_train_err, global_test_err = global_model(all_data)
	lg_train_err, lg_test_err = local_global(all_data, alpha)
	ls.append(local_test_err)
	gs.append(global_test_err)
	lgs.append(lg_test_err)

fig, ax = plt.subplots()
# We change the fontsize of minor ticks label 
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)

plt.plot(alphas, ls, label='Local only', linewidth=3.0)
plt.plot(alphas, gs, label='FedAvg', linewidth=3.0)
plt.plot(alphas,lgs, label='LG-FedAvg', linewidth=3.0)
plt.legend(fontsize=20)
#plt.xlabel('\alpha', fontsize=18)
plt.ylim(5.0, 8.0)
#plt.yticks([])
#plt.ylabel('average test error', fontsize=18)
plt.show()
assert False

#for sigma in sigmas:
#for rho in rhos:
for alpha in alphas:
	rho = 0.5
	sigma = 0.5
	all_data = gen_data(sigma, rho)
	local_train_err, local_test_err = local_model(all_data)
	global_train_err, global_test_err = global_model(all_data)

	# alpha1 = (M-1)/float(M) * rho**2 + float(d)/(M*trainN) * sigma**2
	# alpha2 = (M-1)/float(M) * rho**2 + float((M+1)*d)/(M*trainN) * sigma**2
	# alpha = alpha1 / float(alpha2)
	print 'alpha', alpha

	lg_train_err, lg_test_err = local_global(all_data, alpha)
	print sigma, local_test_err, global_test_err, lg_test_err
	ls.append(local_test_err)
	gs.append(global_test_err)
	lgs.append(lg_test_err)
#assert False

plt.plot(ls, label='l')
plt.plot(gs, label='g')
plt.plot(lgs, label='lg')
plt.legend()
plt.show()



