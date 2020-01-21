import time, os, csv, random
import cma

import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from numpy.linalg import LinAlgError
from numpy.random import standard_normal
from sklearn.decomposition import PCA
from copy import deepcopy
import matplotlib.pyplot as plt

from nevergrad.functions import corefuncs
import tensorflow as tf


def output_data(filename, data):
    file = open(filename, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(file, delimiter='\n', quotechar=' ', quoting=csv.QUOTE_MINIMAL)

    csv_writer.writerow(data)
    file.close()


# Vanilla ES framework to estimate gradients
def es_compute_grads(x, loss_fn, sigma=0.01, pop_size=10):
    grad = 0
    for i in range(pop_size):
        noise = sigma / np.sqrt(len(x)) * np.random.randn(1, len(x))
        noise = noise.reshape(x.shape)
        grad += noise * (loss_fn(x + noise) - loss_fn(x - noise))
    g_hat = grad / (2 * pop_size * sigma ** 2)
    return g_hat


# Using Vanilla ES to estimate the gradient.
def es(x_init, loss_fn, lr=0.2, sigma=0.01, pop_size=10, max_samples=int(1e5)):

    x = deepcopy(x_init)
    xs, ys, ts, errors = [0], [loss_fn(x)], [0.], [0.]
    total_sample, current_iter = 0, 0
    while total_sample < max_samples:
        time_st = time.time()
        g_hat = es_compute_grads(x, loss_fn, sigma=sigma, pop_size=pop_size)
        errors.append(np.dot(2*x, g_hat)/(np.linalg.norm(2*x) * np.linalg.norm(g_hat)))
        x -= lr * g_hat
        xs.append(2*pop_size)
        ys.append(loss_fn(x))
        ts.append(time.time() - time_st)
        total_sample += 2*pop_size
        current_iter += 1
    print('es use time :%.2f sec' % np.sum(ts))
    return xs, ys, ts, errors


# Guided-ES framework to estimate gradients
def ges_compute_grads(x, loss_fn, U, k=1, pop_size=1, sigma=0.1, alpha=0.5):

    # Globalspace param
    a = sigma * np.sqrt(alpha / x.shape[0])
    # Subspace param
    c = sigma * np.sqrt((1 - alpha) / k)
    grad = 0
    for i in range(pop_size):
        if alpha > 0.5:
            noise = a * np.random.randn(1, len(x))
        else:
            noise = a * np.random.randn(1, len(x)) + c * np.random.randn(1, k) @ U.T
        noise = noise.reshape(x.shape)
        grad += noise * (loss_fn(x + noise) - loss_fn(x - noise))
    return grad / (2 * pop_size * sigma ** 2)


def ges(x_init, loss_fn, lr=0.2, sigma=0.01, k=1, pop_size=1, max_samples=int(1e5)):

    x = deepcopy(x_init)
    total_sample, current_iter = 0, 0
    U, surg_grads = None, []
    xs, ys, ts, errors = [0], [loss_fn(x)], [0.], [0.]
    while total_sample < max_samples:
        time_st = time.time()
        if current_iter < k:
            g_hat = ges_compute_grads(x, loss_fn, U, k, pop_size=pop_size, sigma=sigma, alpha=1)
            surg_grads.append(g_hat)
        else:
            U, _ = np.linalg.qr(np.array(surg_grads).T)
            g_hat = ges_compute_grads(x, loss_fn, U, k, pop_size=pop_size, sigma=sigma, alpha=0.5)
            surg_grads.pop(0)
            surg_grads.append(g_hat)
            # sg = loss_fn.compute_gradient(x, bias_coef=1., noise_coef=1.5)[0]
            # surg_grads.append(sg)
        errors.append(np.dot(2*x, g_hat)/(np.linalg.norm(2*x) * np.linalg.norm(g_hat)))
        x -= lr * g_hat
        xs.append(2*pop_size)
        ys.append(loss_fn(x))
        ts.append(time.time() - time_st)
        total_sample += pop_size*2
        current_iter += 1
    print('guided es use time :%.2f sec' % np.sum(ts))
    return xs, ys, ts, errors


# SGES framework to estimate gradients
def sges_compute_grads(x, loss_fn, U=None, k=1, pop_size=1, sigma=0.01, alpha=0.2):

    grad, global_grad, sub_grad = 0, [], []
    grad_loss, random_loss = [], []
    for i in range(pop_size):
        if random.random() < alpha:
            noise = sigma / np.sqrt(k) * np.random.randn(1, k) @ U.T
            noise = noise.reshape(x.shape)
            pos_loss, neg_loss = loss_fn(x + noise), loss_fn(x - noise)
            grad_loss.append(min(pos_loss, neg_loss))
            sub_grad.append(noise * (pos_loss - neg_loss) / sigma**2 /2)
        else:
            noise = sigma / np.sqrt(len(x)) * np.random.randn(1, len(x))
            noise = noise.reshape(x.shape)
            pos_loss, neg_loss = loss_fn(x + noise), loss_fn(x - noise)
            random_loss.append(min(pos_loss, neg_loss))
            global_grad.append(noise * (pos_loss - neg_loss) / sigma**2/2)
        grad += noise * (pos_loss - neg_loss)
    g_hat = grad / (2 * pop_size * sigma ** 2)
    global_grad = np.mean(np.asarray(global_grad), axis=0)
    sub_grad = np.mean(np.asarray(sub_grad), axis=0)

    mean_grad_loss = 10000 if grad_loss is None else np.mean(np.asarray(grad_loss))
    mean_random_loss = 10000 if random_loss is None else np.mean(np.asarray(random_loss))

    return g_hat, mean_grad_loss, mean_random_loss, global_grad, sub_grad


# Using previous k estimated gradients as the surrogate gradient, and Use SGES to optimize.
# k: the gradient subspace dimension.
# alpha: tradeoff between the gradient subspace and entire space(corresponding orthogonal complement)
def sges(x_init, loss_fn, lr=0.2, sigma=0.01, k=1, pop_size=1, max_samples=int(1e5), auto_alpha=True):

    x = deepcopy(x_init)
    alpha, U, surg_grads = 0.5, None, []
    history_info = []
    total_sample, current_iter = 0, 0
    xs, ys, ts, errors  = [0], [loss_fn(x)], [0], [0.]
    while total_sample < max_samples:
        time_st = time.time()
        if current_iter < k:
            g_hat, *_ = sges_compute_grads(x, loss_fn, U, k, pop_size=pop_size, sigma=sigma, alpha=0)
            surg_grads.append(g_hat)
        else:
            U, _ = np.linalg.qr(np.array(surg_grads).T)
            g_hat, mean_grad_loss, mean_random_loss, global_grad, sub_grad = sges_compute_grads(x, loss_fn, U, k, pop_size=pop_size, sigma=sigma, alpha=alpha)
            if auto_alpha:
                alpha = alpha * 1.005 if mean_grad_loss < mean_random_loss else alpha / 1.005
                alpha = 0.7 if alpha > 0.7 else alpha
                alpha = 0.3 if alpha < 0.3 else alpha
            surg_grads.pop(0)
            surg_grads.append(g_hat)
            # surg_grads.append()
        errors.append(np.dot(2*x, g_hat)/(np.linalg.norm(2*x) * np.linalg.norm(g_hat)))
        x -= lr * g_hat
        xs.append(2*pop_size)
        ys.append(loss_fn(x))
        ts.append(time.time() - time_st)
        total_sample += 2*pop_size
        current_iter += 1
    print('sges use time :%.2f sec' % np.sum(ts))
    return xs, ys, ts, errors


def cma_es(x_init, loss_fn, sigma=0.01, pop_size=20, max_samples=int(1e5)):

    x = deepcopy(x_init)
    cma_opt = cma.CMAOptions()
    opt_defaults = cma_opt.defaults()
    cmaes = cma.CMAEvolutionStrategy(x, sigma / np.sqrt(len(x)), {'popsize': pop_size, 'seed': 2020})
    total_sample, current_iter = 0, 0
    xs, ys, ts, errors = [0], [loss_fn(x)], [0], [0.]
    while total_sample < max_samples:
        time_st = time.time()
        population = cmaes.ask()
        fitness = []
        for k in range(pop_size):
            x = np.array(population[k]).reshape(len(x))
            fitness.append(loss_fn(x))
        cmaes.tell(population, np.array(fitness))
        xs.append(pop_size)
        ys.append(loss_fn(np.array(cmaes.mean).reshape(len(x))))
        ts.append(time.time() - time_st)
        errors.append(0.)
        total_sample += pop_size
        current_iter += 1
    print('cma-es use time :%.2f sec' % np.sum(ts))
    return xs, ys, ts, errors


def asebo_compute_grads(x, loss_fn, U, alpha, sigma, min_samples=10, threshold=0.995, default_pop_size=50):
    pca_fail = False
    dims = len(x)
    try:
        pca = PCA()
        pca_fit = pca.fit(U)
        var_exp = pca_fit.explained_variance_ratio_
        var_exp = np.cumsum(var_exp)
        n_samples = np.argmax(var_exp > threshold) + 1
        if n_samples < min_samples:
            n_samples = min_samples
        # n_samples = params['num_sensings']
        U = pca_fit.components_[:n_samples]
        UUT = np.matmul(U.T, U)
        U_ort = pca_fit.components_[n_samples:]
        UUT_ort = np.matmul(U_ort.T, U_ort)
    except LinAlgError:
        UUT = np.zeros([dims, dims])
        n_samples = default_pop_size
        pca_fail = True

    np.random.seed(None)
    cov = (alpha / dims) * np.eye(dims) + ((1 - alpha) / n_samples) * UUT
    # cov *= params['sigma']
    mu = np.repeat(0, dims)
    # A = np.random.multivariate_normal(mu, cov, n_samples)
    A = np.zeros((n_samples, dims))
    try:
        l = cholesky(cov, check_finite=False, overwrite_a=True)
        for i in range(n_samples):
            try:
                A[i] = np.zeros(dims) + l.dot(standard_normal(dims))
            except LinAlgError:
                A[i] = np.random.randn(dims)
    except LinAlgError:
        for i in range(n_samples):
            A[i] = np.random.randn(dims)
    A /= np.linalg.norm(A, axis=-1)[:, np.newaxis]
    A *= sigma

    m = []
    for i in range(n_samples):
        m.append(loss_fn(x + A[i]) - loss_fn(x - A[i]))
    g = np.zeros(dims)
    for i in range(n_samples):
        eps = A[i, :]
        g += eps * m[i]
    g /= (2 * (sigma ** 2) * n_samples)

    if not pca_fail:
        # params['alpha'] = np.linalg.norm(np.dot(g, UUT_ort))/np.linalg.norm(np.dot(g, UUT))
        alpha = np.linalg.norm(np.dot(g, UUT)) / np.linalg.norm(np.dot(g, UUT_ort))
    else:
        alpha = 1.
    return g, n_samples, alpha


def asebo(x_init, loss_fn, lr=0.2, sigma=0.01, k=1, decay=0.95, pop_size=1, max_samples=int(1e5)):

    x = deepcopy(x_init)
    alpha, U = 0., 0
    total_sample, current_iter = 0, 0
    xs, ys, ts, errors = [0], [loss_fn(x)], [0.], [0.]
    while total_sample < max_samples:
        time_st = time.time()
        if current_iter < k:
            g_hat = es_compute_grads(x, loss_fn, pop_size=pop_size, sigma=sigma)
            n_sample = pop_size
        else:
            g_hat, n_sample, alpha = asebo_compute_grads(x, loss_fn, U, sigma=sigma, alpha=alpha, default_pop_size=pop_size)
        errors.append(np.dot(2*x, g_hat)/(np.linalg.norm(2*x) * np.linalg.norm(g_hat)))
        x -= lr * g_hat
        if current_iter == 0:
            U = np.dot(g_hat[:, None], g_hat[None, :])
        else:
            U = decay * U + (1-decay) * np.dot(g_hat[:, None], g_hat[None, :])
        total_sample += n_sample * 2
        current_iter += 1
        xs.append(n_sample*2)
        ys.append(loss_fn(x))
        ts.append(time.time() - time_st)
    print('asebo use time :%.2f sec' % np.sum(ts))
    return xs, ys, ts, errors


# Generate the problem  x: dimension n
def generate_problem(n, func_name, use_tf=False):
    if use_tf:
        if func_name == 'Sphere':
            loss_fn = Sphere(n)
        elif func_name == 'Rosenbrock':
            loss_fn = Rosenbrock(n)
        elif func_name == 'Rastrigin':
            loss_fn = Rastrigin(n)
        elif func_name == 'Lunacek':
            loss_fn = Lunacek(n)
        else:
            raise NotImplemented('%s not implemented' % func_name)
    else:
        if func_name == 'Sphere':
            loss_fn = lambda x: np.linalg.norm(x) ** 2
        elif func_name == 'Rosenbrock':
            loss_fn = corefuncs.rosenbrock
        elif func_name == 'Rastrigin':
            loss_fn = corefuncs.rastrigin
        elif func_name == 'Lunacek':
            loss_fn = lunacek
        else:
            raise NotImplemented('%s not implemented' % func_name)
    x = np.random.randn(n)

    return loss_fn, x


class Function(object):
    def __init__(self, dims, sess=None, seed=2020):
        self.dims = dims
        self.X = tf.placeholder(tf.float32, [None, dims])
        self.sess = sess or tf.Session()
        self.Y = self._build_fn()
        self.g = tf.gradients(self.Y, self.X)
        
        np.random.seed(seed)
        self.bias = np.random.randn(dims)
        self.bias /= np.linalg.norm(self.bias)
    
    def _build_fn(self):
        raise NotImplementedError

    def __call__(self, x):
        flatten = False if x.ndim == 2 else True
        if flatten:
             _x = x[None, :] 
        else:
            _x = x
        assert _x.shape[-1] == self.dims
        y = self.sess.run(self.Y, feed_dict={self.X: _x})    
        if flatten:
            return y[0]
        else:
            return y

    def compute_gradient(self, x, bias_coef=0., noise_coef=0.):
        assert x.ndim == 1 and x.shape[0] == self.dims
        grad = self.sess.run(self.g, feed_dict={self.X: x[None, :]})[0]
        noise = np.random.randn(self.dims)
        noise /= np.linalg.norm(noise)
        grad = grad + (self.bias * bias_coef + noise * noise_coef) * np.linalg.norm(grad, axis=-1)
        return grad

class Sphere(Function):
    def _build_fn(self, ):
        return tf.reduce_sum(tf.square(self.X), axis=-1)


def lunacek(x):
    flatten = False
    if x.ndim == 1:
        flatten = True
        x = x[None, :]
    N = x.shape[-1]
    d = 1
    s = 1 - 1/(2*np.sqrt(N + 20) - 8.2)
    mu1 = 2.5
    mu2 = -np.sqrt(abs((mu1 ** 2 - 1.0) / s))
    first_sum = np.sum(np.square(x - mu1), axis=-1)
    second_sum = np.sum(np.square(x - mu2), axis=-1)
    third_sum = N - np.sum(np.cos(2*np.pi*(x - mu1)), axis=-1)
    y = np.minimum(first_sum, N + second_sum) + 10 * third_sum
    if flatten:
        return y[0]
    else:
        return y

class Lunacek(Function):
    def _build_fn(self, ):
        N = self.dims
        d = 1
        s = 1 - 1/(2*np.sqrt(N + 20) - 8.2)
        mu1 = 2.5
        mu2 = -np.sqrt(abs((mu1 ** 2 - 1.0) / s))
        first_sum = tf.reduce_sum(tf.square(self.X - mu1), axis=-1)
        second_sum = tf.reduce_sum(tf.square(self.X - mu2), axis=-1)
        third_sum = N - tf.reduce_sum(tf.cos(2*np.pi*(self.X - mu1)), axis=-1)
        y = tf.minimum(first_sum, N + second_sum) + 10 * third_sum
        return y

class Rosenbrock(Function):
    def _build_fn(self, ):
        x_m_1 = self.X[:, :-1] - 1
        x_diff = tf.square(self.X[:, :-1]) - self.X[:, 1:]
        y = 100. * tf.reduce_sum(tf.square(x_diff), axis=-1) + tf.reduce_sum(tf.square(x_m_1), axis=-1)
        return y

class Rastrigin(Function):
    def _build_fn(self, ):
        cosi = tf.reduce_sum(tf.cos(2*np.pi*self.X), axis=-1)
        y = 10 * (self.dims - cosi) + tf.reduce_sum(tf.square(self.X), axis=-1)
        return y

def main(args):
    params = vars(args)
    for key, val in params.items():
        print('{}:{}'.format(key, val))
    time.sleep(1)

    all_results = dict()
    random.seed(args.seed)
    np.random.seed(args.seed)
    lr, sigma, pop_size, k,  max_iter = args.lr, args.sigma, args.pop_size, args.k, int(args.max_iter),
    loss_fn, x_init = generate_problem(n=args.dims, func_name=args.func_name)

    xs, ys, ts, errors = es(x_init, loss_fn, lr=lr, sigma=sigma, pop_size=pop_size, max_samples=args.max_iter)
    all_results['ES'] = dict(xs=xs, ys=ys, ts=ts, errors=errors)

    xs, ys, ts, errors = ges(x_init, loss_fn, lr=lr, sigma=sigma, k=k, pop_size=pop_size, max_samples=max_iter)
    all_results['GES'] = dict(xs=xs, ys=ys, ts=ts, errors=errors)

    xs, ys, ts, errors = sges(x_init, loss_fn, lr=lr, sigma=sigma, k=k, pop_size=pop_size, max_samples=max_iter, auto_alpha=True)
    all_results['SGES'] = dict(xs=xs, ys=ys, ts=ts, errors=errors)

    xs, ys, ts, errors = cma_es(x_init, loss_fn, sigma=sigma, pop_size=pop_size*2,  max_samples=max_iter)
    all_results['CMA-ES'] = dict(xs=xs, ys=ys, ts=ts, errors=errors)

    xs, ys, ts, errors = asebo(x_init, loss_fn, lr=lr, sigma=sigma, k=k, pop_size=pop_size, max_samples=max_iter, decay=1-1/k)
    all_results['ASEBO'] = dict(xs=xs, ys=ys, ts=ts, errors=errors)

    os.makedirs(args.root_dir, exist_ok=True)
    for key, dict_result in all_results.items():
        df = pd.DataFrame(data=dict_result)
        save_dir = os.path.join(args.root_dir, args.func_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, '%s-%d.csv' % (key, args.seed))
        df.to_csv(save_path)
        print('save %s result into: %s' % (key, save_path))


if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--dims', type=int, default=1000)
    parse.add_argument('--root_dir', type=str, default='logs')
    parse.add_argument('--func_name', type=str, default='Sphere')
    parse.add_argument('--max_iter', type=float, default=int(1e6))
    parse.add_argument('--seed', type=int, default=2016)
    parse.add_argument('--lr', type=float, default=0.5)
    parse.add_argument('--sigma', type=float, default=0.01)
    parse.add_argument('--pop_size', type=int, default=50)
    parse.add_argument('--k', type=int, default=20)

    args = parse.parse_args()
    main(args)




