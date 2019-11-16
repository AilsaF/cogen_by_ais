import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as layers
import os
import pdb
import argparse
import sys
sys.path.append('/home/tf6/eval_gen-master')
from sampling import samplers_toy as samplers
from theano.sandbox.rng_mrg import MRG_RandomStreams
import operator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle as pickle

sharedX = (lambda X:
           theano.shared(np.asarray(X, dtype=theano.config.floatX)))

rng = np.random.RandomState(1302)
np.random.seed(0)

def sample_data(p, n=100):
    return np.random.multivariate_normal(p, np.array([[0.002,0],[0, 0.002]]), n)

def generate_data(num_data):
    d = np.linspace(0, 360, 6)[:-1]
    x = np.sin(d / 180. * np.pi)
    y = np.cos(d / 180. * np.pi)
    points = np.vstack((y, x)).T
    s0 = sample_data(points[0], num_data+100)
    s1 = sample_data(points[1], num_data+100)
    s2 = sample_data(points[2], num_data+100)
    s3 = sample_data(points[3], num_data+100)
    s4 = sample_data(points[4], num_data+100)
    X = np.vstack((s0, s1, s2, s3, s4))
    ind = np.random.RandomState(seed=2919).permutation(X.shape[0])
    X = X[ind].astype('float32')
    return X


def load_model(hdim, aux):
    filename = '../toy_model/toy_task_ganagaing_lr0.001d_lr0.001hideen2genepoch' + aux + '.pkl'

    def toygenerator(n_hidden, input_var=None):
        network = layers.InputLayer(shape=(None, n_hidden),
                                    input_var=input_var)
        # tanh = lasagne.nonlinearities.tanh
        relu = lasagne.nonlinearities.rectify
        linear = lasagne.nonlinearities.linear
        network = lasagne.layers.DenseLayer(
            network, 20, nonlinearity=relu)
        network = lasagne.layers.DenseLayer(
            network, 20, nonlinearity=relu)
        network = lasagne.layers.DenseLayer(
            network, 2, nonlinearity=linear)
        return network

    gen = toygenerator(hdim)
    print ('load model ' + filename)
    with open(filename, 'r') as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(gen, data)

    def generator(z):
        return lasagne.layers.get_output(gen, z)

    return generator

def plot_img(img, model_name='', epoch=None, n_ex=1000, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.scatter(img[:, 0], img[:, 1])
    plt.axis('on')
    plt.tight_layout()
    plt.show()
    plt.savefig(model_name + 'epoch' + str(epoch) + '_img.png')
    plt.close()


def main():
    hdim = 2
    generator = load_model(hdim, aux)

    X_test = np.array([[1, 0]])

    init_state = np.random.normal(0, 1, size=[num_samples, hdim]).astype(np.float32)
    # priorX = (generator(init_state)).eval()
    finalstate= samplers.run_ais(generator, X_test, num_samples, num_steps, sigma, hdim, L, eps, data,
                                 init_state=init_state, mask=np.array([[0, 1]]), temp_range=temp_range)
    post_img = (generator(finalstate)).eval()
    print post_img.shape

    samples = generate_data(100)
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.scatter(post_img[:, 0], post_img[:, 1], c='y', marker="*")
    plt.xlim((-1.5, 1.5))
    plt.ylim((-1.5, 1.5))
    plt.xlabel('$x_1$', fontsize=18)
    plt.ylabel('$x_2$', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig("../toy_results/aisx{}epoch.pdf".format(int(aux) + 1), bbox_inches='tight')

    plt.clf()
    fig = plt.figure(figsize=(5, 4.8))
    dis = ((post_img - np.array([[1, 0]])) ** 2).mean(axis=1)
    bins = map(lambda x: x * 0.52 / 3, range(20))
    plt.hist(dis, bins=bins)
    plt.ylim((0, 100))
    plt.xlabel('reconstruction error', fontsize=18)
    plt.ylabel('count', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig("../toy_results/aisbar{}epoch.pdf".format(int(aux) + 1), bbox_inches='tight')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gantoy", type=str)
    parser.add_argument("--model_aux", default="14999", type=str)
    parser.add_argument("--prior", default="normal", type=str)
    parser.add_argument("--sigma", default=0.00005, type=float)
    parser.add_argument("--num_test", default=1, type=int)
    parser.add_argument("--num_steps", default=10000, type=int)
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--eps", default=0.01, type=float)
    parser.add_argument("--L", default=20, type=int)
    parser.add_argument("--data", default='continuous', type=str)
    parser.add_argument("--low_temp_range", default=-10, type=float)
    parser.add_argument("--high_temp_range", default=2, type=float)

    args = parser.parse_args()
    model = args.model
    aux = args.model_aux
    prior = args.prior
    num_test = args.num_test
    num_steps = args.num_steps
    num_samples = args.num_samples
    sigma = args.sigma
    eps = args.eps
    L = args.L
    data = args.data
    temp_range = (args.low_temp_range, args.high_temp_range)

    main()

