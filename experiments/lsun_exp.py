import numpy as np
import os
import pdb
import argparse
import sys
sys.path.append('/home/tf6/cogen_AIS/')
sys.path.append('/home/tf6/cogen_AIS/celeba_progan')
from sampling import samplers_celeba as samplers
from celeba_progan import misc, config, network, dataset
misc.init_output_logging()
import random
from PIL import Image
from skimage.measure import compare_ssim as ssim


if __name__ == "__main__":
    print 'Importing Theano...'

os.environ['THEANO_FLAGS'] = ','.join([key + '=' + value for key, value in config.theano_flags.iteritems()])
sys.setrecursionlimit(10000)
import theano
import theano.tensor as T

sharedX = (lambda X:
           theano.shared(np.asarray(X, dtype=theano.config.floatX)))

rng = np.random.RandomState(1302)
np.random.seed(99)


def generate_data():
    data = Image.open('../sample_imgs/lsun_sample.png')
    data = np.asarray(data)
    data = data[:, :, :3]
    data = np.moveaxis(data, -1, 0)
    data = np.expand_dims(data, axis=0)
    data = data.astype(np.float32)
    data = misc.adjust_dynamic_range(data, [0,255], [-1, 1])
    return data


def main(aux, num_steps, temp_range):
    result_subdir = misc.create_result_subdir(config.result_dir, config.run_desc)
    hdim = 512
    X_test = generate_data()
    print "image shape: {}".format(X_test.shape)
    misc.save_image_grid(X_test, os.path.join(result_subdir, 'true.png'), drange=[-1, 1], grid_size=(1, 1))

    box_pos = random.sample(range(1, 60), 2)
    height = random.randint(50, 80)
    width = random.randint(50, 80)
    mask = np.ones_like(X_test)
    mask[:, :, box_pos[0]:box_pos[0] + height, box_pos[1]:box_pos[1] + width] = 0

    misc.save_image_grid(X_test * mask, os.path.join(result_subdir, 'trueBlocked.png'), drange=[-1, 1],
                         grid_size=(1, 1))

    G, _, _ = misc.load_pkl('/home/tf6/cogen_AIS/model/lsun-network-snapshot-{}.pkl'.format(aux))
    Gs = G.create_temporally_smoothed_version(beta=0.999, explicit_updates=True)

    def gen_fn(z):
        output = Gs.eval_nd(z, ignore_unused_inputs=True)
        return output

    init_state = np.random.normal(0, 1, size=[5 , hdim]).astype(np.float32)

    finalstate = samplers.run_ais(gen_fn, X_test, 5, num_steps, hdim, L, eps,
                                         init_state=init_state, save_dir=result_subdir,
                                  box=(box_pos[0], box_pos[0] + height, box_pos[1], box_pos[1] + width),
                                  temp_range=temp_range)

    post_img = gen_fn(finalstate).eval()
    misc.save_image_grid(post_img, result_subdir+'/fake.png'.format(j), drange=[-1, 1],
                         grid_size=(5, 1))

    # calculate ssim
    all_ssim = []
    t = Image.open(result_subdir + '/true.png')
    t = np.array(t)
    f = Image.open(result_subdir + '/fake.png')
    f = np.array(f)

    for i in range(5):
        f1 = f[:128, 128 * i: 128 * (i + 1)]
        all_ssim.append(ssim(t, f1, multichannel=True))
    print all_ssim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_aux", default="010002", type=str)
    parser.add_argument("--num_steps", default=500, type=int)
    parser.add_argument("--eps", default=0.01, type=float)  ##
    parser.add_argument("--L", default=20, type=int)
    parser.add_argument("--low_temp_range", default=-5, type=float)
    parser.add_argument("--high_temp_range", default=5, type=float)

    args = parser.parse_args()
    aux = args.model_aux
    num_steps = args.num_steps
    eps = args.eps
    L = args.L
    # image_num = args.image_num
    temp_range = (args.low_temp_range, args.high_temp_range)

    main(aux, num_steps, temp_range)
