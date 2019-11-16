import time
import numpy as np
from celeba_progan import misc
nax = np.newaxis
import pdb
import theano.tensor as T

DEBUGGER = None


def sigmoid_schedule(num, range=(-4 ,4)):
    """The sigmoid schedule defined in Section 6.2 of the paper. This is defined as:

          gamma_t = sigma(rad * (2t/T - 1))
          beta_t = (gamma_t - gamma_1) / (gamma_T - gamma_1),

    where sigma is the logistic sigmoid. This schedule allocates more distributions near
    the inverse temperature 0 and 1, since these are often the places where the distributon
    changes the fastest.
    """
    if num == 2:
        return [np.asarray(0.0), np.asarray(1.0)]
    t = np.linspace(range[0], range[1], num)
    sigm = 1. / (1. + np.exp(-t))
    return sigm


def ais(problem, schedule, save_dir):
    """Run AIS in the forward direction. Problem is as defined above, and schedule should
    be an array of monotonically increasing values with schedule[0] == 0 and schedule[-1] == 1."""
    index = 1
    # zs = []
    for it, (t0, t1) in enumerate(zip(schedule[:-1], schedule[1:])):
        accept = problem.step(t1.astype(np.float32))
        if (index + 1) % 100 == 0:
            print ("\nsteps %i" % index)
            print ("Accept: " + str(np.mean(accept)))
            # save intermediate image
            # img = problem.generate(problem.h).eval()[:, 0]
            # misc.save_image_grid(img, save_dir + '/{}_AIS_Img.png'.format(index), drange=[-1, 1],
            #                      grid_size=(problem.n_sam, 1))
        index += 1
    finalstate = problem.h.get_value()
    return finalstate

