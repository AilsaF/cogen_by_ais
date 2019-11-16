import numpy as np
from algorithms.hmc import*
import theano
import theano.tensor as T
from algorithms import ais
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.random.seed(123)
sharedX = (lambda X:
           theano.shared(np.asarray(X, dtype=theano.config.floatX)))


class AISPath:
    def __init__(self, generator, obs, num_samples, hdim, L, epsilon, box=(60, 100, 60, 100), init_state=None):
        self.generator = generator
        self.batch_size = obs.shape[0]
        print obs.shape, init_state.shape
        self.img_size = obs.shape[-1]
        self.obs_val = sharedX(np.reshape(obs, [1, self.batch_size, 3, self.img_size, self.img_size]))
        self.obs = T.tensor5()
        self.t = T.scalar()
        self.n_sam = num_samples
        self.hdim = hdim
        self.L = L
        self.eps = epsilon
        # mask
        mask = numpy.ones((1, 3, self.img_size, self.img_size))
        mask[:, :, box[0]:box[1], box[2]:box[3]] = 0
        self.mask = sharedX(mask)
        self.build(self.eps, self.L,init_state = init_state)

        print self.img_size, self.n_sam

    def build(self,
              initial_stepsize,
              n_steps,
              target_acceptance_rate=.65,
              stepsize_dec=0.98,
              stepsize_min=0.0001,
              stepsize_max=0.5,
              stepsize_inc=1.02,
              # used in geometric avg. 1.0 would be not moving at all
              avg_acceptance_slowness=0.9,
              seed=12345,
              init_state=None
              ):
        init_h = init_state
        print ('load init_state')

        # For HMC
        # h denotes current states
        self.h = sharedX(init_h)
        # m denotes momentum
        t = T.scalar()
        self.generated = self.generate(self.h)
        lld = T.reshape(-self.energy_fn(self.h), [self.n_sam, self.batch_size])
        self.eval_lld = theano.function([t], lld, givens={self.obs: self.obs_val, self.t: t})

        # allocate shared variables
        stepsize = sharedX(initial_stepsize)
        avg_acceptance_rate = sharedX(target_acceptance_rate)
        s_rng = TT.shared_randomstreams.RandomStreams(seed)

        # define graph for an `n_steps` HMC simulation
        accept, final_pos = hmc_move(
            s_rng,
            self.h,
            self.energy_fn,
            stepsize,
            n_steps)

        # define the dictionary of updates, to apply on every `simulate` call
        simulate_updates = hmc_updates(
            self.h,
            stepsize,
            avg_acceptance_rate,
            final_pos=final_pos,
            accept=accept,
            stepsize_min=stepsize_min,
            stepsize_max=stepsize_max,
            stepsize_inc=stepsize_inc,
            stepsize_dec=stepsize_dec,
            target_acceptance_rate=target_acceptance_rate,
            avg_acceptance_slowness=avg_acceptance_slowness)

        self.step = theano.function([t], [accept], updates=simulate_updates, givens={self.obs: self.obs_val, self.t: t})

    def init_partition_function(self):
        return 0.

    def prior_logpdf(self, state):
        return (1-self.t) * (-T.sum(T.square(state), [-1]) / (2.) - self.hdim / 2. * np.log(2 * np.pi))

    def likelihood(self, generated):
        masked_diff = generated * T.addbroadcast(self.mask, 0) - T.addbroadcast(self.obs * self.mask, 0)
        return self.t * (-T.sum(T.square(masked_diff), [-1, -2, -3]))

    def energy_fn(self, state):
        generated = self.generate(state)
        state_ = T.reshape(state, [self.n_sam, self.batch_size, self.hdim])
        energy = - (self.prior_logpdf(state_) + self.likelihood(generated))
        return T.reshape(energy, [-1])

    def generate(self, state):
        generated = self.generator(state)
        #print generated.shape.eval()
        #print [self.n_sam, self.batch_size, 3, self.img_size, self.img_size]
        generated = T.reshape(generated, [self.n_sam, self.batch_size, 3, self.img_size, self.img_size])
        return generated

def run_ais(model, obs, num_samples, num_steps, hdim, L, epsilon, schedule=None, init_state=None,
            save_dir='.', box=(60, 100, 60, 100), temp_range=(-4,4)):
    if schedule is None:
        schedule = ais.sigmoid_schedule(num_steps, temp_range)

    path = AISPath(model, obs, num_samples, hdim, L, epsilon, box=box, init_state=init_state)
    lld = ais.ais(path, schedule, save_dir)
    return lld
