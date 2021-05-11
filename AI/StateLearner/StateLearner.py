import numpy as np
from collections import Counter
import random
from numpy.core.numeric import NaN
from numpy.random import dirichlet, multinomial
from scipy.special import softmax
import multiprocessing
import itertools

#Based on GERSHMAN, BLEI, AND NIV (2010)
#blame for bugs: Jeff Rhoades (github.com/rhoadesScholar, 2021)
#inputs for running, besides meta-variables (m and alpha), 
#are just obs = vector of observations at time t, supplied by environment*
class StateLearner:       
    class Particle:
        class K:
            def __init__(self, prior):
                self.obs = [Counter([o]) for o in prior]
                self.count = 1
                return None

            def add_obs(self, obs):#obs must be tuple
                for i, o in enumerate(obs):
                    self.obs[i].update([o])
                self.count += 1

            def get_N(self, i, j):
                return self.obs[i][j]

            def get_Ni_sum(self, i):
                return sum(self.obs[i].values()) + len(self.obs[i].keys())
            
            def get_prob(self, obs):
                probs = np.ones(obs.shape)
                for i, o in enumerate(obs):
                    if not np.isnan(o):#allows for marginalization
                        probs[i] *= (self.get_N(i,o) + 1) / self.get_Ni_sum(i) #Equation A3
                return probs

            def step(self, obs):
                probs = self.get_prob(obs)
                self.add_obs(obs)
                return probs
        
        def __init__(self, prior_dims, alpha=0.1):                        
            self.alpha = alpha
            self.prior_dims = np.array(prior_dims)
            self.Ks = [self.K(self.get_Dir_prior_draw())]
            self.K_num = 1
            self.Cs = [0]
            self.t = 0  
            return None        
                    
        def get_Dir_prior_draw(self):                                                    
            return tuple(np.argwhere(multinomial(1, dirichlet(np.ones(dim))))[0][0] for dim in self.prior_dims)

        def next_k_prob(self): #Accomplishes Equation 1
            probs = np.ndarray((len(self.Ks) + 1, 1))
            for i, k in enumerate(self.Ks):
                Nk = k.count
                Pk = Nk / (self.t + self.alpha)
                probs[i] = Pk
            Pk = self.alpha / (self.t + self.alpha)
            probs[self.K_num] = Pk
            return probs        

        def get_C(self, t):            
            while len(self.Cs) <= t:
                self.Cs.append(random.choices(range(self.K_num + 1), self.next_k_prob())[0])
            return self.Cs[t]

        def draw_C(self):
            return random.choices(range(self.K_num + 1), self.next_k_prob())[0]

        def get_Prob(self, t, obs):
            this_C = self.draw_C()
            if this_C >= self.K_num:                
                return self.get_Dir_prior_draw()
            else:
                return self.Ks[this_C].get_prob(obs)        

        def step(self, obs):
            self.t += 1
            next_C = self.get_C(self.t)
            if next_C >= self.K_num:
                self.K_num += 1
                self.Ks.append(self.K(self.get_Dir_prior_draw()))#SAMPLE FROM PRIOR
            probs = self.Ks[next_C].step(obs)
            return probs

    def __init__(self, prior_dims, draws=100, m=100, alpha=[0.1], to_predict=0, parallel=False):
        if len(alpha) != m:#alpha may be vector for particles that learn at different rates
            self.alpha = np.ones((m,1)) * alpha
        else:
            self.alpha = alpha

        self.Particles = [self.Particle(prior_dims=prior_dims, alpha=self.alpha[i]) for i in range(m)]
        self.Ws = np.ones(m) * 1/m
        self.t = 0
        self.m = m
        self.to_predict = to_predict
        self.pred_probs = np.array([])
        self.parallel = parallel
        self.prior_dims = prior_dims
        self.draws = draws
        return None

    def get_Prob_prod(self, p, t, obs):
        return np.prod(p.get_Prob(t, obs))

    def get_Step_prod(self, p, obs):
        return np.prod(p.step(obs))

    def get_Particle_draws(self):
        counts = multinomial(self.draws, self.Ws)
        these_particles = []
        for i, draws in enumerate(counts):
            for d in range(draws):
                these_particles.append(self.Particles[i])
        return these_particles

    def get_prediction(self, obs, to_predict=NaN, t=NaN):#to_predict is index of obs variable to predict
        if np.isnan(t):
            t = self.t
        if np.isnan(to_predict):
            to_predict = self.to_predict
        
        these_particles = self.get_Particle_draws()# NEED TO RESAMPLE PARTICLES BY WEIGHTS

        if self.parallel:
            temp_obs = obs.copy()
            temp_obs[to_predict] = NaN
        
            pool = multiprocessing.Pool()
            rs = softmax(pool.starmap(self.get_Prob_prod, 
                                        zip(these_particles, itertools.repeat(t), itertools.repeat(temp_obs)), 
                                        chunksize=self.m/8))    

            temp_obs = np.ones(obs.shape) * NaN
            temp_obs[to_predict] = obs[to_predict]
            
            pool = multiprocessing.Pool()
            probs = pool.starmap(self.get_Prob_prod, 
                                zip(these_particles, itertools.repeat(t), itertools.repeat(temp_obs)), 
                                chunksize=self.m/8)
            prob = [sum(rs * probs)]
        else:
            all_probs = np.ndarray((self.draws, np.array(obs).shape[0]))
            for i, p in enumerate(these_particles):
                all_probs[i,:] = p.get_Prob(t, obs)
            raw_probs = all_probs[:,to_predict]
            all_probs[:,to_predict] = 1
            rs = softmax(np.prod(all_probs, axis=1))
            prob = [rs @ raw_probs]#weight samples by likelihoods
            all_probs[:,to_predict] = raw_probs

        return prob, all_probs

    def step(self, obs):
        self.t += 1
        
        pred_prob, _ = self.get_prediction(obs)#predict reward
        self.pred_probs = np.append(self.pred_probs, pred_prob, axis=0)

        if self.parallel:
            pool = multiprocessing.Pool()
            self.Ws = softmax(pool.starmap(self.get_Step_prod, 
                                zip(self.Particles, itertools.repeat(obs)), 
                                chunksize=self.m/8))
        else:
            for i, p in enumerate(self.Particles):
                self.Ws[i] = np.prod(p.step(obs))
            self.Ws = softmax(self.Ws)
            #PROBABLY NEED TO REPLACE DEGENERATE PARTICLES

        return pred_prob
    
    def run(self, all_obs):
        [self.step(obs) for obs in all_obs]
        return self.pred_probs
