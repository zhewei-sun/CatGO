import multiprocessing
import time
import re

import numpy as np
from scipy.optimize import minimize

from tqdm import trange

from util import log_likelihood, normalize

cfre=re.compile(r'cf_(?P<model>.+)_(?P<k>[0-9]+)')

class Categorizer:
    
    def __init__(self, categories, exemplars, queries, query_labels, cf_feats=None):
        
        # Category Names - V
        self.categories = np.asarray(categories)
        
        # Exemplar Vectors for each Category - V * x
        self.exemplars = np.asarray(exemplars)
        
        # Query Vectors - N
        self.queries = np.asarray(queries)
        
        # Query Labels - N (vocab_inds)
        self.query_labels = np.asarray(query_labels)
        
        # Collaborative Filtering Features (distances between all category pairs) - V*V*x
        self.cf_feats = cf_feats
        
        self.parameters = {}
        self.results = {}
        
        self.processes = {}
        
        self.N_query = self.queries.shape[0]
        self.N_cat = self.categories.shape[0]
        self.E = self.queries.shape[1]
        self.CF_dim = cf_feats.shape[0]
        
        self.prior = {'uniform': np.ones(self.N_cat) / float(self.N_cat)}
        
    def set_inds(self, train_inds, test_inds):
        self.train_inds = train_inds
        self.test_inds = test_inds
        
    def set_datadir(self, data_dir):
        self.data_dir = data_dir
        
    def add_prior(self, name, p_prior):
        self.prior[name] = p_prior
        
    def preprocess(self, models, verbose=False):
        # Pre-compute Exemplar distances
#        self.vd_exemplar = []
#        for i in range(self.N_cat):
#            if verbose:
#                if i%500==0:
#                    print(i)
#            vd_dist = np.zeros((self.N_query, self.exemplars[i].shape[0]))
#            for j in range(self.N_query):
#                vd_dist[j,:] = -1*np.linalg.norm(self.exemplars[i] - self.queries[j], axis=1)**2
#            self.vd_exemplar.append(vd_dist)
            
        # Pre-compute 1NN Distances
        self.vd_onenn = np.zeros((self.N_query, self.N_cat))
                
        for j in range(self.N_cat):
            self.vd_onenn[:, j] = np.max(self.vd_exemplar[j], axis=1)
            
        # Pre-compute Prototypes
        prototypes = np.zeros((self.N_cat, self.E))
        for i in range(self.N_cat):
            prototypes[i] = np.mean(self.exemplars[i], axis=0)
            
        self.vd_prototype = np.zeros((self.N_query, self.N_cat))
        
        for i in range(self.N_query):
            if verbose:
                if i%500==0:
                    print(i)
            self.vd_prototype[i] = np.linalg.norm(prototypes - self.queries[i], axis=1)
                
        self.vd_prototype = -1*self.vd_prototype**2

        
    def run_categorization(self, models=['onenn', 'exemplar', 'prototype'], prior='uniform', verbose=False):
        
        # Preprocess
        self.preprocess(models, verbose=verbose)
        
        # Fork - run_model
        for i in trange(len(models)):
            model = models[i]
            if model == 'onenn':
                self.run_onenn(self.train_inds, self.test_inds, prior=prior, verbose=verbose)
            if model == 'exemplar':
                self.run_exemplar(self.train_inds, self.test_inds, prior=prior, verbose=verbose)
            if model == 'prototype':
                self.run_prototype(self.train_inds, self.test_inds, prior=prior, verbose=verbose)
            cf_match = cfre.search(model)
            if cf_match is not None:
                self.run_cf(self.train_inds, self.test_inds, cf_match['model'], int(cf_match['k'])+1, prior=prior, verbose=verbose)
           
    def run_categorization_batch(self, models=['onenn', 'exemplar', 'prototype'], prior='uniform', verbose=False):
        # Preprocess
        self.preprocess(models, verbose=verbose)
        
        # Fork - run_model
        self.kill_all_processes()
        self.processes = {}
        
        for model in models:
            if model == 'onenn':
                self.processes[model] = multiprocessing.Process(target=self.run_onenn, args=[self.train_inds, self.test_inds, prior])
            if model == 'exemplar':
                self.processes[model] = multiprocessing.Process(target=self.run_exemplar, args=[self.train_inds, self.test_inds, prior])
            if model == 'prototype':
                self.processes[model] = multiprocessing.Process(target=self.run_prototype, args=[self.train_inds, self.test_inds, prior])
            cf_match = cfre.search(model)
            if cf_match is not None:
                self.processes[model] = multiprocessing.Process(target=self.run_cf, args=[self.train_inds, self.test_inds, cf_match['model'], int(cf_match['k'])+1, prior])
             
        for key, p in self.processes.items():
            print('Starting Process: '+key)
            p.start()
            
        # Periodically Check Completion
        closed = set()
        for i in trange(len(self.processes)):
            waiting = True
            while waiting:
                for key, p in self.processes.items():
                    if not p.is_alive():
                        if key not in closed:
                            closed.add(key)
                            waiting=False
                            break
                if not waiting:
                    break
                time.sleep(5)
                
        # TODO: read return value, update parameters
            
    def kill_all_processes(self):
        for key, p in self.processes.items():
            if p.is_alive():
                p.kill()           
                
    def run_onenn(self, train_ind, test_ind, prior='uniform', verbose=False):
        kernel = lambda params,inds,verbose: self.search_kernel(self.search_onenn, params, [inds], prior_name=prior, verbose=verbose)
        return self.run_model(kernel, 'onenn', [1], ((10**-2, 10**2),), train_ind, test_ind, prior_name=prior, verbose=verbose)
        
    def run_exemplar(self, train_ind, test_ind, prior='uniform', verbose=False):
        kernel = lambda params,inds,verbose: self.search_kernel(self.search_exemplar, params, [inds], prior_name=prior, verbose=verbose)
        return self.run_model(kernel, 'exemplar', [1], ((10**-2, 10**2),), train_ind, test_ind, prior_name=prior, verbose=verbose)
    
    def run_prototype(self, train_ind, test_ind, prior='uniform', verbose=False):
        kernel = lambda params,inds,verbose: self.search_kernel(self.search_prototype, params, [inds], prior_name=prior, verbose=verbose)
        return self.run_model(kernel, 'prototype', [1], ((10**-2, 10**2),), train_ind, test_ind, prior_name=prior, verbose=verbose)
        
    def run_cf(self, train_ind, test_ind, model, k, prior='uniform', verbose=False):
        kernel = lambda params,inds,verbose: self.search_kernel(self.search_cf, params, [inds, model, k, prior], prior_name=prior, verbose=verbose)
        cf_init = [1,1] + (self.CF_dim-1) * [1.0/self.CF_dim]
        cf_bounds = [(10**-2, 10**2),(10**-2, 10**2)] + (self.CF_dim-1) * [(0,1)]
        if self.CF_dim == 1:
            cf_constraints = None
        else:
            cf_constraints = {'type': 'ineq', 'fun': lambda x: 1.0 - np.sum(x[2:]) }
        return self.run_model(kernel, 'cf_'+model+'_'+str(k-1), cf_init, cf_bounds, train_ind, test_ind, prior_name=prior, constraints=cf_constraints, verbose=verbose)

    def run_model(self, kernel, ker_name, init, bounds, train_ind, test_ind, prior_name='uniform', constraints=None, verbose=False):
        
        # Minimizer
        if constraints is None:
            result = minimize(lambda x:kernel(x, train_ind, verbose)[0], init, bounds=bounds)
        else:
            result = minimize(lambda x:kernel(x, train_ind, verbose)[0], init, bounds=bounds, constraints=constraints)

        # Save Results
        
        if verbose:
            print("[params = "+str(result.x)+"]")
        
        nll_train, likelihood_train = kernel(result.x, train_ind, verbose)
        np.save(self.data_dir+'l_'+ker_name+'_'+prior_name+'_train.npy', likelihood_train)
    
        nll_test, likelihood_test = kernel(result.x, test_ind, verbose)
        np.save(self.data_dir+'l_'+ker_name+'_'+prior_name+'_test.npy', likelihood_test)
        
        if verbose:
            print("Log_Likelihood (Train): " + str(nll_train))
            print("Log_Likelihood (Test): " + str(nll_test))
            
        return result.x

    
    def search_kernel(self, likelihood_func, likelihood_params, likelihood_args, prior_name='uniform', train=True, verbose=True):
        inds = likelihood_args[0]
        N = inds.shape[0]
        
        l_likelihood = likelihood_func(likelihood_params, likelihood_args)
        p_prior = self.prior[prior_name]
        
        l_posterior = normalize(l_likelihood * p_prior, axis=1)
        p_posterior = l_posterior[np.arange(N), self.query_labels[inds]]
        
        if verbose:
            print("[params = %s]" % str(likelihood_params))
            print("Log_likelihood: %f" % log_likelihood(p_posterior))
        
        return log_likelihood(p_posterior), l_posterior
    
    def search_onenn(self, params, args):
        
        h = params[0]
        inds = args[0]
        
        l_onenn = normalize(np.exp(self.vd_onenn[inds]/h), axis=1)
    
        return l_onenn
    
    def search_exemplar(self, params, args):
        
        h = params[0]
        inds = args[0]
        
        N = inds.shape[0]
        l_exemplar = np.zeros((N, self.N_cat))
            
        for j in range(self.N_cat):
            l_exemplar[:,j] = np.sum(np.exp(self.vd_exemplar[j][inds] / h ), axis=1)
        
        return normalize(l_exemplar, axis=1)
    
    def search_prototype(self, params, args):
        
        h = params[0]
        inds = args[0]
            
        l_prototype = normalize(np.exp(self.vd_prototype[inds]/h), axis=1)
        
        return l_prototype
    
    def search_cf(self, params, args):
        
        h_model = params[0]
        h_word = params[1]
        if len(params) > 2:
            alphas = np.asarray(params[2:]+[1-np.sum(params[2:])])[:, np.newaxis, np.newaxis]
        else:
            alphas = np.asarray([1])[:, np.newaxis, np.newaxis]
        
        inds = args[0]
        model = args[1]
        k = args[2]
        prior_name = args[3]
        
        N = inds.shape[0]
        
        if model == 'onenn':
            l_likelihood = self.search_onenn(params, args)
        if model == 'prototype':
            l_likelihood = self.search_prototype(params, args)
        if model == 'exemplar':
            l_likelihood = self.search_exemplar(params, args)
        
        p_prior = self.prior[prior_name]
        l_model = normalize(l_likelihood * p_prior, axis=1)
        
        cf_feats_weighted = np.sum(self.cf_feats * alphas, axis=0)
        
        neighbors = np.zeros((self.N_cat, k), dtype=np.int32)
        for i in range(self.N_cat):
            neighbors[i,:] = np.argsort(cf_feats_weighted[i,:])[:k]
            
        vd_vocab = np.exp(-1*cf_feats_weighted**2/h_word)
        
        vd_vocab_cache = normalize(np.stack([vd_vocab[np.arange(self.N_cat), neighbors[:,i]] for i in range(k)], axis=1), axis=1)

        l_model_cache = np.reshape(l_model[:, neighbors[:,:k]], (N,-1))
        vvc_flat = np.reshape(vd_vocab_cache, -1)
        l_margin = normalize(np.sum(np.reshape(l_model_cache * vvc_flat, (N, self.N_cat, k)), axis=2), axis=1)

        return l_margin
    
        
    def get_rankings(self, l_model, inds):
        N = inds.shape[0]
        ranks = np.zeros((N, self.N_cat), dtype=np.int32)
        rankings = np.zeros(N, dtype=np.int32)
        
        for i in range(N):
            ranks[i] = np.argsort(l_model[i])[::-1]
            rankings[i] = ranks[i].tolist().index(self.query_labels[inds[i]])+1
            
        return rankings
        
    
    def compute_results(self, models=['onenn', 'exemplar', 'prototype'], prior='uniform'):
        
        self.results['random'] = {'nll_train': log_likelihood(np.ones(self.train_inds.shape[0])/self.N_cat), \
                          'nll_test': log_likelihood(np.ones(self.test_inds.shape[0])/self.N_cat), \
                          'rank_train': self.N_cat/2.0, \
                          'rank_test': self.N_cat/2.0}
        
        for model in models:
            self.results[model] = {}
            
            l_model_train = np.load(self.data_dir+'l_'+model+'_'+prior+'_train.npy')
            self.results[model]['nll_train'] = log_likelihood(l_model_train[np.arange(self.train_inds.shape[0]), self.query_labels[self.train_inds]])
            self.results[model]['rank_train'] = np.mean(self.get_rankings(l_model_train, self.train_inds))
            
            l_model_test = np.load(self.data_dir+'l_'+model+'_'+prior+'_test.npy')
            self.results[model]['nll_test'] = log_likelihood(l_model_test[np.arange(self.test_inds.shape[0]), self.query_labels[self.test_inds]])
            self.results[model]['rank_test'] = np.mean(self.get_rankings(l_model_test, self.test_inds))
    
    def summarize_model(self, model):
        print('['+model.upper()+']')
        print("Log_Likelihood (Train): " + str(self.results[model]['nll_train']))
        print("Log_Likelihood (Test): " + str(self.results[model]['nll_test']))
        print("Expected_Rank (Train): " + str(self.results[model]['rank_train']))
        print("Expected_Rank (Test): " + str(self.results[model]['rank_test']))
    
    def summarize(self, models=['onenn', 'exemplar', 'prototype'], prior='uniform'):
        
        self.summarize_model('random')
        
        for model in models:
            self.summarize_model(model)
        
        
        
        
        
        
        
        
        
        