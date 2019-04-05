#import io
#import pickle
import multiprocessing
import time
import re
import tqdm

import numpy as np
#import pandas as pd
import scipy.spatial.distance as dist
#from scipy.stats import norm
from scipy.optimize import minimize

#from collections import defaultdict, namedtuple

#from nltk.corpus import stopwords as sw
#from gensim.utils import simple_preprocess

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
        
        self.N_query = self.queries.shape[0]
        self.N_cat = self.categories.shape[0]
        self.E = self.queries.shape[1]
        
        self.prior = {'uniform': np.ones(self.N_cat) / float(self.N_cat)}
        
    def set_inds(self, train_inds, test_inds):
        self.train_inds = train_inds
        self.test_inds = test_inds
        
    def set_datadir(self, data_dir):
        self.data_dir = data_dir
        
    def add_prior(self, name, p_prior):
        self.prior[name] = p_prior
        
    def preprocess(self, models):
        # Pre-compute Exemplar distances
#        self.vd_exemplar = []
#        for i in range(self.N_cat):
#            if i%500==0:
#                print(i)
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
            if i%500==0:
                print(i)
            self.vd_prototype[i] = np.linalg.norm(prototypes - self.queries[i], axis=1)
                
        self.vd_prototype = -1*self.vd_prototype**2

        
    def train_categorization(self, models=['onenn', 'exemplar', 'prototype'], prior='uniform'):
        
        # Preprocess
        self.preprocess(models)
        
        # Fork - run_model
        for model in models:
            if model == 'onenn':
                self.run_onenn(self.train_inds, self.test_inds, prior=prior, verbose=True)
            if model == 'exemplar':
                self.run_exemplar(self.train_inds, self.test_inds, prior=prior, verbose=True)
            if model == 'prototype':
                self.run_prototype(self.train_inds, self.test_inds, prior=prior, verbose=True)
            cf_match = cfre.search(model)
            if cf_match is not None:
                self.run_cf(self.train_inds, self.test_inds, cf_match['model'], int(cf_match['k'])+1, prior=prior, verbose=True)
           
    def train_categorization(self, models=['onenn', 'exemplar', 'prototype'], prior='uniform'):
            
    def run_onenn(self, train_ind, test_ind, prior='uniform', verbose=False):
        kernel = lambda params,inds,verbose: self.search_kernel(self.search_onenn, params, [inds], prior_name=prior, verbose=verbose)
        self.run_model(kernel, 'onenn', [1], ((10**-2, 10**2),), train_ind, test_ind, verbose)
        
    def run_exemplar(self, train_ind, test_ind, prior='uniform', verbose=False):
        kernel = lambda params,inds,verbose: self.search_kernel(self.search_exemplar, params, [inds], prior_name=prior, verbose=verbose)
        self.run_model(kernel, 'exemplar', [1], ((10**-2, 10**2),), train_ind, test_ind, verbose)
    
    def run_prototype(self, train_ind, test_ind, prior='uniform', verbose=False):
        kernel = lambda params,inds,verbose: self.search_kernel(self.search_prototype, params, [inds], prior_name=prior, verbose=verbose)
        self.run_model(kernel, 'prototype', [1], ((10**-2, 10**2),), train_ind, test_ind, verbose)
        
    def run_cf(self, train_ind, test_ind, model, k, prior='uniform', verbose=False):
        kernel = lambda params,inds,verbose: self.search_kernel(self.search_cf, params, [inds, model, k], prior_name=prior, verbose=verbose)
        self.run_model(kernel, 'cf_'+model+'_'+str(k-1), [1,1], ((10**-2, 10**2),(10**-2, 10**2),), train_ind, test_ind, verbose)

    def run_model(self, kernel, ker_name, init, bounds, train_ind, test_ind, verbose=False):
        
        # Minimizer
        result = minimize(lambda x:kernel(x, train_ind, verbose=verbose)[0], init, bounds=bounds)
        
        # Update Parameter
        self.parameters[ker_name] = result.x
        
        # Save Results
        
        if verbose:
            print("[params = "+str(result.x)+"]")
        
        nll_train, likelihood_train = kernel(result.x, train_ind, verbose=False)
        #np.save(self.data_dir+'l_'+ker_name+'_train.npy', likelihood_train)
    
        nll_test, likelihood_test = kernel(result.x, test_ind, verbose=False)
        #np.save(self.data_dir+'l_'+ker_name+'_test.npy', likelihood_test)
        
        if verbose:
            print("Log_Likelihood (Train): " + str(nll_train))
            print("Log_Likelihood (Test): " + str(nll_test))

    
    def search_kernel(self, likelihood_func, likelihood_params, likelihood_args, prior_name='uniform', verbose=True):
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
    
    def search_cf(self, params, args):#h_word, h_model, model, k, inds, verbose=True):
        
        h_model = params[0]
        h_word = params[1]
        
        inds = args[0]
        model = args[1]
        k = args[2]
        
        N = inds.shape[0]
    
        if model == 'onenn' or model == 'prototype':
            if model == 'onenn':
                vd_model = self.vd_onenn
            elif model == 'prototype':
                vd_model = self.vd_prototype
            vocab_dist = np.exp(vd_model[inds]/h_model)
        
        elif model == 'exemplar':
            vocab_dist = np.zeros((N, self.N_cat))
            for j in range(self.N_cat):
                vocab_dist[:,j] = np.sum(np.exp(self.vd_exemplar[j][inds] / h_model ), axis=1)
        #TODO: add prior
        l_model = normalize(vocab_dist, axis=1)
        
        neighbors = np.zeros((self.N_cat, k), dtype=np.int32)
        for i in range(self.N_cat):
            neighbors[i,:] = np.argsort(self.cf_feats[0][i,:])[:k]
            
        vd_vocab = np.exp(-1*self.cf_feats[0]**2/h_word)
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        