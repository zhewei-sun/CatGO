import pickle
import multiprocessing
import time
import re

import numpy as np
from scipy.optimize import minimize

from tqdm import trange

from .util import log_likelihood, normalize
#from util import log_likelihood, normalize

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
        
        self.prior = {'uniform': np.ones((self.N_query, self.N_cat)) / float(self.N_cat)}
        
    def set_inds(self, train_inds, test_inds):
        self.train_inds = train_inds
        self.test_inds = test_inds
        
    def set_datadir(self, data_dir):
        self.data_dir = data_dir
        
    def save_parameters(self):
        param_file = open(self.data_dir+"parameters.pkl","wb")
        pickle.dump(self.parameters,param_file)
        param_file.close()
        
    def add_prior(self, name, l_prior):
        self.prior[name] = l_prior
        
    def preprocess(self, models, verbose=False):
        
        print("Pre-processing Distances...")
        time.sleep(0.5)
        
        # Pre-compute Exemplar distances
        self.vd_exemplar = []
        for i in trange(self.N_cat):
            if verbose:
                if i%500==0:
                    print(i)
            vd_dist = np.zeros((self.N_query, self.exemplars[i].shape[0]))
            for j in range(self.N_query):
                vd_dist[j,:] = -1*np.linalg.norm(self.exemplars[i] - self.queries[j], axis=1)**2
            self.vd_exemplar.append(vd_dist)
            
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
        
        print("Pre-processing Complete!")

        
    def run_categorization(self, models=['onenn', 'exemplar', 'prototype'], prior='uniform', verbose=False):
        
        # Preprocess
        self.preprocess(models, verbose=verbose)
        print("Optimizing Kernels...")
        time.sleep(0.5)
        
        # Fork - run_model
        for i in trange(len(models)):
            model = models[i]
            if model == 'prior':
                self.run_prior(None, prior)
            if model == 'onenn':
                self.run_onenn(None, self.train_inds, self.test_inds, prior=prior, verbose=verbose)
            if model == 'exemplar':
                self.run_exemplar(None, self.train_inds, self.test_inds, prior=prior, verbose=verbose)
            if model == 'prototype':
                self.run_prototype(None, self.train_inds, self.test_inds, prior=prior, verbose=verbose)
            cf_match = cfre.search(model)
            if cf_match is not None:
                self.run_cf(None, self.train_inds, self.test_inds, cf_match['model'], int(cf_match['k'])+1, prior=prior, verbose=verbose)
           
        self.save_parameters()
        
    def run_categorization_batch(self, models=['onenn', 'exemplar', 'prototype'], prior='uniform', j=4, verbose=False):
        # Preprocess
        self.preprocess(models, verbose=verbose)
        print("Optimizing Kernels...")
        time.sleep(0.5)
        
        # Fork - run_model
        self.kill_all_processes()
        self.processes = {}
        
        q = multiprocessing.Queue()
        
        for model in models:
            if model == 'prior':
                self.processes[model] = multiprocessing.Process(target=self.run_prior, args=[q, prior])
            if model == 'onenn':
                self.processes[model] = multiprocessing.Process(target=self.run_onenn, args=[q, self.train_inds, self.test_inds, prior])
            if model == 'exemplar':
                self.processes[model] = multiprocessing.Process(target=self.run_exemplar, args=[q, self.train_inds, self.test_inds, prior])
            if model == 'prototype':
                self.processes[model] = multiprocessing.Process(target=self.run_prototype, args=[q, self.train_inds, self.test_inds, prior])
            cf_match = cfre.search(model)
            if cf_match is not None:
                self.processes[model] = multiprocessing.Process(target=self.run_cf, args=[q, self.train_inds, self.test_inds, cf_match['model'], int(cf_match['k'])+1, prior])
        
        procs = list(self.processes.items())
        N_proc = len(procs)
        c_fork = j
        for key, p in procs[:j]:
            if verbose:
                print('Starting Process: '+key)
            p.start()
            
        # Periodically Check Completion
        closed = set()
        for i in trange(N_proc):
            waiting = True
            while waiting:
                for key, p in procs[:c_fork]:
                    if not p.is_alive():
                        if key not in closed:
                            closed.add(key)
                            if c_fork < N_proc:
                                if verbose:
                                    print('Starting Process: '+procs[c_fork][0])
                                procs[c_fork][1].start()
                                c_fork += 1
                            waiting=False
                            break
                if not waiting:
                    break
                time.sleep(5)
                
        # Read return value, update parameters
        for key, p in procs:
            p.join()
            
        for i in range(N_proc):
            res = q.get()
            self.parameters[res[0]] = res[1]
            
        self.save_parameters()
        
            
    def kill_all_processes(self):
        for key, p in self.processes.items():
            if p.is_alive():
                p.kill()           
                
    def run_onenn(self, q, train_ind, test_ind, prior='uniform', verbose=False):
        kernel = lambda params,inds,verbose: self.search_kernel(self.search_onenn, params, [inds], prior_name=prior, verbose=verbose)
        self.run_model(kernel, 'onenn', [1], ((10**-2, 10**2),), train_ind, test_ind, q, prior_name=prior, verbose=verbose)
        
    def run_exemplar(self, q, train_ind, test_ind, prior='uniform', verbose=False):
        kernel = lambda params,inds,verbose: self.search_kernel(self.search_exemplar, params, [inds], prior_name=prior, verbose=verbose)
        self.run_model(kernel, 'exemplar', [1], ((10**-2, 10**2),), train_ind, test_ind, q, prior_name=prior, verbose=verbose)
    
    def run_prototype(self, q, train_ind, test_ind, prior='uniform', verbose=False):
        kernel = lambda params,inds,verbose: self.search_kernel(self.search_prototype, params, [inds], prior_name=prior, verbose=verbose)
        self.run_model(kernel, 'prototype', [1], ((10**-2, 10**2),), train_ind, test_ind, q, prior_name=prior, verbose=verbose)
      
    def run_prior(self, q, prior_name='uniform'):
        l_prior = normalize(self.prior[prior_name], axis=1)
        
        np.save(self.data_dir+'l_prior_'+prior_name+'_train.npy', l_prior[self.train_inds])
        np.save(self.data_dir+'l_prior_'+prior_name+'_test.npy', l_prior[self.test_inds])

        if q is not None:
            q.put(['prior', {}])
        
    def run_cf(self, q, train_ind, test_ind, model, k, prior='uniform', verbose=False):
        kernel = lambda params,inds,verbose: self.search_kernel(self.search_cf, params, [inds, model, k, prior], prior_name=prior, verbose=verbose)
        cf_init = [1,1] + (self.CF_dim-1) * [1.0/self.CF_dim]
        cf_bounds = [(10**-2, 10**2),(10**-2, 10**2)] + (self.CF_dim-1) * [(0,1)]
        if self.CF_dim == 1:
            cf_constraints = None
        else:
            cf_constraints = {'type': 'ineq', 'fun': lambda x: 1.0 - np.sum(x[2:]) }
        self.run_model(kernel, 'cf_'+model+'_'+str(k-1), cf_init, cf_bounds, train_ind, test_ind, q, prior_name=prior, constraints=cf_constraints, verbose=verbose)

    def run_model(self, kernel, ker_name, init, bounds, train_ind, test_ind, q, prior_name='uniform', constraints=None, verbose=False):
        
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
        
        if q is not None:
            q.put([ker_name, result.x])
        else:
            self.parameters[ker_name] = result.x
    
    def search_kernel(self, likelihood_func, likelihood_params, likelihood_args, prior_name='uniform', verbose=True):
        inds = likelihood_args[0]
        N = inds.shape[0]
        
        l_likelihood = likelihood_func(likelihood_params, likelihood_args)
        l_prior = self.prior[prior_name][inds]
        
        l_posterior = normalize(l_likelihood * l_prior, axis=1)
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
            l_exemplar[:,j] = np.mean(np.exp(self.vd_exemplar[j][inds] / h ), axis=1)
        
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
            alphas = np.concatenate([params[2:], [1-np.sum(params[2:])]])[:, np.newaxis, np.newaxis]
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
        
        # Use Log Prior for CF models
        #l_prior = self.prior[prior_name]
        #l_model = normalize(l_likelihood * l_prior[inds], axis=1)
        l_model = normalize(l_likelihood, axis=1)
        
        cf_feats_weighted = np.sum(self.cf_feats**2 * alphas, axis=0)
        
        neighbors = np.zeros((self.N_cat, k), dtype=np.int32)
        for i in range(self.N_cat):
            neighbors[i,:] = np.argsort(cf_feats_weighted[i,:])[:k]
            
        vd_vocab = np.exp(-1*cf_feats_weighted/h_word)
        
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
    
    def get_roc(self, rankings):
        roc = np.zeros(self.N_cat+1)
        for rank in rankings:
            roc[rank]+=1
        for i in range(1,self.N_cat+1):
            roc[i] = roc[i] + roc[i-1]
        return roc / rankings.shape[0]
        
    
    def compute_results(self, models=['onenn', 'exemplar', 'prototype'], prior='uniform'):
        
        self.results['random'] = {'nll_train': log_likelihood(np.ones(self.train_inds.shape[0])/self.N_cat), \
                          'nll_test': log_likelihood(np.ones(self.test_inds.shape[0])/self.N_cat), \
                          'rank_train': self.N_cat/2.0, \
                          'rank_test': self.N_cat/2.0, \
                          'roc_train': np.asarray([(i/float(self.N_cat)) for i in range(self.N_cat+1)]), \
                          'roc_test': np.asarray([(i/float(self.N_cat)) for i in range(self.N_cat+1)])}
        
        for model in models:
            self.results[model] = {}
            
            l_model_train = np.load(self.data_dir+'l_'+model+'_'+prior+'_train.npy')
            train_rankings = self.get_rankings(l_model_train, self.train_inds)
            self.results[model]['nll_train'] = log_likelihood(l_model_train[np.arange(self.train_inds.shape[0]), self.query_labels[self.train_inds]])
            self.results[model]['rank_train'] = np.mean(train_rankings)
            self.results[model]['roc_train'] = self.get_roc(train_rankings)
            
            l_model_test = np.load(self.data_dir+'l_'+model+'_'+prior+'_test.npy')
            test_rankings = self.get_rankings(l_model_test, self.test_inds)
            self.results[model]['nll_test'] = log_likelihood(l_model_test[np.arange(self.test_inds.shape[0]), self.query_labels[self.test_inds]])
            self.results[model]['rank_test'] = np.mean(test_rankings)
            self.results[model]['roc_test'] = self.get_roc(test_rankings)
    
    def summarize_model(self, model):
        print('['+model.upper()+']')
        print("Log_Likelihood (Train): " + str(self.results[model]['nll_train']))
        print("Log_Likelihood (Test): " + str(self.results[model]['nll_test']))
        print("AUC (Train): " + str(np.mean(self.results[model]['roc_train'])))
        print("AUC (Test): " + str(np.mean(self.results[model]['roc_test'])))
        print("Expected_Rank (Train): " + str(self.results[model]['rank_train']))
        print("Expected_Rank (Test): " + str(self.results[model]['rank_test']))
    
    def summarize(self, models=['onenn', 'exemplar', 'prototype'], prior='uniform'):
        
        self.summarize_model('random')
        
        for model in models:
            self.summarize_model(model)
        