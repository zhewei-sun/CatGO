# Copyright (C) 2019-2021 Zhewei Sun

import pickle
import multiprocessing
import time
import re

import numpy as np
from scipy.optimize import minimize

from tqdm import trange

<<<<<<< HEAD
from .util import log_likelihood, normalize, proto_only
=======
from .util import log_likelihood, normalize
>>>>>>> e73f6c71f4d87e52b5486f33cc36593be6fc5b94

cfre=re.compile(r'cf_(?P<model>.+)_(?P<k>[0-9]+)')
knnre=re.compile(r'knn_(?P<k>[0-9]+)')

class Categorizer:
    
    def __init__(self, categories, exemplars, cf_feats=None):
        
        # Category Names - V
        self.categories = np.asarray(categories)
        
        # Exemplar Vectors for each Category - V * x
        self.exemplars = np.asarray(exemplars)
        
        # Collaborative Filtering Features (distances between all category pairs) - V*V*x
        self.cf_feats = cf_feats
        
        self.parameters = {}
        self.results = {}
        
        self.processes = {}
        
        self.N_cat = self.categories.shape[0]
        
        if cf_feats is not None:
            self.CF_dim = cf_feats.shape[0]
        
        self.prior = {}
        
    def set_datadir(self, data_dir):
        self.data_dir = data_dir
        
    def save_parameters(self):
        with open(self.data_dir+"parameters.pkl","wb") as param_file:
            pickle.dump(self.parameters,param_file)
        
    def load_parameters(self):
        with open(self.data_dir+"parameters.pkl","rb") as param_file:
            self.parameters = pickle.load(param_file)
        
    def add_prior(self, name, l_prior):
        self.prior[name] = l_prior
        
    def preprocess(self, models, verbose=False):
        
        print("Pre-processing Distances...")
        time.sleep(0.5)
        
        N_query = self.queries.shape[0]
        
        if not proto_only(models):
        
            # Pre-compute Exemplar distances
            self.vd_exemplar = []
            for i in trange(self.N_cat):
                vd_dist = np.zeros((N_query, self.exemplars[i].shape[0]))
                for j in range(N_query):
                    vd_dist[j,:] = np.sort(-1*np.linalg.norm(self.exemplars[i] - self.queries[j], axis=1)**2)[::-1]
                self.vd_exemplar.append(vd_dist)

            # Pre-compute 1NN Distances
            self.vd_onenn = np.zeros((N_query, self.N_cat))

            for j in range(self.N_cat):
                self.vd_onenn[:, j] = np.max(self.vd_exemplar[j], axis=1)
            
        # Pre-compute Prototypes
        E = self.queries.shape[1]
        prototypes = np.zeros((self.N_cat, E))
        for i in range(self.N_cat):
            prototypes[i] = np.mean(self.exemplars[i], axis=0)
            
        self.vd_prototype = np.zeros((N_query, self.N_cat))
        
        for i in range(N_query):
            self.vd_prototype[i] = np.linalg.norm(prototypes - self.queries[i], axis=1)
                
        self.vd_prototype = -1*self.vd_prototype**2
        
        print("Pre-processing Complete!")

        
    def run_categorization(self, queries, query_labels, models=['onenn', 'exemplar', 'prototype'], prior='uniform', mode='train', verbose=False):
        
        # Query Vectors - N
        self.queries = np.asarray(queries)

        # Query Labels - N (vocab_inds)
        self.query_labels = np.asarray(query_labels)
        
        if mode != 'train':
            self.load_parameters()
            
        if prior=='uniform':
            self.prior['uniform'] = np.ones((self.queries.shape[0], self.N_cat)) / float(self.N_cat)
        
        # Preprocess
        self.preprocess(models, verbose=verbose)
        print("Optimizing Kernels...")
        time.sleep(0.5)
        
        # Fork - run_model
        for i in trange(len(models)):
            model = models[i]
            knn_match = knnre.search(model)
            if knn_match is not None:
                self.run_knn(None, int(knn_match['k']), prior=prior, mode=mode, verbose=verbose)
            if model == 'prior':
                self.run_prior(None, prior, mode=mode)
            if model == 'onenn':
                self.run_onenn(None, prior=prior, mode=mode, verbose=verbose)
            if model == 'exemplar':
                self.run_exemplar(None, prior=prior, mode=mode, verbose=verbose)
            if model == 'prototype':
                self.run_prototype(None, prior=prior, mode=mode, verbose=verbose)
            cf_match = cfre.search(model)
            if cf_match is not None:
                self.run_cf(None, cf_match['model'], int(cf_match['k'])+1, prior=prior, mode=mode, verbose=verbose)
        
        if mode == 'train':
            self.save_parameters()
        
    def run_categorization_batch(self, queries, query_labels, models=['onenn', 'exemplar', 'prototype'], prior='uniform', mode='train', j=4, verbose=False):
        
        # Query Vectors - N
        self.queries = np.asarray(queries)

        # Query Labels - N (vocab_inds)
        self.query_labels = np.asarray(query_labels)
        
        if mode != 'train':
            self.load_parameters()
            
        if prior=='uniform':
            self.prior['uniform'] = np.ones((self.queries.shape[0], self.N_cat)) / float(self.N_cat)
        
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
                self.processes[model] = multiprocessing.Process(target=self.run_prior, args=[q, prior, mode])
            if model == 'onenn':
                self.processes[model] = multiprocessing.Process(target=self.run_onenn, args=[q, prior, mode])
            if model == 'exemplar':
                self.processes[model] = multiprocessing.Process(target=self.run_exemplar, args=[q, prior, mode])
            if model == 'prototype':
                self.processes[model] = multiprocessing.Process(target=self.run_prototype, args=[q, prior, mode])
            cf_match = cfre.search(model)
            if cf_match is not None:
                self.processes[model] = multiprocessing.Process(target=self.run_cf, args=[q, cf_match['model'], int(cf_match['k'])+1, prior, mode])
        
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
            
        if mode == 'train':
            for i in range(N_proc):
                res = q.get()
                self.parameters[res[0]] = res[1]

            self.save_parameters()
        
            
    def kill_all_processes(self):
        for key, p in self.processes.items():
            if p.is_alive():
                p.kill()           
                
    def run_onenn(self, q, prior='uniform', mode='train', verbose=False):
        kernel = lambda params, verbose: self.search_kernel(self.search_onenn, params, [], prior_name=prior, verbose=verbose)
        self.run_model(kernel, 'onenn', [1], ((10**-2, 10**2),), q, prior_name=prior, mode=mode, verbose=verbose)
        
    def run_knn(self, q, k, prior='uniform', mode='train', verbose=False):
        kernel = lambda params,verbose: self.search_kernel(self.search_knn, params, [k], prior_name=prior, verbose=verbose)
        self.run_model(kernel, 'knn_'+str(k), [1], ((10**-2, 10**2),), q, prior_name=prior, mode=mode, verbose=verbose)
        
    def run_exemplar(self, q, prior='uniform', mode='train', verbose=False):
        kernel = lambda params,verbose: self.search_kernel(self.search_exemplar, params, [], prior_name=prior, verbose=verbose)
        self.run_model(kernel, 'exemplar', [1], ((10**-2, 10**2),), q, prior_name=prior, mode=mode, verbose=verbose)
    
    def run_prototype(self, q, prior='uniform', mode='train', verbose=False):
        kernel = lambda params,verbose: self.search_kernel(self.search_prototype, params, [], prior_name=prior, verbose=verbose)
        self.run_model(kernel, 'prototype', [1], ((10**-2, 10**2),), q, prior_name=prior, mode=mode, verbose=verbose)
      
    def run_prior(self, q, prior_name='uniform', mode='train'):
        l_prior = normalize(self.prior[prior_name], axis=1)
        
        np.save(self.data_dir+'/l_prior_'+prior_name+'_'+mode+'.npy', l_prior)

        if q is not None:
            q.put(['prior', {}])
        
    def run_cf(self, q, model, k, prior='uniform', mode='train', verbose=False):
        kernel = lambda params,verbose: self.search_kernel(self.search_cf, params, [model, k, prior], prior_name=prior, verbose=verbose)
        cf_init = [1,1] + (self.CF_dim-1) * [1.0/self.CF_dim]
        cf_bounds = [(10**-2, 10**2),(10**-2, 10**2)] + (self.CF_dim-1) * [(0,1)]
        if self.CF_dim == 1:
            cf_constraints = None
        else:
            cf_constraints = {'type': 'ineq', 'fun': lambda x: 1.0 - np.sum(x[2:]) }
        self.run_model(kernel, 'cf_'+model+'_'+str(k-1), cf_init, cf_bounds, q, prior_name=prior, constraints=cf_constraints, mode=mode, verbose=verbose)

    def run_model(self, kernel, ker_name, init, bounds, q, prior_name='uniform', constraints=None, mode='train', verbose=False):
        
        if mode == 'train':
            # Minimizer
            if constraints is None:
                result = minimize(lambda x:kernel(x, verbose)[0], init, bounds=bounds)
            else:
                result = minimize(lambda x:kernel(x, verbose)[0], init, bounds=bounds, constraints=constraints)

            # Save Results
            
            p = result.x

            if verbose:
                print("[params = "+str(result.x)+"]")
        else:
            p = self.parameters[ker_name]
        
        nll_train, likelihood_train = kernel(p, verbose)
        np.save(self.data_dir+'/l_'+ker_name+'_'+prior_name+'_'+mode+'.npy', likelihood_train)
            
        if verbose:
            print("Log_Likelihood ("+mode+"): " + str(nll_train))
        
        if mode == 'train':
            if q is not None:
                q.put([ker_name, result.x])
            else:
                self.parameters[ker_name] = result.x
        
        
    
    def search_kernel(self, likelihood_func, likelihood_params, likelihood_args, prior_name='uniform', verbose=True):
        
        N = self.queries.shape[0]
        
        l_likelihood = likelihood_func(likelihood_params, likelihood_args)
        l_prior = self.prior[prior_name]
        
        l_posterior = normalize(l_likelihood * l_prior, axis=1)
        p_posterior = l_posterior[np.arange(N), self.query_labels]
        
        if verbose:
            print("[params = %s]" % str(likelihood_params))
            print("Log_likelihood: %f" % log_likelihood(p_posterior))
        
        return log_likelihood(p_posterior), l_posterior
    
    def search_onenn(self, params, args):
        
        h = params[0]
        
        l_onenn = normalize(np.exp(self.vd_onenn/h), axis=1)
    
        return l_onenn
    
    def search_knn(self, params, args):
        
        h = params[0]
        k = args[0]
        
        N = self.queries.shape[0]
        l_knn = np.zeros((N, self.N_cat))
            
        for j in range(self.N_cat):
            l_knn[:,j] = np.mean(np.exp(self.vd_exemplar[j][:, :k] / h ), axis=1)
    
        return l_knn
    
    def search_exemplar(self, params, args):
        
        h = params[0]
        
        N = self.queries.shape[0]
        l_exemplar = np.zeros((N, self.N_cat))
            
        for j in range(self.N_cat):
            l_exemplar[:,j] = np.mean(np.exp(self.vd_exemplar[j] / h ), axis=1)
        
        return normalize(l_exemplar, axis=1)
    
    def search_prototype(self, params, args):
        
        h = params[0]
            
        l_prototype = normalize(np.exp(self.vd_prototype/h), axis=1)
        
        return l_prototype
    
    def search_cf(self, params, args):
        
        h_model = params[0]
        h_word = params[1]
        if len(params) > 2:
            alphas = np.concatenate([params[2:], [1-np.sum(params[2:])]])[:, np.newaxis, np.newaxis]
        else:
            alphas = np.asarray([1])[:, np.newaxis, np.newaxis]
            
        model = args[0]
        k = args[1]
        prior_name = args[2]
        
        N = self.queries.shape[0]
        
        knn_match = knnre.search(model)
        if knn_match is not None:
            l_likelihood = self.search_knn(params, [args[0], int(knn_match['k'])])
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
    
        
    def get_rankings(self, l_model, query_labels):
        N = l_model.shape[0]
        ranks = np.zeros((N, self.N_cat), dtype=np.int32)
        rankings = np.zeros(N, dtype=np.int32)
        
        for i in range(N):
            ranks[i] = np.argsort(l_model[i])[::-1]
            rankings[i] = ranks[i].tolist().index(query_labels[i])+1
            
        return rankings
    
    def get_roc(self, rankings):
        roc = np.zeros(self.N_cat+1)
        for rank in rankings:
            roc[rank]+=1
        for i in range(1,self.N_cat+1):
            roc[i] = roc[i] + roc[i-1]
        return roc / rankings.shape[0]
        
    
<<<<<<< HEAD
    def compute_metrics(self, query_labels, models=['onenn', 'exemplar', 'prototype'], prior='uniform', mode='train'):
        
        results = {}
        
        results['random'] = {'nll': log_likelihood(np.ones(query_labels.shape[0])/self.N_cat), \
                          'rank': self.N_cat/2.0, \
                          'roc': np.asarray([(i/float(self.N_cat)) for i in range(self.N_cat+1)])}
        
        for model in models:
            results[model] = {}
            
            l_model = np.load(self.data_dir+'l_'+model+'_'+prior+'_'+mode+'.npy')
            rankings = self.get_rankings(l_model, query_labels)
            results[model]['nll'] = log_likelihood(l_model[np.arange(l_model.shape[0]), query_labels])
            results[model]['rank'] = rankings
            results[model]['roc'] = self.get_roc(rankings)
            
        return results
    
    def summarize_model(self, model, results, mode='train'):
        print('['+model.upper()+']')
        print("Log_Likelihood ("+mode+"): " + str(results['nll']))
        print("AUC ("+mode+"): " + str(np.mean(results['roc'])))
        print("Expected_Rank ("+mode+"): " + str(np.mean(results['rank'])))
    
    def summarize_results(self, query_labels, models=['onenn', 'exemplar', 'prototype'], prior='uniform', mode='train'):
        
        results = self.compute_metrics(query_labels, models=models, prior=prior, mode=mode)
        
        self.summarize_model('random', results['random'], mode=mode)
        
        for model in models:
            self.summarize_model(model, results[model], mode=mode)

=======
    def compute_results(self, models=['onenn', 'exemplar', 'prototype'], metrics=['nll', 'auc', 'erank'], prior='uniform'):
        
        self.results['random'] = {'metrics': set(['nll', 'auc', 'erank', 'classify']), \
                          'nll_train': log_likelihood(np.ones(self.train_inds.shape[0])/self.N_cat), \
                          'nll_test': log_likelihood(np.ones(self.test_inds.shape[0])/self.N_cat), \
                          'rank_train': self.N_cat/2.0, \
                          'rank_test': self.N_cat/2.0, \
                          'roc_train': np.asarray([(i/float(self.N_cat)) for i in range(self.N_cat+1)]), \
                          'roc_test': np.asarray([(i/float(self.N_cat)) for i in range(self.N_cat+1)]), \
                          'acc_train': 1.0 / self.N_cat, \
                          'acc_test': 1.0 / self.N_cat }
        
        for model in models:
            if model not in self.results:
                self.results[model] = {'metrics':set()}
            
            l_model_train = np.load(self.data_dir+'l_'+model+'_'+prior+'_train.npy')
            train_rankings = self.get_rankings(l_model_train, self.train_inds)
            
            l_model_test = np.load(self.data_dir+'l_'+model+'_'+prior+'_test.npy')
            test_rankings = self.get_rankings(l_model_test, self.test_inds)
            
            for metric in metrics:
                
                if metric in self.results['random']['metrics']:
                    self.results[model]['metrics'].add(metric)
                else:
                    continue
                
                if metric == 'nll':
                    self.results[model]['nll_train'] = log_likelihood(l_model_train[np.arange(self.train_inds.shape[0]), self.query_labels[self.train_inds]])
                    self.results[model]['nll_test'] = log_likelihood(l_model_test[np.arange(self.test_inds.shape[0]), self.query_labels[self.test_inds]])
        
                if metric == 'erank':
                    self.results[model]['rank_train'] = np.mean(train_rankings)
                    self.results[model]['rank_test'] = np.mean(test_rankings)
            
                if metric == 'auc':
                    self.results[model]['roc_train'] = self.get_roc(train_rankings)
                    self.results[model]['roc_test'] = self.get_roc(test_rankings)
            
                if metric == 'classify':
                    self.results[model]['acc_train'] = np.sum(train_rankings==1) / float(len(self.train_inds))
                    self.results[model]['acc_test'] = np.sum(test_rankings==1) / float(len(self.test_inds))
            

    
    def summarize_model(self, model, metrics):
        
        if model not in self.results:
            print("[Warning] Kernel "+model+" does not exist or has not been optimized.")
            return
        
        if len(metrics) > 0:
            print('['+model.upper()+']')
        
        for metric in metrics:
            if metric not in self.results[model]['metrics']:
                print("[Warning] Metric "+metric+" does not exist or has not been computed.")
                continue
            if metric == 'nll':
                print("Log_Likelihood (Train): " + str(self.results[model]['nll_train']))
                print("Log_Likelihood (Test): " + str(self.results[model]['nll_test']))
            if metric == 'auc':
                print("AUC (Train): " + str(np.mean(self.results[model]['roc_train'])))
                print("AUC (Test): " + str(np.mean(self.results[model]['roc_test'])))
            if metric == 'erank':
                print("Expected_Rank (Train): " + str(self.results[model]['rank_train']))
                print("Expected_Rank (Test): " + str(self.results[model]['rank_test']))
            if metric == 'classify':
                print("Classification Accuracy (Train): " + str(self.results[model]['acc_train']))
                print("Classification Accuracy (Test): " + str(self.results[model]['acc_test']))
    
    def summarize(self, models=['onenn', 'exemplar', 'prototype'], metrics=['nll', 'auc', 'erank'], prior='uniform'):
        
        self.summarize_model('random', metrics)
        
        for model in models:
            self.summarize_model(model, metrics)
        
>>>>>>> e73f6c71f4d87e52b5486f33cc36593be6fc5b94
