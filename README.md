# CatGO - Python Library for Categorization
CatGO is a light-weight Python library for categorization models.

A full tutorial can be found in the /tutorial directory, where I demo the code on image categorization.

The models implemented in this package are described in detail in the following paper, with one exception that instead of summing all stimuli in the exemplar model, we compute an average.


[Sun et al. (2019) - Slang Generation as Categorization](http://www.cs.toronto.edu/~zheweisun/files/cogsci19_slang.pdf)

Please consider citing the above paper if you find this package to be useful.

### Quick Starter Guide

To use the library, you'll need to install the following dependencies:

- Python 3
- Numpy
- Scipy
- Matplotlib
- tqdm

Most of which should be available in a standard scientific python distribution.

To run the code, simply download this repo and create a symbolic link in your code's home directory:

```
ln -s [Directory of the CatGO repo] CatGO
```

Then simply import the library within your code. No installation required!

```
from CatGO import categorize
```

### Documentation

```
Categorizer(categories, exemplars, queries, query_labels, cf_feats=None)
```

This is the main constructor where all model inputs are passed in:

- *categories* - An ordered array of category names.

- *exemplars* - A nested array of exemplar vectors. The *i*'th element in the array should be an array containing all exemplars corresponding to category *i* as specified in *categories*.

- *queries* - An array of vectors containing all stimuli to be queried. This can be further split into a training set for parameter estimation and a test set for evaluation.

- *query_labels* - An array of integer containing the category labels for all examples in *queries*, corresponding to the order specified in *categories*.

- *cf_feats* - A 3D matrix of category similarity to be used in collaborative filtering. Each sub-matrix should be of size N x N where N is the number of categories, and should specify the distances between all pairs of categories. The variable should be of shape k x N x N, where k is the number of feature maps to be used.

```
categorizer.set_inds(train_inds, test_inds)
```

Specifies indices for training-testing split of the queried examples. Indices should correspond to the input array *queries*.

- *train_inds* - array specifying indices of examples to be used for model estimation.

- *test_inds* - array specifying indices of examples to be used for prediction and evaluation.

```
categorizer.set_datadir(data_dir)
```

Sets the data directory to output the results. For each optimized kernel, two likelihood tables will be generated for both the training and testing examples.

- *data_dir* - string specifying the output directory.

```
categorizer.add_prior(name, l_prior):
```

Adds a custom prior distribution to be applied to the categorization models. By default, CatGO uses a built-in uniform prior.

- *name* - Name of the prior to be used.

- *l_prior* - An array of normalized probabilities for each of the N categories.

```
categorizer.run_categorization(models=['onenn', 'exemplar', 'prototype'], prior='uniform')
```

Optimizes all categorization kernels specified using the training examples from *queries*. The available kernels are:

1. *onenn* - One Nearest Neighbor (1NN) kernel
2. *exemplar* - Exemplar kernel
3. *prototype* - Prototype kernel
4. *cf_onenn_[k]* - Collaboratively filtered 1NN kernel
5. *cf_exemplar_[k]* - Collaboratively filtered Exemplar kernel
6. *cf_prototype_[k]* - Collaboratively filtered Prototype kernel

Where *k* is the number of neighbors to consider in collaborative filtering (e.g. cf_onenn_1, cf_prototype_5).

- *models* - A list of kernels to be optimized.

- *prior* - The prior to use for all categorization models.

```
categorizer.run_categorization_batch(models=['onenn', 'exemplar', 'prototype'], prior='uniform')
```

Same as *categorizer.run_categorization* except kernels are optimized in parallel.

- *models* - A list of kernels to be optimized.

- *prior* - The prior to use for all categorization models.

```
categorizer.compute_results(models=['onenn', 'exemplar', 'prototype'], prior='uniform')
```

Computes the following metrics on training and testing examples from *queries* for the prior and all specified kernels:

1. Negative log likelihood
2. AUC
3. Expected Rank

- *models* - A list of kernels where results needs to be computed. All kernels specified here should be optimized beforehand using *categorizer.run_categorization*.

- *prior* - The prior to use for all categorization models.

```
categorizer.summarize(models=['onenn', 'exemplar', 'prototype'], prior='uniform')
```

Summarizes the results computed in *categorizer.compute_results*.

- *models* - A list of kernels where results needs to be summarized. All kernels specified here should be optimized beforehand using *categorizer.run_categorization* and have results computed using *categorizer.compute_results*.

- *prior* - The prior to use for all categorization models.

