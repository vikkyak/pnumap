.. -*- mode: rst -*-

====
PossNessUMAP
====

PossNessUMAP is a modification of UMAP incorporating possibility–necessity theory into neighborhood graphs and embeddings. 
It extends UMAP for enhanced uncertainty modeling.

PossNessUMAP modifies UMAP in the following ways:

- Replaces fuzzy membership strength with possibility–necessity-based similarity.
- Replaces compute_membership_strengths with compute_membership_strengths_possibility_necessity.
- Replaces smooth_knn_dist with smooth_knn_dist_possibility_necessity.
- Adds tunable parameters (`alpha`, `beta`, `sharpness`) to control edge weights.
- Retains all UMAP APIs and structure for easy drop-in replacement.

===
UMAP
===
Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction
technique that can be used for visualisation similarly to t-SNE, but also for
general non-linear dimension reduction. The algorithm is founded on three
assumptions about the data:

1. The data is uniformly distributed on a Riemannian manifold;
2. The Riemannian metric is locally constant (or can be approximated as such);
3. The manifold is locally connected.

From these assumptions it is possible to model the manifold with a fuzzy
topological structure. The embedding is found by searching for a low dimensional
projection of the data that has the closest possible equivalent fuzzy
topological structure.



----------
Installing
----------
pip install git+https://github.com/vikkyak/pnumap.git

PNUMAP depends similar to UMAP upon ``scikit-learn``, and thus ``scikit-learn``'s dependencies
such as ``numpy`` and ``scipy``. UMAP adds a requirement for ``numba`` for
performance reasons. The original version used Cython, but the improved code
clarity, simplicity and performance of Numba made the transition necessary.

Requirements:

* Python 3.6 or greater
* numpy
* scipy
* scikit-learn
* numba
* tqdm
* `pynndescent <https://github.com/lmcinnes/pynndescent>`_

Recommended packages:

* For plotting
   * matplotlib
   * datashader
   * holoviews
* for Parametric PNUMAP
   * tensorflow > 2.0.0


---------------
How to use PNUMAP
---------------

The umap package inherits from sklearn classes, and thus drops in neatly
next to other sklearn transformers with an identical calling API.

.. code:: python

    import pnumap
    from sklearn.datasets import load_digits

    digits = load_digits()

    embedding = pnumap.PossNessUMAP().fit_transform(digits.data)

There are a number of parameters that can be set for the UMAP class; the
major ones are as follows:

 -  ``n_neighbors``: This determines the number of neighboring points used in
    local approximations of manifold structure. Larger values will result in
    more global structure being preserved at the loss of detailed local
    structure. In general this parameter should often be in the range 5 to
    50, with a choice of 10 to 15 being a sensible default.

 -  ``min_dist``: This controls how tightly the embedding is allowed compress
    points together. Larger values ensure embedded points are more evenly
    distributed, while smaller values allow the algorithm to optimise more
    accurately with regard to local structure. Sensible values are in the
    range 0.001 to 0.5, with 0.1 being a reasonable default.

 -  ``metric``: This determines the choice of metric used to measure distance
    in the input space. A wide variety of metrics are already coded, and a user
    defined function can be passed as long as it has been JITd by numba.

=========================
PossNessUMAP Parameters
=========================

+------------+------------------+-------------------------------------------------------------+
| Parameter  | Typical Range    | Description                                                 |
+============+==================+=============================================================+
| alpha      | 0.5 to 10        | Controls possibility influence; higher values → possibility |
|            |                  | dominates.                                                  |
+------------+------------------+-------------------------------------------------------------+
| beta       | 0.1 to 2         | Controls necessity influence; higher values → necessity     |
|            |                  | dominates.                                                  |
+------------+------------------+-------------------------------------------------------------+
| sharpness  | 1 to 50 (log)    | Controls confidence decay steepness (soft → hard gating).   |
+------------+------------------+-------------------------------------------------------------+



An example of making use of these options:

.. code:: python

    import pnumap
    from sklearn.datasets import load_digits

    digits = load_digits()

    embedding = PossNessUMAP(n_neighbors=5,
                          min_dist=0.3,
                          metric='correlation').fit_transform(digits.data)

UMAP also supports fitting to sparse matrix data. For more details
please see `the UMAP documentation <https://umap-learn.readthedocs.io/>`_




.. code:: python

    import pnumap
    import pnumap.plot
    from sklearn.datasets import load_digits

    digits = load_digits()

    mapper = PossNessUMAP().fit(digits.data)
    pnumap.plot.points(mapper, labels=digits.target)

The plotting package offers basic plots, as well as interactive plots with hover
tools and various diagnostic plotting options. See the documentation for more details.




An example of making use of these options (based on a subsample of the mnist_784 dataset):

.. code:: python

   =============================
Quickstart: MNIST with PossNessUMAP
=============================

Install dependencies:

.. code:: bash

    pip install pnumap scikit-learn matplotlib pandas

Then try the following example:

.. code:: python

    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    from pnumap import PossNessUMAP
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load and preprocess MNIST
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data / 255.0  # Normalize pixel values to [0, 1]
    y = pd.Series(mnist.target)
    X_scaled = StandardScaler().fit_transform(X)  # Standard scaling

    # Initialize and fit PossNessUMAP
    reducer = PossNessUMAP(
        n_neighbors=15,
        n_components=2,
        metric='euclidean',
        sharpness=5.0,
        alpha=2.0,
        beta=1.0,
        random_state=42
    )
    embedding = reducer.fit_transform(X_scaled)

    # Encode labels and plot
    y_codes = y.astype('category').cat.codes
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y_codes, cmap='Spectral', s=5)
    plt.colorbar()
    plt.title('PossNessUMAP projection of the MNIST dataset')
    plt.show()





