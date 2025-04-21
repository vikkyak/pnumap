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
.. code:: bash

    pip install git+https://github.com/vikkyak/pnumap.git

PNUMAP depends similarly to UMAP upon ``scikit-learn`` and its dependencies
such as ``numpy`` and ``scipy``. It adds a requirement for ``numba`` for
performance reasons.

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

-------------------
How to use PNUMAP
-------------------

PossNessUMAP inherits from sklearn classes, and thus drops in neatly
next to other sklearn transformers with an identical calling API.

.. code:: python

    import pnumap
    from sklearn.datasets import load_digits

    digits = load_digits()

    embedding = pnumap.PossNessUMAP().fit_transform(digits.data)

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

=============================
Basic Example with Digits Dataset
=============================

.. code:: python

    from sklearn.datasets import load_digits
    from sklearn.preprocessing import scale
    from pnumap import PossNessUMAP
    import matplotlib.pyplot as plt

    digits = load_digits()
    X = scale(digits.data)
    y = digits.target

    reducer = PossNessUMAP(random_state=42)
    embedding = reducer.fit_transform(X)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='Spectral', s=5)
    plt.colorbar(boundaries=range(11))
    plt.title('PossNessUMAP projection of the Digits dataset')
    plt.show()

===================================
Quickstart: MNIST with PossNessUMAP
===================================

.. code:: bash

    pip install pnumap scikit-learn matplotlib pandas

.. code:: python

    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    from pnumap import PossNessUMAP
    import pandas as pd
    import matplotlib.pyplot as plt

    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data / 255.0
    y = pd.Series(mnist.target)
    X_scaled = StandardScaler().fit_transform(X)

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

    y_codes = y.astype('category').cat.codes
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y_codes, cmap='Spectral', s=5)
    plt.colorbar()
    plt.title('PossNessUMAP projection of the MNIST dataset')
    plt.show()

==================================================
Suppressing Warnings and Comparing with UMAP, t-SNE, PCA
==================================================

.. code:: python

    import warnings
    import os
    import sys
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from umap import UMAP
    from pnumap import PossNessUMAP

    class SuppressStdErr:
        def __enter__(self):
            self._stderr = sys.stderr
            self._devnull = open(os.devnull, 'w')
            sys.stderr = self._devnull
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stderr = self._stderr
            self._devnull.close()

    warnings.filterwarnings("ignore")

    with SuppressStdErr():
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X = mnist.data / 255.0
        y = pd.Series(mnist.target)
        X_scaled = StandardScaler().fit_transform(X)
        y_codes = y.astype('category').cat.codes

        reducers = {
            "PossNessUMAP": PossNessUMAP(alpha=2.0, beta=1.0, sharpness=5.0, random_state=42),
            "UMAP": UMAP(random_state=42),
            "t-SNE": TSNE(n_components=2, random_state=42),
            "PCA": PCA(n_components=2)
        }

        embeddings = {name: reducer.fit_transform(X_scaled) for name, reducer in reducers.items()}
        scores = {name: silhouette_score(embed, y_codes) for name, embed in embeddings.items()}

    plt.figure(figsize=(12, 10))
    for i, (name, embedding) in enumerate(embeddings.items()):
        plt.subplot(2, 2, i + 1)
        plt.scatter(embedding[:, 0], embedding[:, 1], c=y_codes, cmap='Spectral', s=5)
        plt.title(f'{name} (Silhouette = {scores[name]:.3f})')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()



[console_scripts]
pnumap = pnumap.__main__:main

