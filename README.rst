.. -*- mode: rst -*-

====
PossNessUMAP
====

PossNessUMAP is a novel modification of UMAP that incorporates possibility–necessity theory into the construction of neighborhood graphs and embeddings.
This extension aims to provide more expressive uncertainty modeling while retaining UMAP's scalability and interpretability.

**Core Modifications:**

- Replaces fuzzy membership strength with possibility–necessity-based similarity.
- Replaces `compute_membership_strengths` with `compute_membership_strengths_possibility_necessity`.
- Replaces `smooth_knn_dist` with `smooth_knn_dist_possibility_necessity`.
- Adds tunable parameters (`alpha`, `beta`, `sharpness`) for precise control of edge weights.
- Fully compatible with UMAP APIs; supports seamless drop-in replacement.

===
UMAP Overview
===
Uniform Manifold Approximation and Projection (UMAP) is a non-linear dimension reduction technique useful for both visualization and general-purpose reduction. UMAP relies on the following assumptions:

1. The data lies on a Riemannian manifold.
2. The Riemannian metric is locally constant.
3. The manifold is locally connected.

From these assumptions, UMAP models the data's structure using a fuzzy topological representation and finds a low-dimensional embedding with equivalent structure.

----------
Installing
----------
To install PossNessUMAP:

.. code:: bash

    pip install git+https://github.com/vikkyak/pnumap.git

**Requirements:**

* Python >= 3.6
* numpy
* scipy
* scikit-learn
* numba
* tqdm
* `pynndescent <https://github.com/lmcinnes/pynndescent>`_

**Recommended (for plotting):**

* matplotlib
* datashader
* holoviews

**For Parametric PNUMAP (optional):**

* tensorflow >= 2.0.0

----------------------
How to Use PossNessUMAP
----------------------

PossNessUMAP integrates seamlessly with the scikit-learn API:

.. code:: python

    import pnumap
    from sklearn.datasets import load_digits

    digits = load_digits()
    embedding = pnumap.PossNessUMAP().fit_transform(digits.data)

**Key Parameters (same as UMAP):**

- ``n_neighbors``: Number of neighbors used in local manifold approximation. [5–50, default: 15]
- ``min_dist``: Controls how tightly points are packed in low-dimensional space. [0.001–0.5, default: 0.1]
- ``metric``: Distance metric (e.g., 'euclidean', 'cosine').

=========================
PossNessUMAP Parameters
=========================

+------------+------------------+-------------------------------------------------------------+
| Parameter  | Typical Range    | Description                                                 |
+============+==================+=============================================================+
| alpha      | 0.5 to 10        | Controls possibility influence; higher values → possibility dominates. |
+------------+------------------+-------------------------------------------------------------+
| beta       | 0.1 to 2         | Controls necessity influence; higher values → necessity dominates.     |
+------------+------------------+-------------------------------------------------------------+
| sharpness  | 1 to 50 (log)    | Controls confidence decay steepness (soft → hard gating).           |
+------------+------------------+-------------------------------------------------------------+

-----------------------------
Basic Example: Digits Dataset
-----------------------------

PossNessUMAP can be used as a direct replacement for UMAP. Here's a simple example:

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

---------------------------------
Quickstart: MNIST with PossNessUMAP
---------------------------------

Install dependencies:

.. code:: bash

    pip install pnumap scikit-learn matplotlib pandas

Then try:

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

