from functools import partial

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from imodels import (
    GreedyTreeClassifier, GreedyTreeRegressor, HSTreeClassifierCV, HSTreeRegressorCV
)
from util import ModelConfig

RANDOM_FOREST_DEFAULT_KWARGS = {'random_state': 0}
ESTIMATORS_CLASSIFICATION = [
    [ModelConfig('CART', GreedyTreeClassifier, 'max_depth', n)
     for n in [1, 2, 3, 5, 7, 10]],
    [ModelConfig('HSCART_Ridge', partial(HSTreeClassifierCV, estimator=DecisionTreeClassifier(max_depth=n)),
                 'max_depth', n, other_params={"shrinkage_scheme_":"ridge"})
     for n in [1, 2, 3, 5, 7, 10]],
    [ModelConfig('HSCART_TV', partial(HSTreeClassifierCV, estimator=DecisionTreeClassifier(max_depth=n)),
                 'max_depth', n, other_params={"shrinkage_scheme_": "tv"})
     for n in [1, 2, 3, 5, 7, 10]],
    [ModelConfig('Random_Forest', RandomForestClassifier, 'n_estimators', n, other_params=RANDOM_FOREST_DEFAULT_KWARGS)
     for n in [3, 10, 25, 50]],
    [ModelConfig('Gradient_Boosting', GradientBoostingClassifier, 'n_estimators', n,
                 other_params=RANDOM_FOREST_DEFAULT_KWARGS)
     for n in [10, 50, 100]],
]

ENSEMBLE_ESTIMATOR_NUMS = [3, 10, 25, 50]
TREE_DEPTHS = [1, 2, 3, 4, 5, 7, 8, 10, 15, 20, 25]
LEAF_RANGE = [3, 5, 10,15, 20]
ESTIMATORS_REGRESSION = [
    [ModelConfig('CART_(MSE)', DecisionTreeRegressor, 'max_leaf_nodes', n)
     for n in LEAF_RANGE],
    # [Model('CART_(MAE)', GreedyTreeRegressor, 'max_depth', n, other_params={'criterion': 'absolute_error'})
    #  for n in TREE_DEPTHS],
    [ModelConfig('HSCART_Ridge', partial(HSTreeRegressorCV, estimator=DecisionTreeRegressor(max_leaf_nodes=n)),
                 'max_leaf_nodes', n, other_params={"shrinkage_scheme_":"ridge"})
     for n in LEAF_RANGE],
    [ModelConfig('HSCART_TV', partial(HSTreeRegressorCV, estimator=DecisionTreeRegressor(max_leaf_nodes=n)),
                 'max_leaf_nodes', n, other_params={"shrinkage_scheme_": "tv"})
     for n in LEAF_RANGE]
]

"""
# gosdt experiments
from imodels import OptimalTreeClassifier
from imodels.experimental import HSOptimalTreeClassifierCV

ESTIMATORS_CLASSIFICATION = [
    [ModelConfig("OptimalTreeClassifier", OptimalTreeClassifier, "regularization", 0.3)],
    [ModelConfig("HSOptimalTreeClassifierCV", HSOptimalTreeClassifierCV, "reg_param", r)
     for r in np.arange(0, 0.0051, 0.001)]
]

# bart experiments
from imodels.experimental.bartpy import BART, HSBARTCV
ESTIMATORS_CLASSIFICATION = [
    [ModelConfig("BART", BART,
                 other_params={"classification": True, "n_trees": 30, "n_samples": 100, "n_chains": 4})],
    [ModelConfig("HSBARTCV", HSBARTCV)]
]

ESTIMATORS_REGRESSION = [
    [ModelConfig("BART", BART,
                 other_params={"classification": False, "n_trees": 30, "n_samples": 100, "n_chains": 4})]
]
"""