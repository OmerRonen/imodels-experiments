import copy

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import time
from sklearn.model_selection import train_test_split
from imodels import get_clean_dataset, HSTreeRegressorCV
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from datasets import DATASETS_REGRESSION, DATASETS_CLASSIFICATION


def cv_ccp(tree, X_train, y_train):
    path = tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    cv_scores = []
    is_cls = "Classifier" in tree.__class__.__name__
    base_tree = DecisionTreeClassifier if is_cls else DecisionTreeRegressor
    # get 20 alphas evenly spaced between the min and max
    ccp_alphas = np.arange(np.min(ccp_alphas), np.max(ccp_alphas), (np.max(ccp_alphas) - np.min(ccp_alphas)) / 20)

    for ccp_alpha in ccp_alphas:
        tree = base_tree(random_state=0, ccp_alpha=ccp_alpha)
        scores = cross_val_score(tree, X_train, y_train, cv=5)
        cv_scores.append(scores.mean())
    best_alpha = ccp_alphas[np.argmax(cv_scores)]
    final_reg = base_tree(random_state=42, ccp_alpha=best_alpha)
    final_reg.fit(X_train, y_train)
    return final_reg


from sklearn.metrics import mean_squared_error, roc_auc_score


def evaluate_estimator(estimator, X, y):
    """
    Evaluate a scikit-learn estimator and return MSE or AUC depending on the estimator type.

    Parameters:
        estimator (object): A trained scikit-learn estimator.
        X (array-like): Input data to use for prediction.
        y (array-like): Ground truth labels.

    Returns:
        float: Mean squared error (MSE) if the estimator is a regressor, or area under the ROC curve (AUC) if the estimator is a binary classifier.
    """
    if "Classifier" in estimator.__class__.__name__:
        # If the estimator is a binary classifier, use AUC
        y_pred_proba = estimator.predict_proba(X)[:, 1]
        return roc_auc_score(y, y_pred_proba)
    else:
        # If the estimator is a regressor, use MSE
        y_pred = estimator.predict(X)
        return mean_squared_error(y, y_pred)


def main():
    # get the data
    performance = {"ccp": [], "tv": []}
    running_time = {"ccp": [], "tv": []}
    data_sizes = []
    dataset_names = []
    ests = {0: DecisionTreeRegressor, 1: DecisionTreeClassifier}
    for problem_type, datasets in enumerate([DATASETS_REGRESSION, DATASETS_CLASSIFICATION]):
        for d in datasets:
            X, y, feat_names = get_clean_dataset(d[1], data_source=d[2])
            # train test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            data_sizes.append(X_train.shape[0])
            dataset_names.append(d[0])

            # fit a decision tree
            tree = ests[problem_type](random_state=42)  # DecisionTreeRegressor(random_state=42)
            tree.fit(X_train, y_train)
            # calculate area under roc curve
            cart_perf = evaluate_estimator(tree, X_test, y_test)
            # perform ccp pruning and get the time and test performance (mean squared error)
            t0 = time.time()
            tree_ccp = cv_ccp(tree, X_train, y_train)
            ccp_time = time.time() - t0
            running_time['ccp'].append(ccp_time)
            performance['ccp'].append(evaluate_estimator(tree_ccp, X_test, y_test) / cart_perf)
            # do tv shrinkage and get time and test performance
            t2 = time.time()
            tree_tv = HSTreeRegressorCV(copy.deepcopy(tree), shrinkage_scheme_="tv")
            tv_shrink_time = time.time() - t2
            running_time['tv'].append(tv_shrink_time)
            performance['tv'].append(evaluate_estimator(tree_tv, X_test, y_test) / cart_perf)
            # save the performance and running time to a single json file
            import json
            with open('ccp_expr.json', 'w') as f:
                json.dump({"performance": performance, "running_time": running_time, "data_sizes": data_sizes, "dataset": dataset_names}, f)

    # # plot performance vs time for both methods on the same plot, make the dot proportional to the size of the dataset
    # data_sizes = 4* (np.array(data_sizes) / np.max(data_sizes))
    #
    # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    # ax.scatter(running_time['ccp'], performance['ccp'], s=data_sizes, label='ccp')
    # ax.scatter(running_time['tv'], performance['tv'], s=data_sizes, label='tv')
    # ax.set_xlabel('running time (seconds)')
    # ax.set_ylabel('test mse')
    # ax.legend()
    # plt.show()


if __name__ == '__main__':
    main()
