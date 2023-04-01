import copy

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import time
from sklearn.model_selection import train_test_split
from imodels import get_clean_dataset, HSTreeRegressorCV
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from datasets import DATASETS_REGRESSION




def cv_ccp(tree, X_train, y_train):
    path = tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    cv_scores = []
    for ccp_alpha in ccp_alphas:
        tree = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha)
        scores = cross_val_score(tree, X_train, y_train, cv=5)
        cv_scores.append(scores.mean())
    best_alpha = ccp_alphas[np.argmax(cv_scores)]
    final_reg = DecisionTreeRegressor(random_state=42, ccp_alpha=best_alpha)
    final_reg.fit(X_train, y_train)
    return final_reg


def main():
    # get the data
    performance = {"ccp":[], "tv": []}
    running_time = {"ccp":[], "tv": []}
    data_sizes = []

    for d in DATASETS_REGRESSION:
        X, y, feat_names = get_clean_dataset(d[1], data_source=d[2])
        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        data_sizes.append(X_train.shape[0])
        # fit a decision tree
        tree = DecisionTreeRegressor(random_state=42)
        tree.fit(X_train, y_train)
        # get cart test rmse
        mse_cart = mean_squared_error(y_test, tree.predict(X_test))
        # perform ccp pruning and get the time and test performance (mean squared error)
        t0 = time.time()
        tree_ccp = cv_ccp(tree, X_train, y_train)
        ccp_time = time.time() - t0
        running_time['ccp'].append(ccp_time)
        mse_ccp = mean_squared_error(y_test, tree_ccp.predict(X_test)) / mse_cart
        performance['ccp'].append(mse_ccp)
        # do tv shrinkage and get time and test performance
        t2 = time.time()
        tree_tv = HSTreeRegressorCV(copy.deepcopy(tree), shrinkage_scheme_="tv")
        tv_shrink_time = time.time() - t2
        running_time['tv'].append(tv_shrink_time)
        mse_tv = mean_squared_error(y_test, tree_tv.predict(X_test)) / mse_cart
        performance['tv'].append(mse_tv)

    # save the performance and running time to a single json file
    import json
    with open('ccp_expr.json', 'w') as f:
        json.dump({"performance": performance, "running_time": running_time, "data_sizes": data_sizes}, f)


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