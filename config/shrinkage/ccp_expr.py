import copy

import numpy as np
import matplotlib.pyplot as plt

import json
from sklearn.metrics import mean_squared_error
import time
from sklearn.model_selection import train_test_split
from imodels import get_clean_dataset, HSTreeRegressorCV, HSTreeClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from datasets import DATASETS_REGRESSION, DATASETS_CLASSIFICATION

FNAME = "ccp_expr"

def cv_ccp(tree, X_train, y_train):
    path = tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    cv_scores = []
    is_cls = "Classifier" in tree.__class__.__name__
    base_tree = DecisionTreeClassifier if is_cls else DecisionTreeRegressor
    # base_tree = DecisionTreeRegressor
    # get quantiles every 5% of the range of ccp_alphas
    ccp_alphas = np.quantile(path.ccp_alphas, np.arange(0, 1.02, 0.1))
    scoring = "roc_auc" if is_cls else "neg_mean_squared_error"
    # scoring = "neg_mean_squared_error"
    for ccp_alpha in ccp_alphas:
        tree = base_tree(random_state=42, ccp_alpha=ccp_alpha)
        scores = cross_val_score(tree, X_train, y_train, cv=3, scoring=scoring)
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
        # y_pred_proba = estimator.predict_proba(X)[:, 1]
        y_pred_proba = estimator.predict(X)
        return roc_auc_score(y, y_pred_proba)
    else:
        # If the estimator is a regressor, use MSE
        y_pred = estimator.predict(X)
        return mean_squared_error(y, y_pred)


def main():
    # get the data
    performance = {"ccp": {}, "tv": {}}
    running_time = {"ccp": {}, "tv": {}}
    data_sizes = []
    dataset_names = []
    ests = {0: DecisionTreeRegressor, 1: DecisionTreeRegressor}
    ests_shrink = {0: HSTreeRegressorCV, 1: HSTreeRegressorCV}
    for problem_type, datasets in enumerate([DATASETS_REGRESSION, DATASETS_CLASSIFICATION]):
        for d in datasets:
            tv_shrink_times = []
            ccp_times = []
            tv_perf = []
            ccp_perf = []
            X, y, feat_names = get_clean_dataset(d[1], data_source=d[2])

            data_sizes.append(int(0.7 * X.shape[0]))
            dataset_names.append(d[0])
            for seed in range(10):
                # train test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

                # fit a decision tree
                tree = ests[problem_type](random_state=42)  # DecisionTreeRegressor(random_state=42)
                tree.fit(X_train, y_train)
                # calculate area under roc curve
                cart_perf = evaluate_estimator(tree, X_test, y_test)
                # perform ccp pruning and get the time and test performance (mean squared error)
                t0 = time.time()
                tree_ccp = cv_ccp(tree, X_train, y_train)
                ccp_time = time.time() - t0
                ccp_times.append(ccp_time)
                ccp_perf.append(evaluate_estimator(tree_ccp, X_test, y_test) / cart_perf)

                # do tv shrinkage and get time and test performance
                t2 = time.time()
                tree_tv = ests_shrink[problem_type](copy.deepcopy(tree), shrinkage_scheme_="tv")
                tv_shrink_time = time.time() - t2
                tv_shrink_times.append(tv_shrink_time)
                tv_perf.append(evaluate_estimator(tree_tv, X_test, y_test) / cart_perf)
            running_time['tv'][d[0]] = tv_shrink_times
            performance['tv'][d[0]] = tv_perf
            running_time['ccp'][d[0]] = ccp_times
            performance['ccp'][d[0]] = ccp_perf
            # save the performance and running time to a single json file
            import json
            with open(f'{FNAME}.json', 'w') as f:
                json.dump({"performance": performance, "running_time": running_time, "data_sizes": data_sizes,
                           "dataset": dataset_names}, f)

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


def plot_ccp_expr():
    # load running_time, performance and data_sizes from config/shrinkage/ccp_expr.json file
    with open(f'/accounts/campus/omer_ronen/projects/imodels-experiments/{FNAME}.json', 'r') as f:
        data = json.load(f)
    running_time_seeds = data['running_time']
    running_time = {}
    # get the average running time across 10 seeds for both ccp and tv
    running_time['ccp'] = np.mean(np.array(list(running_time_seeds['ccp'].values())), axis=1)
    running_time['tv'] = np.mean(np.array(list(running_time_seeds['tv'].values())), axis=1)
    # do the same for performance
    performance_seeds = data['performance']
    performance = {}
    performance['ccp'] = np.mean(np.array(list(performance_seeds['ccp'].values())), axis=1)
    performance['tv'] = np.mean(np.array(list(performance_seeds['tv'].values())), axis=1)
    # now get the stds for the means (divide by sqrt number of seeds) across 10 seeds for performance
    performance_std = {}
    performance_std['ccp'] = np.std(np.array(list(performance_seeds['ccp'].values())), axis=1) / np.sqrt(10)
    performance_std['tv'] = np.std(np.array(list(performance_seeds['tv'].values())), axis=1) / np.sqrt(10)

    data_sizes = data['data_sizes']
    # data_sizes = 100 * (np.array(data_sizes) / np.max(data_sizes))

    # find nice color for ccp and tv
    import matplotlib
    colors = ["green", "blue"]
    # plot the results

    # two panels scatter plots, one is scatter of running time vs data size and the other is performance per dataset bar plot add stds
    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    # plt performance vs time for both methods on the same plot, make the dot proportional to the size of the dataset
    # ax[0].scatter(data_sizes, np.log(np.array(running_time['ccp']) + 1), label='ccp', c=colors[0])
    # ax[0].scatter(data_sizes, np.log(np.array(running_time['tv']) + 1), label='tv', c=colors[1])
    # add stds for the sctter plot but don't connect the points
    ax[0].errorbar(data_sizes, np.log(np.array(running_time['ccp']) + 1), yerr=performance_std['ccp'], fmt='o',
                   label='ccp', c=colors[0])
    ax[0].errorbar(data_sizes, np.log(np.array(running_time['tv']) + 1), yerr=performance_std['tv'], fmt='o', label='tv',
                   c=colors[1])
    ax[0].set_xlabel('data size')
    ax[0].set_ylabel('running time (log seconds)')
    ax[0].legend()
    ax[0].set_title("Scaling of running time")
    # bar plot of performance per dataset put the two methods side by side and orientate the x ticks add stds
    ax[1].bar(np.arange(len(performance['ccp'])) - 0.2, performance['ccp'], width=0.4, label='ccp', color=colors[0],
              yerr=performance_std['ccp'])
    ax[1].bar(np.arange(len(performance['tv'])) + 0.2, performance['tv'], width=0.4, label='tv', color=colors[1],
              yerr=performance_std['tv'])
    ax[1].set_xticks(np.arange(len(performance['ccp'])))
    ax[1].set_xticklabels(data['dataset'], rotation=90)
    # color the datasets ticks blue if they are clasisfication and red if they are regression
    for i, d in enumerate(data['dataset']):
        if d in [d[0] for d in DATASETS_CLASSIFICATION]:
            ax[1].get_xticklabels()[i].set_color('orange')
        else:
            ax[1].get_xticklabels()[i].set_color('purple')
    # make "MSE or AUC relative to CART" the y label and color MSE in purple and AUC in orange
    ylabel = "MSE or AUC relative to CART"
    # write the y label in latex
    # ax[1].set_ylabel(ylabel)

    label_x = -0.2
    s = 0.3
    ax[1].text(label_x, s + 0.03, r"MSE", color='purple', rotation='vertical', transform=ax[1].transAxes)
    ax[1].text(label_x, s+0.115, r"or", color='black', rotation='vertical', transform=ax[1].transAxes)
    ax[1].text(label_x, s+0.16, r"AUC", color='orange', rotation='vertical', transform=ax[1].transAxes)
    ax[1].text(label_x, s+0.25, r"relative to CART", color='black', rotation='vertical', transform=ax[1].transAxes)



    # ax[1].legend()
    ax[1].set_title("Performance")

    # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    # # plt performance vs time for both methods on the same plot, make the dot proportional to the size of the dataset
    # ax.scatter(np.log(np.array(running_time['ccp'])+1), performance['ccp'], s=data_sizes, label='ccp')
    # ax.scatter(np.log(np.array(running_time['tv'])+1), performance['tv'], s=data_sizes, label='tv')
    # ax.set_xlabel('running time (seconds, log scale)')
    # ax.set_ylabel('test performance (MSE or AUC) relative to CART')
    #
    # ax.legend(loc = "lower right")
    # ax.set_title("Cross validated pruning")
    # # add dotted red line at y=1
    ax[1].axhline(y=1, color='r', linestyle='--', alpha=0.5)
    fig.tight_layout()
    plt.savefig(f"{FNAME}.png", dpi=300)


if __name__ == '__main__':
    main()
    plot_ccp_expr()
