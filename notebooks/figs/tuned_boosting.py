import logging
import matplotlib as mpl
import numpy as np
import os
import dvu
from matplotlib import pyplot as plt
from os.path import join as oj
from os.path import dirname

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score

from imodels import FIGSRegressor, FIGSClassifier, get_clean_dataset

# from config.figs.datasets import DATASETS_REGRESSION, DATASETS_CLASSIFICATION
DATASETS_CLASSIFICATION = [
    # classification datasets from original random forests paper
    # page 9: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
    ("sonar", "sonar", "pmlb"),
    ("heart", "heart", 'imodels'),
    ("breast-cancer", "breast_cancer", 'imodels'), # this is the wrong breast-cancer dataset (https://new.openml.org/search?type=data&sort=runs&id=13&status=active)
    ("haberman", "haberman", 'imodels'),
    ("ionosphere", "ionosphere", 'pmlb'),
    ("diabetes", "diabetes", "pmlb"),
    # ("liver", "8", "openml"), # note: we omit this dataset bc it's label was found to be incorrect (see caveat here: https://archive.ics.uci.edu/ml/datasets/liver+disorders#:~:text=The%207th%20field%20(selector)%20has%20been%20widely%20misinterpreted%20in%20the%20past%20as%20a%20dependent%20variable%20representing%20presence%20or%20absence%20of%20a%20liver%20disorder.)
    # ("credit-g", "credit_g", 'imodels'), # like german-credit, but more feats
    ("german-credit", "german", "pmlb"),

    # clinical-decision rules
    # ("iai-pecarn", "iai_pecarn.csv", "imodels"),

    # popular classification datasets used in rule-based modeling / fairness
    # page 7: http://proceedings.mlr.press/v97/wang19a/wang19a.pdf
    ("juvenile", "juvenile_clean", 'imodels'),
    ("recidivism", "compas_two_year_clean", 'imodels'),
    ("credit", "credit_card_clean", 'imodels'),
    ("readmission", 'readmission_clean', 'imodels'),  # v big
]

DATASETS_REGRESSION = [
    # leo-breiman paper random forest uses some UCI datasets as well
    # pg 23: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
    ('friedman1', 'friedman1', 'synthetic'),
    ('friedman2', 'friedman2', 'synthetic'),
    ('friedman3', 'friedman3', 'synthetic'),
    ("diabetes-regr", "diabetes", 'sklearn'),
    ('abalone', '183', 'openml'),
    ("echo-months", "1199_BNG_echoMonths", 'pmlb'),
    ("satellite-image", "294_satellite_image", 'pmlb'),
    ("california-housing", "california_housing", 'sklearn'),  # this replaced boston-housing due to ethical issues
    ("breast-tumor", "1201_BNG_breastTumor", 'pmlb'),  # this one is v big (100k examples)

]
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

dvu.set_style()
mpl.rcParams['figure.dpi'] = 250

cb2 = '#66ccff'
cb = '#1f77b4'
cr = '#cc0000'
cp = '#cc3399'
cy = '#d8b365'
cg = '#5ab4ac'

DIR_FIGS = oj(dirname(os.path.realpath(__file__)), 'figures')
DSET_METADATA = {'sonar': (208, 60), 'heart': (270, 15), 'breast-cancer': (277, 17), 'haberman': (306, 3),
                 'ionosphere': (351, 34), 'diabetes': (768, 8), 'german-credit': (1000, 20), 'juvenile': (3640, 286),
                 'recidivism': (6172, 20), 'credit': (30000, 33), 'readmission': (101763, 150), 'friedman1': (200, 10),
                 'friedman2': (200, 4), 'friedman3': (200, 4), 'abalone': (4177, 8), 'diabetes-regr': (442, 10),
                 'california-housing': (20640, 8), 'satellite-image': (6435, 36), 'echo-months': (17496, 9),
                 'breast-tumor': (116640, 9), "vo_pati": (100, 100), "radchenko_james": (300, 50),
                 'tbi-pecarn': (42428, 121), 'csi-pecarn': (3313, 36), 'iai-pecarn': (12044, 58),
                 }

COLORS = {
    'FIGS': 'black',
    'CART': 'orange',  # cp,
    'Rulefit': 'green',
    'C45': cb,
    'CART_(MSE)': 'orange',
    'CART_(MAE)': cg,
    'FIGS_(Reweighted)': cg,
    'FIGS_(Include_Linear)': cb,
    'GBDT-1': cp,
    'GBDT-2': cg,
    'GBDT-3': cb,
    'Dist-GB-FIGS': cg,
    'Dist-RF-FIGS': cp,
    'Dist-RF-FIGS-3': 'green',
    'RandomForest': 'gray',
    'GBDT': 'black',
    'BFIGS': 'green',
    'TAO': cb,
}
def tune_boosting(X, y, budget, is_classification=True):
    gb_model = GradientBoostingClassifier if is_classification else GradientBoostingRegressor
    metric = roc_auc_score if is_classification else r2_score
    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    models_scores = {}
    models = {}
    for n_trees in range(1, int(budget / 2)):
        max_depth = max(int(np.floor(np.log2(budget / n_trees))), 1)
        LOGGER.info(f"tuning model with {n_trees} trees and max depth {max_depth}")

        model = gb_model(n_estimators=n_trees + 1, max_depth=max_depth)
        model.fit(X_train, y_train)
        models_scores[n_trees] = metric(y_test, model.predict(X_test))
        models[n_trees] = model
    # fit the best model on all the data
    n_trees_best = max(models_scores, key=models_scores.get)
    max_depth = int(np.ceil(np.log2(n_trees_best + 1)))
    model_best = gb_model(n_estimators=n_trees_best + 1, max_depth=max_depth)
    model_best.fit(X, y)
    return model_best


def figs_vs_boosting(X, y, budget,depth, n_seeds=10, only_boosting=False):
    is_classification = len(np.unique(y)) == 2
    metric = roc_auc_score if is_classification else r2_score
    n_estimators = budget // (np.sum([2**i for i in range(depth)]))

    scores = {"figs": [], "boosting": []}

    for _ in range(n_seeds):
        # split to train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        if n_estimators > 0:
            gb_model = GradientBoostingClassifier if is_classification else GradientBoostingRegressor
            gb = gb_model(n_estimators=n_estimators, max_depth=depth)
            gb.fit(X_train, y_train)
            gb_score = metric(y_test, gb.predict(X_test))
            scores["boosting"].append(gb_score)
        else:
            scores["boosting"].append(np.nan)

        if only_boosting:
            continue

        figs_model = FIGSClassifier if is_classification else FIGSRegressor
        figs = figs_model(max_rules=budget)
        figs.fit(X_train, y_train)
        figs_score = metric(y_test, figs.predict(X_test))
        scores["figs"].append(figs_score)


    return scores


def analyze_datasets(datasets, fig_name=None):
    n_cols = 3
    n_rows = int(np.ceil(len(datasets) / n_cols))

    # n_rules_per_tree = np.sum([2**i for i in range(depth)])

    # make a list of multipication of n_rules_per_tree as long as it is less than 20
    budgets = np.arange(5, 21)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    n_seeds = 20
    for i, d in enumerate(datasets):
        if isinstance(d, str):
            dset_name = d
        elif isinstance(d, tuple):
            dset_name = d[0]
        ax = axes[i // n_cols, i % n_cols]
        X, y, feat_names = get_clean_dataset(d[1], data_source=d[2])
        means = {"figs": [], "boosting d1": [], "boosting d2": [], "boosting d3":[]}
        std = {"figs": [], "boosting d1": [], "boosting d2": [], "boosting d3":[]}
        for budget in budgets:
            scores = figs_vs_boosting(X, y, budget=budget, n_seeds=n_seeds, depth=1)
            means["figs"].append(np.mean(scores["figs"]))
            means["boosting d1"].append(np.mean(scores["boosting"]))
            std["figs"].append(np.std(scores["figs"]) / np.sqrt(n_seeds))
            std["boosting d1"].append(np.std(scores["boosting"]) / np.sqrt(n_seeds))
            for d in [2,3]:
                scores = figs_vs_boosting(X, y, budget=budget, n_seeds=n_seeds, depth=d)
                is_na = np.isnan(scores["boosting"]).sum() > 0
                if is_na:
                    # set mean and std to nan
                    means[f"boosting d{d}"].append(np.nan)
                    std[f"boosting d{d}"].append(np.nan)
                    continue
                means[f"boosting d{d}"].append(np.nanmean(scores["boosting"]))
                std[f"boosting d{d}"].append(np.nanstd(scores["boosting"]) / np.sqrt(n_seeds))
        # make plot with error bars vs budget
        ax.errorbar(budgets, means["figs"], yerr=std["figs"], label="FIGS", color=COLORS["FIGS"])
        for depth in [1,2,3]:
            ax.errorbar(budgets, means[f"boosting d{depth}"], yerr=std[f"boosting d{depth}"], label=f"GB (max_depth = {depth})", color=COLORS[f"GBDT-{depth}"])
        ax.set_title(dset_name.capitalize().replace('-', ' ') + f' ($n={DSET_METADATA.get(dset_name, (-1))[0]}$)',
                  fontsize='medium')
        ax.set_xlabel("# of rules")
        ylab = "AUC" if len(np.unique(y)) == 2 else "R2"
        ax.set_ylabel(ylab)
        ax.legend()
    plt.tight_layout()
    # fig_name = "figs_vs_boosting_regression.png" if
    plt.savefig(f"{fig_name}.png")


def main():
    # for depth in [1,2,3]:
    analyze_datasets(DATASETS_REGRESSION, fig_name=f"figs_vs_boosting_regression")
    analyze_datasets(DATASETS_CLASSIFICATION, fig_name=f"figs_vs_boosting_classification")


if __name__ == '__main__':
    main()
