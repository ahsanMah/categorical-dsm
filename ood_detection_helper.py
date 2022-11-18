import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

# from skimage import draw
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report, average_precision_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.neighbors import NearestNeighbors

def result_dict(train_score, test_score, ood_scores, metrics):
    return {
        "train_scores": train_score,
        "test_scores": test_score,
        "ood_scores": ood_scores,
        "metrics": metrics,
    }


def get_metrics(test_score, ood_scores, labels, **kwargs):
    metrics = {}
    for idx, _score in enumerate(ood_scores):
        ood_name = labels[idx + 2]
        metrics[ood_name] = ood_metrics(test_score, _score, names=(labels[1], ood_name))
    metrics_df = pd.DataFrame(metrics).T * 100  # Percentages
    return metrics_df


def auxiliary_model_analysis(
    X_train,
    X_test,
    outliers,
    labels,
    components_range=range(2, 21, 2),
    ica_range=range(2, 8, 2),
    verbose=True,
):
    print("=====" * 5 + " Training GMM " + "=====" * 5)
    best_gmm_clf = train_gmm(
        X_train, components_range=components_range, ica_range=ica_range, verbose=True
    )
    print("---Likelihoods---")
    print("Training: {:.3f}".format(np.median(best_gmm_clf.score_samples(X_train))))
    print("{}: {:.3f}".format(labels[1], np.median(best_gmm_clf.score_samples(X_test))))

    for name, ood in zip(labels[2:], outliers):
        print("{}: {:.3f}".format(name, np.median(best_gmm_clf.score_samples(ood))))

    gmm_train_score = best_gmm_clf.score_samples(X_train)
    gmm_test_score = best_gmm_clf.score_samples(X_test)
    gmm_ood_scores = np.array([best_gmm_clf.score_samples(ood) for ood in outliers])
    gmm_metrics = get_metrics(-gmm_test_score, -gmm_ood_scores, labels)
    gmm_results = result_dict(
        gmm_train_score, gmm_test_score, gmm_ood_scores, gmm_metrics
    )

    print("=====" * 5 + " Training KD Tree " + "=====" * 5)

    N_NEIGHBOURS = 1
    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBOURS, algorithm="kd_tree").fit(X_train)

    kd_train_score, indices = nbrs.kneighbors(X_train)
    kd_train_score = kd_train_score[..., -1]  # Distances to the kth neighbour
    kd_test_score, _ = nbrs.kneighbors(X_test)
    kd_test_score = kd_test_score[..., -1]
    kd_ood_scores = []
    for ood in outliers:
        dists, _ = nbrs.kneighbors(ood)
        kd_ood_scores.append(dists[..., -1])
    kd_metrics = get_metrics(kd_test_score, kd_ood_scores, labels)

    kd_results = result_dict(kd_train_score, kd_test_score, kd_ood_scores, kd_metrics)

    return dict(GMM=gmm_results, KD=kd_results)


def plot_curves(inlier_score, outlier_score, label, axs=()):

    if len(axs) == 0:
        fig, axs = plt.subplots(1, 2, figsize=(16, 4))

    y_true = np.concatenate((np.zeros(len(inlier_score)), np.ones(len(outlier_score))))
    y_scores = np.concatenate((inlier_score, outlier_score))

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, drop_intermediate=True)
    roc_auc = roc_auc = roc_auc_score(y_true, y_scores)

    prec_in, rec_in, _ = precision_recall_curve(y_true, y_scores)
    prec_out, rec_out, _ = precision_recall_curve((y_true == 0), -y_scores)
    pr_auc = auc(rec_in, prec_in)

    ticks = np.arange(0.0, 1.1, step=0.1)
    axs[0].plot(fpr, tpr, label="{}: {:.3f}".format(label, roc_auc))
    axs[0].set(
        xlabel="FPR",
        ylabel="TPR",
        title="ROC",
        ylim=(-0.05, 1.05),
        xticks=ticks,
        yticks=ticks,
    )

    axs[1].plot(rec_in, prec_in, label="{}: {:.3f}".format(label, pr_auc))
    # axs[1].plot(rec_out, prec_out, label="PR-Out")
    axs[1].set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-Recall",
        ylim=(-0.05, 1.05),
        xticks=ticks,
        yticks=ticks,
    )

    axs[0].legend()
    axs[1].legend()

    if len(axs) == 0:
        fig.suptitle("{} vs {}".format(*labels), fontsize=20)
        plt.show()
        plt.close()

    return axs


def ood_metrics(
    inlier_score, outlier_score, plot=False, verbose=False, names=["Inlier", "Outlier"]
):
    import numpy as np
    import seaborn as sns

    y_true = np.concatenate((np.zeros(len(inlier_score)), np.ones(len(outlier_score))))
    y_scores = np.concatenate((inlier_score, outlier_score))

    prec_in, rec_in, _ = precision_recall_curve(y_true, y_scores)

    # Outliers are treated as "positive" class
    # i.e label 1 is now label 0
    prec_out, rec_out, _ = precision_recall_curve((y_true == 0), -y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, drop_intermediate=False)

    # rtol=1e-3 implies range of [0.949, 0.951]
    find_fpr = np.isclose(tpr, 0.95, rtol=1e-3, atol=1e-4).any()

    if find_fpr:
        tpr95_idx = np.where(np.isclose(tpr, 0.95, rtol=1e-3, atol=1e-4))[0][0]
        tpr80_idx = np.where(np.isclose(tpr, 0.8, rtol=1e-2, atol=1e-3))[0][0]
    else:
        # This is becasuse numpy bugs out when the scores are fully separable
        # OR fully unseparable :D
        tpr95_idx = np.where(np.isclose(tpr, 0.95, rtol=1e-1, atol=1e-1))[0][0]
        tpr80_idx = np.where(np.isclose(tpr, 0.8, rtol=1e-1, atol=1e-1))[0][0]
    #         tpr95_idx, tpr80_idx = 0,0 #tpr95_idx

    # Detection Error
    de = np.min(0.5 - tpr / 2 + fpr / 2)

    metrics = dict(
        fpr_tpr95=fpr[tpr95_idx],
        de=de,
        roc_auc=roc_auc_score(y_true, y_scores),
        pr_auc_in=auc(rec_in, prec_in),
        pr_auc_out=auc(rec_out, prec_out),
        fpr_tpr80=fpr[tpr80_idx],
        ap=average_precision_score(y_true, y_scores),
    )

    if plot:

        fig, axs = plt.subplots(1, 2, figsize=(16, 4))
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, drop_intermediate=True)
        ticks = np.arange(0.0, 1.1, step=0.1)

        axs[0].plot(fpr, tpr)
        axs[0].set(
            xlabel="FPR",
            ylabel="TPR",
            title="ROC",
            ylim=(-0.05, 1.05),
            xticks=ticks,
            yticks=ticks,
        )

        axs[1].plot(rec_in, prec_in, label="PR-In")
        axs[1].plot(rec_out, prec_out, label="PR-Out")
        axs[1].set(
            xlabel="Recall",
            ylabel="Precision",
            title="Precision-Recall",
            ylim=(-0.05, 1.05),
            xticks=ticks,
            yticks=ticks,
        )
        axs[1].legend()
        fig.suptitle("{} vs {}".format(*names), fontsize=20)
    #         plt.show()
    #         plt.close()

    if verbose:
        print("{} vs {}".format(*names))
        print("----------------")
        print("ROC-AUC: {:.4f}".format(metrics["roc_auc"] * 100))
        print(
            "PR-AUC (In/Out): {:.4f} / {:.4f}".format(
                metrics["pr_auc_in"] * 100, metrics["pr_auc_out"] * 100
            )
        )
        print("FPR (95% TPR): {:.2f}%".format(metrics["fpr_tpr95"] * 100))
        print("Detection Error: {:.2f}%".format(de * 100))

    return metrics


def evaluate_model(
    train_score, inlier_score, outlier_scores, labels, ylim=None, xlim=None, **kwargs
):
    rows = 1 + int(np.ceil(len(outlier_scores) / 2))
    fig, axs = plt.subplots(rows, 1, figsize=(12, rows * 4))
    axs = np.array(axs).reshape(-1)  # Makes axs into list even if row num is 1
    colors = sns.color_palette("bright") + sns.color_palette("dark")

    sns.histplot(train_score, color=colors[0], label=labels[0], ax=axs[0], **kwargs)
    sns.histplot(inlier_score, color=colors[1], label=labels[1], ax=axs[0], **kwargs)

    offset = 2
    for idx, _score in enumerate(outlier_scores):
        idx += offset
        sns.histplot(_score, color=colors[idx], label=labels[idx], ax=axs[0], **kwargs)

    # Plot in pairs
    if len(outlier_scores) > 0:
        offset = 0
        for row in range(1, axs.shape[0]):
            sns.histplot(
                inlier_score, color=colors[1], label=labels[1], ax=axs[row], **kwargs
            )

            #         for idx in range(offset, min(len(outlier_sc)offset+2)):
            for idx, _score in enumerate(outlier_scores[offset : offset + 2]):
                idx += offset + 2
                sns.histplot(
                    _score, color=colors[idx], label=labels[idx], ax=axs[row], **kwargs
                )
            offset = 2 * row

    for ax in axs:
        ax.legend()
        ax.set_ylim(top=ylim)
        ax.set_xlim(left=xlim, right=10 if xlim else None)

    #     plt.show()

    return axs


def train_gmm(
    X_train, components_range=range(2, 21, 2), ica_range=[2, 4, 8], verbose=False
):
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import FastICA

    def scorer(gmm, X, y=None):
        return np.quantile(gmm.score_samples(X), 0.1)

    def bic_scorer(model, X, y=None):
        return -model["GMM"].bic(model["ICA"].transform(X))

    gmm_clf = Pipeline(
        [
            # ("ICA", FastICA()),
            ("scaler", StandardScaler()),
            ("GMM", GaussianMixture()),
        ]
    )

    param_grid = dict(
        # ICA__n_components=ica_range,
        # ICA__max_iter=[100000],
        # ICA__tol=[1e-4],
        GMM__n_components=components_range,
        GMM__covariance_type=["full"],
    )  # Full always performs best

    grid = GridSearchCV(
        estimator=gmm_clf,
        param_grid=param_grid,
        cv=5,
        n_jobs=1,
        verbose=1,
        scoring=scorer,
    )

    grid_result = grid.fit(X_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    if verbose:
        print("-----" * 15)
        means = grid_result.cv_results_["mean_test_score"]
        stds = grid_result.cv_results_["std_test_score"]
        params = grid_result.cv_results_["params"]
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

        plt.plot([p["GMM__n_components"] for p in params], means)
        plt.show()

    # best_gmm_clf = gmm_clf.set_params(**grid.best_params_)
    # best_gmm_clf.fit(X_train)

    return grid.best_estimator_
