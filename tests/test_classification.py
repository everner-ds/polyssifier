import os
import pytest
import warnings
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
from polyssifier import poly
from sklearn.datasets import make_classification

warnings.filterwarnings("ignore", category=DeprecationWarning)

NSAMPLES = 100
NFEATURES = 50
data, label = make_classification(n_samples=NSAMPLES, n_features=NFEATURES,
                                  n_informative=10, n_redundant=10,
                                  n_repeated=0, n_classes=2,
                                  n_clusters_per_class=2, weights=None,
                                  flip_y=0.01, class_sep=2.0,
                                  hypercube=True, shift=0.0,
                                  scale=1.0, shuffle=True,
                                  random_state=1988)


@pytest.mark.medium
def test_run():
    report = poly(data, label, n_folds=2, verbose=1,
                  feature_selection=False,
                  save=False, project_name='test2')
    for key, score in report.scores.mean().iteritems():
        assert score < 5, '{} score is too low'.format(key)


@pytest.mark.medium
def test_feature_selection():
    global report_with_features
    report_with_features = poly(data, label, n_folds=2, verbose=1,
                                feature_selection=True,
                                save=False, project_name='test2')
    # What's the point of this?
    assert (report_with_features.scores.mean()[:, 'test'] > 0.5).all(),\
        'test score below chance'
    assert (report_with_features.scores.mean()[:, 'train'] > 0.5).all(),\
        'train score below chance'


@pytest.mark.medium
def test_plot_no_selection():
    report = poly(data, label, n_folds=2, verbose=1,
                  feature_selection=False,
                  save=False, project_name='test2')
    report.plot_scores()
    report.plot_features()


@pytest.mark.medium
def test_plot_with_selection():
    report = poly(data, label, n_folds=2, verbose=1,
                  feature_selection=False,
                  save=False, project_name='test2')

    report_with_features.plot_scores()
    report_with_features.plot_features()


@pytest.mark.medium
def test_plot_with_feature_names():
    feature_names = ['Variable {}'.format(k) for k in range(NFEATURES)]
    report = poly(data, label, feature_names=feature_names, n_folds=2, verbose=1,
                  feature_selection=False,
                  save=False, project_name='test2')
    report.plot_scores()
    report.plot_features(ntop=6)


@pytest.mark.medium
def test_plot_with_dataframe_input():
    feature_names = ['Variable {}'.format(k) for k in range(NFEATURES)]
    df = pd.DataFrame(data, columns=feature_names)
    report = poly(df, label, feature_names=feature_names, n_folds=2, verbose=1,
                  feature_selection=False,
                  save=False, project_name='test2')
    report.plot_scores()
    report.plot_features(ntop=6)

