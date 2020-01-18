import numpy as np
import pandas as pd
from sklearn import metrics


def load_ICU_data(path):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'martial_status', 'occupation', 'relationship', 'race', 'sex',
                    'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target']
    input_data = (pd.read_csv(path, names=column_names,
                              na_values="?", sep=r'\s*,\s*', engine='python')
                  .loc[lambda df: df['race'].isin(['White', 'Black'])])
    # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
    sensitive_attribs = ['race', 'sex']
    Z = (input_data.loc[:, sensitive_attribs]
         .assign(race=lambda df: (df['race'] == 'White').astype(int),
                 sex=lambda df: (df['sex'] == 'Male').astype(int)))

    # targets; 1 when someone makes over 50k , otherwise 0
    y = (input_data['target'] == '>50K').astype(int)

    # features; note that the 'target' and sentive attribute columns are dropped
    X = (input_data
         .drop(columns=['target', 'race', 'sex', 'fnlwgt'])
         .fillna('Unknown')
         .pipe(pd.get_dummies, drop_first=True))

    print("features X: {X.shape[0]} samples, {X.shape[1]} attributes")
    print("targets y: {y.shape} samples")
    print("sensitives Z: {Z.shape[0]} samples, {Z.shape[1]} attributes")
    return X, y, Z


def p_rule(y_pred, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    odds = y_z_1.mean() / y_z_0.mean()
    return np.min([odds, 1/odds]) * 100


def plot_distributions(y_true, Z_true, y_pred, Z_pred=None, epoch=None):

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    subplot_df = (
        Z_true
        .assign(race=lambda x: x['race'].map({1: 'white', 0: 'black'}))
        .assign(sex=lambda x: x['sex'].map({1: 'male', 0: 'female'}))
        .assign(y_pred=y_pred)
    )
    _subplot(subplot_df, 'race', ax=axes[0])
    _subplot(subplot_df, 'sex', ax=axes[1])
    _performance_text(fig, y_true, Z_true, y_pred, Z_pred, epoch)
    fig.tight_layout()
    return fig


def _subplot(subplot_df, col, ax):
    ax.set_title('Sensitive attribute: {col}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 7)
    ax.set_yticks([])
    ax.set_ylabel('Prediction distribution')
    ax.set_xlabel(r'$P({{income>50K}}|z_{{{}}})$'.format(col))


def _performance_text(y_test, Z_test, y_pred, Z_pred=None, epoch=None):

    if epoch is not None:
        print ("Training epoch %d" %epoch)

    clf_roc_auc = metrics.roc_auc_score(y_test, y_pred)
    clf_accuracy = metrics.accuracy_score(y_test, y_pred > 0.5) * 100
    #p_rules = {'race': p_rule(y_pred, Z_test['race']),
    #           'sex': p_rule(y_pred, Z_test['sex']),}
    print ("Classifier performance: ROC AUC: %f Accuracy: %f" %(clf_roc_auc,clf_accuracy))
    #print ('\n'.join(["Satisfied p%-rules:"] +
    #                             ["- {attr}: {p_rules[attr]:.0f}%-rule"
    #                              for attr in p_rules.keys()]))
    if Z_pred is not None:
        adv_acc1 = metrics.accuracy_score(Z_test[:,0], Z_pred['race'] > 0.5) * 100
        adv_acc2 = metrics.accuracy_score(Z_test[:,1], Z_pred['sex'] > 0.5) * 100
        adv_roc_auc = metrics.roc_auc_score(Z_test, Z_pred)
        print ("Adversary performance: ROC AUC: %f acc1: %f, acc2: %f" %(adv_roc_auc, adv_acc1, adv_acc2))
    return clf_roc_auc,clf_accuracy,adv_acc1,adv_acc2,adv_roc_auc









