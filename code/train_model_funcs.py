
# coding: utf-8

# In[9]:


import scipy
import warnings
import scipy.sparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
np.set_printoptions(threshold = 123456789)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from collections import Counter
from itertools import product
from chemocommons import *
from copy import deepcopy
from tqdm import tqdm

import sklearn.metrics as metrics
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_validate, GridSearchCV, KFold, RepeatedKFold

from skmultilearn.ext import Keras, Meka
from skmultilearn.adapt import MLkNN, BRkNNaClassifier, BRkNNbClassifier, MLTSVM, MLARAM
from skmultilearn.problem_transform import ClassifierChain, LabelPowerset, BinaryRelevance
from skmultilearn.ensemble import LabelSpacePartitioningClassifier, RakelD, RakelO, MajorityVotingClassifier
from skmultilearn.cluster import NetworkXLabelGraphClusterer, LabelCooccurrenceGraphBuilder, FixedLabelSpaceClusterer, MatrixLabelSpaceClusterer, RandomLabelSpaceClusterer

from iterstrat.ml_stratifiers import RepeatedMultilabelStratifiedKFold

from skmultilearn.cluster import MatrixLabelSpaceClusterer
from sklearn.cluster import KMeans

# In[ ]:


def hamming_score(y_true, y_pred):
    """
        make sure the Ys must be dense numpy.ndarray
    """
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred.todense())
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true.todense())
    return metrics.hamming_loss(y_pred, y_true)

def aiming(y_true, y_pred):
    """
        make sure the Ys must be dense numpy.ndarray
    """
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.toarray().astype(int)#np.array(y_pred.todense())
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.toarray().astype(int)
    count_pred = y_pred.sum(axis=1)
    intersection = np.bitwise_and(y_pred, y_true).sum(axis=1)
    count_pred[count_pred == 0] = 1 # add a pseudonumber to avoid zero division
    return (intersection/count_pred).mean(axis=0)

def coverage(y_true, y_pred):
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.toarray().astype(int)
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.toarray().astype(int)
    count_true = y_true.sum(axis=1)
    intersection = np.bitwise_and(y_pred, y_true).sum(axis=1)
    return (intersection/count_true).mean(axis=0)

def accuracy_multilabel(y_true, y_pred):
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.toarray().astype(int)
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.toarray().astype(int)
    intersection = np.bitwise_and(y_pred, y_true).sum(axis=1)
    union = np.bitwise_or(y_pred, y_true).sum(axis=1)
    return (intersection/union).mean(axis=0)
    
def absolute_true(y_true, y_pred):
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred.todense())
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true.todense())
    row_equal = []
    for i in range(y_pred.shape[0]):
        row_equal.append(np.array_equal(y_pred[i,:], y_true[i,:]))
    return sum(row_equal)/y_pred.shape[0]

def absolute_false(y_true, y_pred):
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.toarray().astype(int)
    return (np.bitwise_xor(y_true, y_pred).sum(axis=1) == y_pred.shape[1]).mean()
    
def precision_score(y_true, y_pred):
    if not isinstance(y_true, np.ndarray): y_true = y_true.toarray().ravel()
    if not isinstance(y_pred, np.ndarray): y_pred = y_pred.toarray().ravel()
    if len(y_true.shape) != 1: y_true = y_true.ravel()
    if len(y_pred.shape) != 1: y_pred = y_pred.ravel()
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels = [0, 1]).ravel().tolist()
    try:
        precision = tp / (tp + fp)
    except:
        precision = 0
    return precision

def recall_score(y_true, y_pred):
    if not isinstance(y_true, np.ndarray): y_true = y_true.toarray().ravel()
    if not isinstance(y_pred, np.ndarray): y_pred = y_pred.toarray().ravel()
    if len(y_true.shape) != 1: y_true = y_true.ravel()
    if len(y_pred.shape) != 1: y_pred = y_pred.ravel()
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels = [0, 1]).ravel().tolist()
    try:
        recall = tp / (tp + fn)
    except:
        recall = 0
    return recall

def f1_score(y_true, y_pred):
    if not isinstance(y_true, np.ndarray): y_true = y_true.toarray().ravel()
    if not isinstance(y_pred, np.ndarray): y_pred = y_pred.toarray().ravel()
    if len(y_true.shape) != 1: y_true = y_true.ravel()
    if len(y_pred.shape) != 1: y_pred = y_pred.ravel()
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels = [0, 1]).ravel().tolist()
    try:
        precision = tp / (tp + fp)
    except:
        precision = 0
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f2_score(y_true, y_pred):
    if not isinstance(y_true, np.ndarray): y_true = y_true.toarray().ravel()
    if not isinstance(y_pred, np.ndarray): y_pred = y_pred.toarray().ravel()
    if len(y_true.shape) != 1: y_true = y_true.ravel()
    if len(y_pred.shape) != 1: y_pred = y_pred.ravel()
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels = [0, 1]).ravel().tolist()
    try:
        precision = tp / (tp + fp)
    except:
        precision = 0
    recall = tp / (tp + fn)
    f2 = (5 * precision * recall) / (4 * precision + recall)
    return f2

def auc_aupr(y_true, y_score):
    if not isinstance(y_true, np.ndarray): y_true = y_true.toarray().ravel()
    if not isinstance(y_score, np.ndarray): y_score = y_score.toarray().ravel()
    if len(y_true.shape) != 1: y_true = y_true.ravel()
    if len(y_score.shape) != 1: y_score = y_score.ravel()
    auc = metrics.roc_auc_score(y_true, y_score)
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(recall, precision)
    return auc, aupr


# In[10]:


def binary_performance_evaluation(y_true, y_pred, labels = [0, 1]): # labels=[negative,positive]

    if not isinstance(y_true, np.ndarray): y_true = y_true.toarray().ravel()
    if not isinstance(y_pred, np.ndarray): y_pred = y_pred.toarray().ravel()

    if len(y_true.shape) != 1: y_true = y_true.ravel()
    if len(y_pred.shape) != 1: y_pred = y_pred.ravel()
        
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels = labels).ravel().tolist()
    try:
        precision = tp / (tp + fp)
    except:
        precision = 0
    recall = tp / (tp + fn)
    
    if tp != 0:
        f1_score = (2 * precision * recall) / (precision + recall)
        f2_score = (5 * precision * recall) / (4 * precision + recall)
    else: 
        f1_score, f2_score = 0, 0
        
    return (tn, fp, fn, tp), precision, recall, f1_score, f2_score

def normalized(x_train, x_test):
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    #print('Normalized: x_train.shape = {}, x_test.shape = {}'.format(x_train.shape, x_test.shape))
    return x_train, x_test

def read_data(pwd, filename, norm_idx = False, generate_lil = False):
    import numpy as np
    import scipy.sparse
    with np.load(pwd + filename, allow_pickle = True) as f:
        x_t_np = f['x_t_np']
        y_t_np = f['y_t_np']
        if norm_idx: 
            norm_col_idx = f['need_normalized_col_index']
        else:
            norm_col_idx = np.array(range(x_t_np.shape[1]))
    if generate_lil:
        x_t_lil = scipy.sparse.lil_matrix(x_t_np, dtype = np.int64)
        y_t_lil = scipy.sparse.lil_matrix(y_t_np, dtype = np.int64)
        assert (x_t_lil.toarray() == x_t_np).all(), 'x_t_np != x_t_lil'
        return x_t_np, y_t_np, x_t_lil, y_t_lil, norm_col_idx
    else: 
        return x_t_np, y_t_np, norm_col_idx

# In[11]:


def run_model(clf, x, y, norm_idx, normalized_ = False):
    rmskf = RepeatedMultilabelStratifiedKFold(n_splits = 10, n_repeats = 5, random_state = 19961231)
    
    train_index_all, test_index_all = [], []
    y_pred_train_all, y_pred_test_all = [], []
    y_pred_prob_train_all, y_pred_prob_test_all = [], []
    
    auc_train_all, auc_test_all, aupr_train_all, aupr_test_all = [], [], [], []
    confusion_matrix_train_all, precision_train_all, recall_train_all, f1_score_train_all, f2_score_train_all = [],[],[],[],[]
    confusion_matrix_test_all, precision_test_all, recall_test_all, f1_score_test_all, f2_score_test_all = [],[],[],[],[]
    
    hamming_score_train_all, hamming_score_test_all = [], []
    aiming_score_train_all, aiming_score_test_all = [], []
    coverage_score_train_all, coverage_score_test_all = [], []
    accuracy_multilabel_train_all, accuracy_multilabel_test_all = [], []
    absolute_true_score_train_all, absolute_true_score_test_all = [], []
    
    if not isinstance(x, np.ndarray) and not isinstance(y, np.ndarray): 
        rmskf_split = rmskf.split(x.toarray(), y.toarray())
    else:
        rmskf_split = rmskf.split(x, y)
    
    for train_index, test_index in tqdm(rmskf_split): #train_index与test_index为下标

        train_index_all.append(train_index) 
        test_index_all.append(test_index)
        
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if normalized_: 
            #print(x_train[:, norm_idx].shape, x_test[:, norm_idx].shape)
            x_train[:, norm_idx], x_test[:, norm_idx] = normalized(x_train[:, norm_idx], x_test[:, norm_idx])
        # fit
        clf.fit(x_train, y_train)
        # predict label
        y_pred_train = clf.predict(x_train)
        y_pred_test = clf.predict(x_test)
        y_pred_train_all.append(y_pred_train)
        y_pred_test_all.append(y_pred_test)
        # predict probability
        try:
            y_pred_prob_train = clf.predict_proba(x_train)
            y_pred_prob_test = clf.predict_proba(x_test)
            y_pred_prob_train_all.append(y_pred_prob_train)
            y_pred_prob_test_all.append(y_pred_prob_test)
            
            auc_train, aupr_train = auc_aupr(y_train, y_pred_prob_train)
            auc_test, aupr_test = auc_aupr(y_test, y_pred_prob_test)
            print('TEST: AUC = {}, AUPR = {}'.format(auc_test, aupr_test))
            auc_train_all.append(auc_train)
            auc_test_all.append(auc_test)
            aupr_train_all.append(aupr_train)
            aupr_test_all.append(aupr_test)
        except:
            None
               
        confusion_matrix_train, precision_train, recall_train, f1_score_train, f2_score_train = binary_performance_evaluation(y_train, y_pred_train)
        confusion_matrix_test, precision_test, recall_test, f1_score_test, f2_score_test = binary_performance_evaluation(y_test, y_pred_test)

        confusion_matrix_train_all.append(confusion_matrix_train)
        precision_train_all.append(precision_train)
        recall_train_all.append(recall_train)
        f1_score_train_all.append(f1_score_train)
        f2_score_train_all.append(f2_score_train)

        confusion_matrix_test_all.append(confusion_matrix_test)
        precision_test_all.append(precision_test)
        recall_test_all.append(recall_test)
        f1_score_test_all.append(f1_score_test)
        f2_score_test_all.append(f2_score_test)
            

        hamming_score_train_all.append(hamming_score(y_train, y_pred_train))
        hamming_score_test_all.append(hamming_score(y_test, y_pred_test))

        aiming_score_train_all.append(aiming(y_train, y_pred_train))
        aiming_score_test_all.append(aiming(y_test, y_pred_test))

        coverage_score_train_all.append(coverage(y_train, y_pred_train))
        coverage_score_test_all.append(coverage(y_test, y_pred_test))

        accuracy_multilabel_train_all.append(accuracy_multilabel(y_train, y_pred_train))
        accuracy_multilabel_test_all.append(accuracy_multilabel(y_test, y_pred_test))

        absolute_true_score_train_all.append(absolute_true(y_train, y_pred_train))
        absolute_true_score_test_all.append(absolute_true(y_test, y_pred_test))
    
    train_scores = (hamming_score_train_all, accuracy_multilabel_train_all, aiming_score_train_all, coverage_score_train_all, absolute_true_score_train_all, confusion_matrix_train_all, precision_train_all, recall_train_all, f1_score_train_all, f2_score_train_all, auc_train_all, aupr_train_all)
    test_scores = (hamming_score_test_all, accuracy_multilabel_test_all, aiming_score_test_all, coverage_score_test_all, absolute_true_score_test_all, confusion_matrix_test_all, precision_test_all, recall_test_all, f1_score_test_all, f2_score_test_all, auc_test_all, aupr_test_all)
    
    print('Binary metrics:')
    print('AVG Train: \nprecision = {:.4f}, recall = {:.4f}, f1 = {:.4f}, f2 = {:.4f}, auc = {:.4f}, aupr = {:.4f}'.format( np.mean(precision_train_all), np.mean(recall_train_all), np.mean(f1_score_train_all), np.mean(f2_score_train_all), np.mean(auc_train_all), np.mean(aupr_train_all)))
    print('AVG Test: \nprecision = {:.4f}, recall = {:.4f}, f1 = {:.4f}, f2 = {:.4f}, auc = {:.4f}, aupr = {:.4f}'.format(np.mean(precision_test_all), np.mean(recall_test_all), np.mean(f1_score_test_all), np.mean(f2_score_test_all), np.mean(auc_test_all), np.mean(aupr_test_all)))    


    print('Multi-label metrics:')
    print('AVG Train: \nhamming_score = {:.4f}, accuracy_multilabel_score = {:.4f}'.format(np.mean(hamming_score_train_all), np.mean(accuracy_multilabel_train_all)))
    print('AVG Test: \nhamming_score = {:.4f}, accuracy_multilabel_score = {:.4f}'.format(np.mean(hamming_score_test_all), np.mean(accuracy_multilabel_test_all)))        

    print('AVG Train: \naiming_score = {:.4f}, coverage_score = {:.4f}'.format(np.mean(aiming_score_train_all), np.mean(coverage_score_train_all)))
    print('AVG Test: \naiming_score = {:.4f}, coverage_score = {:.4f}'.format(np.mean(aiming_score_test_all), np.mean(coverage_score_test_all)))        

    print('AVG Train: \nabsolute_true_score = {:.4f}'.format(np.mean(absolute_true_score_train_all)))
    print('AVG Test: \nabsolute_true_score = {:.4f}'.format(np.mean(absolute_true_score_test_all)))    

    return train_index_all, test_index_all, y_pred_train_all, y_pred_test_all, train_scores, test_scores

def results_show(train_param_aupr_dict, test_param_aupr_dict, train_param_f2_dict, test_param_f2_dict):
    print('===============================================================================')
    if train_param_aupr_dict != {} and test_param_aupr_dict != {}:       
        best_train_aupr_param = max(train_param_aupr_dict, key = train_param_aupr_dict.get)
        best_test_aupr_param = max(test_param_aupr_dict, key = test_param_aupr_dict.get)
        print('Final Best train AUPR: best_param = {}, train = {}, test = {}'.format(best_train_aupr_param, train_param_aupr_dict[best_train_aupr_param], test_param_aupr_dict[best_train_aupr_param]))
        print('Final Best test  AUPR: best_param = {}, train = {}, test = {}'.format(best_test_aupr_param, train_param_aupr_dict[best_test_aupr_param], test_param_aupr_dict[best_test_aupr_param]))

    best_train_f2_param = max(train_param_f2_dict, key = train_param_f2_dict.get)
    best_test_f2_param = max(test_param_f2_dict, key = test_param_f2_dict.get)
    print('Final Best train F2: best_param = {}, train = {}, test = {}'.format(best_train_f2_param, train_param_f2_dict[best_train_f2_param], test_param_f2_dict[best_train_f2_param]))
    print('Final Best test  F2: best_param = {}, train = {}, test = {}'.format(best_test_f2_param, train_param_f2_dict[best_test_f2_param], test_param_f2_dict[best_test_f2_param]))
    return 

def run_model_gip(clf, x_train_all, x_test_all, y_train_all, y_test_all):

    y_pred_train_all, y_pred_test_all = [], []
    y_pred_prob_train_all, y_pred_prob_test_all = [], []
    
    auc_train_all, auc_test_all, aupr_train_all, aupr_test_all = [], [], [], []
    confusion_matrix_train_all, precision_train_all, recall_train_all, f1_score_train_all, f2_score_train_all = [],[],[],[],[]
    confusion_matrix_test_all, precision_test_all, recall_test_all, f1_score_test_all, f2_score_test_all = [],[],[],[],[]
    
    hamming_score_train_all, hamming_score_test_all = [], []
    aiming_score_train_all, aiming_score_test_all = [], []
    coverage_score_train_all, coverage_score_test_all = [], []
    accuracy_multilabel_train_all, accuracy_multilabel_test_all = [], []
    absolute_true_score_train_all, absolute_true_score_test_all = [], []
    
    for i in tqdm(range(50)): #train_index与test_index为下标
        x_train, x_test, y_train, y_test = x_train_all[i], x_test_all[i], y_train_all[i], y_test_all[i]
        clf.fit(x_train, y_train)
        # predict label
        y_pred_train = clf.predict(x_train)
        y_pred_test = clf.predict(x_test)
        y_pred_train_all.append(y_pred_train)
        y_pred_test_all.append(y_pred_test)
        # predict probability
        try:
            y_pred_prob_train = clf.predict_proba(x_train)
            y_pred_prob_test = clf.predict_proba(x_test)
            
            y_pred_prob_train_all.append(y_pred_prob_train)
            y_pred_prob_test_all.append(y_pred_prob_test)
            
            auc_train, aupr_train = auc_aupr(y_train, y_pred_prob_train)
            auc_test, aupr_test = auc_aupr(y_test, y_pred_prob_test)
            
            auc_train_all.append(auc_train)
            auc_test_all.append(auc_test)
            aupr_train_all.append(aupr_train)
            aupr_test_all.append(aupr_test)
        except:
            None
               
        confusion_matrix_train, precision_train, recall_train, f1_score_train, f2_score_train = binary_performance_evaluation(y_train, y_pred_train)
        confusion_matrix_test, precision_test, recall_test, f1_score_test, f2_score_test = binary_performance_evaluation(y_test, y_pred_test)

        confusion_matrix_train_all.append(confusion_matrix_train)
        precision_train_all.append(precision_train)
        recall_train_all.append(recall_train)
        f1_score_train_all.append(f1_score_train)
        f2_score_train_all.append(f2_score_train)

        confusion_matrix_test_all.append(confusion_matrix_test)
        precision_test_all.append(precision_test)
        recall_test_all.append(recall_test)
        f1_score_test_all.append(f1_score_test)
        f2_score_test_all.append(f2_score_test)
            

        hamming_score_train_all.append(hamming_score(y_train, y_pred_train))
        hamming_score_test_all.append(hamming_score(y_test, y_pred_test))

        aiming_score_train_all.append(aiming(y_train, y_pred_train))
        aiming_score_test_all.append(aiming(y_test, y_pred_test))

        coverage_score_train_all.append(coverage(y_train, y_pred_train))
        coverage_score_test_all.append(coverage(y_test, y_pred_test))

        accuracy_multilabel_train_all.append(accuracy_multilabel(y_train, y_pred_train))
        accuracy_multilabel_test_all.append(accuracy_multilabel(y_test, y_pred_test))

        absolute_true_score_train_all.append(absolute_true(y_train, y_pred_train))
        absolute_true_score_test_all.append(absolute_true(y_test, y_pred_test))
    
    train_scores = (hamming_score_train_all, accuracy_multilabel_train_all, aiming_score_train_all, coverage_score_train_all, absolute_true_score_train_all, confusion_matrix_train_all, precision_train_all, recall_train_all, f1_score_train_all, f2_score_train_all, auc_train_all, aupr_train_all)
    test_scores = (hamming_score_test_all, accuracy_multilabel_test_all, aiming_score_test_all, coverage_score_test_all, absolute_true_score_test_all, confusion_matrix_test_all, precision_test_all, recall_test_all, f1_score_test_all, f2_score_test_all, auc_test_all, aupr_test_all)
    
    print('Binary metrics:')
    print('AVG Train: \nprecision = {:.4f}, recall = {:.4f}, f1 = {:.4f}, f2 = {:.4f}, auc = {:.4f}, aupr = {:.4f}'.format( np.mean(precision_train_all), np.mean(recall_train_all), np.mean(f1_score_train_all), np.mean(f2_score_train_all), np.mean(auc_train_all), np.mean(aupr_train_all)))
    print('AVG Test: \nprecision = {:.4f}, recall = {:.4f}, f1 = {:.4f}, f2 = {:.4f}, auc = {:.4f}, aupr = {:.4f}'.format(np.mean(precision_test_all), np.mean(recall_test_all), np.mean(f1_score_test_all), np.mean(f2_score_test_all), np.mean(auc_test_all), np.mean(aupr_test_all)))    


    print('Multi-label metrics:')
    print('AVG Train: \nhamming_score = {:.4f}, accuracy_multilabel_score = {:.4f}'.format(np.mean(hamming_score_train_all), np.mean(accuracy_multilabel_train_all)))
    print('AVG Test: \nhamming_score = {:.4f}, accuracy_multilabel_score = {:.4f}'.format(np.mean(hamming_score_test_all), np.mean(accuracy_multilabel_test_all)))        

    print('AVG Train: \naiming_score = {:.4f}, coverage_score = {:.4f}'.format(np.mean(aiming_score_train_all), np.mean(coverage_score_train_all)))
    print('AVG Test: \naiming_score = {:.4f}, coverage_score = {:.4f}'.format(np.mean(aiming_score_test_all), np.mean(coverage_score_test_all)))        

    print('AVG Train: \nabsolute_true_score = {:.4f}'.format(np.mean(absolute_true_score_train_all)))
    print('AVG Test: \nabsolute_true_score = {:.4f}'.format(np.mean(absolute_true_score_test_all)))    

    return y_pred_train_all, y_pred_test_all, train_scores, test_scores


def best_kmeans_k(y):
    
    max_score = [0, 0]
    for i in range(684, 685): # 3, y.shape[1]
        y_pred_kmeans = KMeans(n_clusters = i, random_state = 19961231).fit_predict(y.T)
        score = metrics.silhouette_score(y.T, y_pred_kmeans) # metrics.calinski_harabaz_score(y.T, y_pred_kmeans)
        if max_score[1] < score: max_score = [i, score]
#     print('Best k = {}, calinski_harabaz_score = {}'.format(max_score[0], max_score[1]))
    
    return max_score[0]

def run_model_kmeans(ptc, x, y, norm_idx, normalized_ = False):
    rmskf = RepeatedMultilabelStratifiedKFold(n_splits = 10, n_repeats = 5, random_state = 19961231)
    
    train_index_all, test_index_all = [], []
    y_pred_train_all, y_pred_test_all = [], []
    y_pred_prob_train_all, y_pred_prob_test_all = [], []
    
    auc_train_all, auc_test_all, aupr_train_all, aupr_test_all = [], [], [], []
    confusion_matrix_train_all, precision_train_all, recall_train_all, f1_score_train_all, f2_score_train_all = [],[],[],[],[]
    confusion_matrix_test_all, precision_test_all, recall_test_all, f1_score_test_all, f2_score_test_all = [],[],[],[],[]
    
    hamming_score_train_all, hamming_score_test_all = [], []
    aiming_score_train_all, aiming_score_test_all = [], []
    coverage_score_train_all, coverage_score_test_all = [], []
    accuracy_multilabel_train_all, accuracy_multilabel_test_all = [], []
    absolute_true_score_train_all, absolute_true_score_test_all = [], []
    
    if not isinstance(x, np.ndarray) and not isinstance(y, np.ndarray): 
        rmskf_split = rmskf.split(x.toarray(), y.toarray())
    else:
        rmskf_split = rmskf.split(x, y)
    
    for train_index, test_index in tqdm(rmskf_split): #train_index与test_index为下标

        train_index_all.append(train_index) 
        test_index_all.append(test_index)
        
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if normalized_: 
            #print(x_train[:, norm_idx].shape, x_test[:, norm_idx].shape)
            x_train[:, norm_idx], x_test[:, norm_idx] = normalized(x_train[:, norm_idx], x_test[:, norm_idx])
        # fit
        k = best_kmeans_k(y_train)  ###
        print('k = ', k)
        matrix_clusterer = MatrixLabelSpaceClusterer(clusterer = KMeans(n_clusters = k, random_state = 19961231))  ###
        clf = LabelSpacePartitioningClassifier(ptc, matrix_clusterer)  ###
        clf.fit(x_train, y_train)
        # predict label
        y_pred_train = clf.predict(x_train)
        y_pred_test = clf.predict(x_test)
        y_pred_train_all.append(y_pred_train)
        y_pred_test_all.append(y_pred_test)
        # predict probability
        try:
            y_pred_prob_train = clf.predict_proba(x_train)
            y_pred_prob_test = clf.predict_proba(x_test)
            
            y_pred_prob_train_all.append(y_pred_prob_train)
            y_pred_prob_test_all.append(y_pred_prob_test)
            
            auc_train, aupr_train = auc_aupr(y_train, y_pred_prob_train)
            auc_test, aupr_test = auc_aupr(y_test, y_pred_prob_test)
            
            auc_train_all.append(auc_train)
            auc_test_all.append(auc_test)
            aupr_train_all.append(aupr_train)
            aupr_test_all.append(aupr_test)
        except:
            None
               
        confusion_matrix_train, precision_train, recall_train, f1_score_train, f2_score_train = binary_performance_evaluation(y_train, y_pred_train)
        confusion_matrix_test, precision_test, recall_test, f1_score_test, f2_score_test = binary_performance_evaluation(y_test, y_pred_test)

        confusion_matrix_train_all.append(confusion_matrix_train)
        precision_train_all.append(precision_train)
        recall_train_all.append(recall_train)
        f1_score_train_all.append(f1_score_train)
        f2_score_train_all.append(f2_score_train)

        confusion_matrix_test_all.append(confusion_matrix_test)
        precision_test_all.append(precision_test)
        recall_test_all.append(recall_test)
        f1_score_test_all.append(f1_score_test)
        f2_score_test_all.append(f2_score_test)
            

        hamming_score_train_all.append(hamming_score(y_train, y_pred_train))
        hamming_score_test_all.append(hamming_score(y_test, y_pred_test))

        aiming_score_train_all.append(aiming(y_train, y_pred_train))
        aiming_score_test_all.append(aiming(y_test, y_pred_test))

        coverage_score_train_all.append(coverage(y_train, y_pred_train))
        coverage_score_test_all.append(coverage(y_test, y_pred_test))

        accuracy_multilabel_train_all.append(accuracy_multilabel(y_train, y_pred_train))
        accuracy_multilabel_test_all.append(accuracy_multilabel(y_test, y_pred_test))

        absolute_true_score_train_all.append(absolute_true(y_train, y_pred_train))
        absolute_true_score_test_all.append(absolute_true(y_test, y_pred_test))
    
    train_scores = (hamming_score_train_all, accuracy_multilabel_train_all, aiming_score_train_all, coverage_score_train_all, absolute_true_score_train_all, confusion_matrix_train_all, precision_train_all, recall_train_all, f1_score_train_all, f2_score_train_all, auc_train_all, aupr_train_all)
    test_scores = (hamming_score_test_all, accuracy_multilabel_test_all, aiming_score_test_all, coverage_score_test_all, absolute_true_score_test_all, confusion_matrix_test_all, precision_test_all, recall_test_all, f1_score_test_all, f2_score_test_all, auc_test_all, aupr_test_all)
    
    print('Binary metrics:')
    print('AVG Train: \nprecision = {:.4f}, recall = {:.4f}, f1 = {:.4f}, f2 = {:.4f}, auc = {:.4f}, aupr = {:.4f}'.format( np.mean(precision_train_all), np.mean(recall_train_all), np.mean(f1_score_train_all), np.mean(f2_score_train_all), np.mean(auc_train_all), np.mean(aupr_train_all)))
    print('AVG Test: \nprecision = {:.4f}, recall = {:.4f}, f1 = {:.4f}, f2 = {:.4f}, auc = {:.4f}, aupr = {:.4f}'.format(np.mean(precision_test_all), np.mean(recall_test_all), np.mean(f1_score_test_all), np.mean(f2_score_test_all), np.mean(auc_test_all), np.mean(aupr_test_all)))    


    print('Multi-label metrics:')
    print('AVG Train: \nhamming_score = {:.4f}, accuracy_multilabel_score = {:.4f}'.format(np.mean(hamming_score_train_all), np.mean(accuracy_multilabel_train_all)))
    print('AVG Test: \nhamming_score = {:.4f}, accuracy_multilabel_score = {:.4f}'.format(np.mean(hamming_score_test_all), np.mean(accuracy_multilabel_test_all)))        

    print('AVG Train: \naiming_score = {:.4f}, coverage_score = {:.4f}'.format(np.mean(aiming_score_train_all), np.mean(coverage_score_train_all)))
    print('AVG Test: \naiming_score = {:.4f}, coverage_score = {:.4f}'.format(np.mean(aiming_score_test_all), np.mean(coverage_score_test_all)))        

    print('AVG Train: \nabsolute_true_score = {:.4f}'.format(np.mean(absolute_true_score_train_all)))
    print('AVG Test: \nabsolute_true_score = {:.4f}'.format(np.mean(absolute_true_score_test_all)))    

    return train_index_all, test_index_all, y_pred_train_all, y_pred_test_all, train_scores, test_scores