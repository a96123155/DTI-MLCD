import sys
sys.path.append('/home/chujunyi/2_Program/1_code/3.1_multilabel')
from train_model_funcs import *
from skmultilearn.cluster.igraph import IGraphLabelGraphClusterer
import igraph as ig
###
pwd = '/home/chujunyi/2_Program/2_output_file/2_multilabel/1_X_Y_data/'
filename = 'GPCR_U_D_xy_np_CTD_PFAM.npz'

params_name = ['method', 'max_depth', 'n_estimators', 'max_features']
params_list = [['fastgreedy'],#, 'infomap', 'label_propagation', 'multilevel', 'walktrap', 'leading_eigenvector']],
               [None, 100, 80, 60, 20],
               [1500, 1200, 1000, 800, 600, 400, 100],
               ['sqrt']]
###
with np.load(pwd + filename) as f:
    x_d_np, y_d_np, norm_idx = f['x_d_np'], f['y_d_np'], f['norm_idx']

all_params = list(product(*params_list))
num_params = len(all_params)

train_param_aupr_dict, test_param_aupr_dict = {}, {}
train_param_f2_dict, test_param_f2_dict = {}, {}

for idx, param in enumerate(all_params):
    print('{}/{} iteration: method = {}, max_depth = {}, n_estimators = {}, max_features = {}'.format( ###
         idx + 1, num_params, param[0], param[1], param[2], param[3]))###
    base_classifier = RandomForestClassifier(max_depth = param[1], ###
                                 n_estimators = param[2], ###
                                 max_features = param[3], ###
                                 random_state = 19961231, n_jobs = -1) ###
    graph_builder = LabelCooccurrenceGraphBuilder(weighted = True, include_self_edges = False)###
    problem_transform_classifier = LabelPowerset(classifier = base_classifier, require_dense = [True, True])###
    clusterer_igraph = IGraphLabelGraphClusterer(graph_builder = graph_builder, method = param[0])###
    clf = LabelSpacePartitioningClassifier(problem_transform_classifier, clusterer_igraph)###

    _,_,_,_,train_score, test_score = run_model(clf, x_d_np, y_d_np, norm_idx, True) ###
    
    
    if train_score[-1] != [] and test_score[-1] != []:
        train_param_aupr_dict[(param[0], param[1], param[2], param[3])] = np.mean(train_score[-1]) ###
        test_param_aupr_dict[(param[0], param[1], param[2], param[3])] = np.mean(test_score[-1]) ###
                
    train_param_f2_dict[(param[0], param[1], param[2], param[3])] = np.mean(train_score[-3]) ###
    test_param_f2_dict[(param[0], param[1], param[2], param[3])] = np.mean(test_score[-3]) ###

    if (idx+1) % 6 == 0: ###
        print('Previous {} iterations results:'.format(idx+1))
        results_show(train_param_aupr_dict, test_param_aupr_dict, train_param_f2_dict, test_param_f2_dict)
    
results_show(train_param_aupr_dict, test_param_aupr_dict, train_param_f2_dict, test_param_f2_dict)