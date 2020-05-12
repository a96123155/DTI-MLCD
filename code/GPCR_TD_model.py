import sys
sys.path.append('/home/chujunyi/2_Program/1_code/3.1_multilabel')
from train_model_funcs import *
from skmultilearn.cluster.igraph import IGraphLabelGraphClusterer
import igraph as ig
###
pwd = '/home/chujunyi/2_Program/2_output_file/2_multilabel/1_X_Y_data/'
filename = 'GPCR_U_T_xy_np_rdkit_morgan_radius2.npz'
params_name = ['method']
params_list = [['fastgreedy', 'infomap', 'label_propagation', 'multilevel', 'walktrap', 'leading_eigenvector']]

###
with np.load(pwd + filename, allow_pickle = True) as f:
    x_t_np = f['x_t_np']
    y_t_np = f['y_t_np']

all_params = list(product(*params_list))
num_params = len(all_params)

train_param_aupr_dict, test_param_aupr_dict = {}, {}
train_param_f2_dict, test_param_f2_dict = {}, {}

for idx, param in enumerate(all_params):
    print('{}/{} iteration: method = {}'.format( ###
        idx + 1, num_params, param[0]))###
    base_classifier = RandomForestClassifier(max_depth = 40, ###
                   n_estimators = 1000, ###
                   max_features = 'sqrt', ###
                   random_state = 19961231, n_jobs = -1)
    ptc = LabelPowerset(classifier = base_classifier, require_dense = [True, True]) ###
    graph_builder = LabelCooccurrenceGraphBuilder(weighted = True, include_self_edges = False)###
    clusterer_igraph = IGraphLabelGraphClusterer(graph_builder = graph_builder, method = param[0])###
    clf = LabelSpacePartitioningClassifier(ptc, clusterer_igraph)  ###
       
    _,_,_,_,train_score, test_score = run_model(clf, x_t_np, y_t_np, norm_idx = False, normalized_ = False) ###
    
    
    if train_score[-1] != [] and test_score[-1] != []:
        train_param_aupr_dict[(param[0])] = np.mean(train_score[-1]) ###
        test_param_aupr_dict[(param[0])] = np.mean(test_score[-1]) ###
                
    train_param_f2_dict[(param[0])] = np.mean(train_score[-3]) ###
    test_param_f2_dict[(param[0])] = np.mean(test_score[-3]) ###

    if (idx+1) % 6 == 0: ###
        print('Previous {} iterations results:'.format(idx+1))
        results_show(train_param_aupr_dict, test_param_aupr_dict, train_param_f2_dict, test_param_f2_dict)
    
results_show(train_param_aupr_dict, test_param_aupr_dict, train_param_f2_dict, test_param_f2_dict)

