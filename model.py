"""
 Main Part of auto ML is here The BLACKBOX!
"""

# sklearn models

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# normalizers 

from sklearn import preprocessing
import umap

# dimension reduction

from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA, PCA, KernelPCA
import umap


# train test split data
from sklearn.model_selection import train_test_split

# ploting
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import itertools

# helpers


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def train(models, X_train, y_train):
    """
    train  all models model

    Arguments:
    models -- all models for train
    X_train -- input features
    y_train -- output lables
    


    Return:
    models -- trained model
    models.keys -- models type
    """
    for model_type, model in models.items():
        model = model.fit(X_train, y_train)
    return models, models.keys
    
    
def find(X, Y, test_size= .2, preprocessing_type= 0, dimension_reduction_type= 0, models_type= 0, verbose= 1, normalize_validation_data= False,  random_state= 42):
    """
    find best paramter for your data with semi-auto ML pipline

    Arguments:
    X -- input features
    Y -- output lables
    split_type  -- how data splite between train and test
    preprocessing_type  -- kind of nomalizer, 0  = without any processing, 1  = full
    dimension_reduction_type  -- kind of dimension reduction, 0  = without any dimension reduction, 1  = full dimension reduction
    models_type  -- model types, 0  = simple models , 1  = full models
    verbose  --to print process , verbose = 1 print per model else not print result per model


    Return:
    None
    """
    



    """
    
    
    """
    ##########################
    #### Checking Arguments #####
    ##########################
    assert len(X) == len(Y)
    assert (test_size > 0 and test_size < 1)
    assert (dimension_reduction_type == 1 or dimension_reduction_type == 0)
    assert (models_type == 1 or models_type == 0)
    assert (verbose >= 0 or verbose < 10)
    assert (type(normalize_validation_data) == type(True))
    
    X = np.array(X)
    unique_labels = np.unique(Y)
    unique_labels_lenght = len(unique_labels)
    feature_lenght = len(X[0])
    n_samples = len(X)
    
    ##########################
    #### Set Normazilers #####
    ##########################

    X_normaized ={
    'normal':X,    
    }
    if preprocessing_type == 1:
        normalizers = {
        'MinMaxScaler':preprocessing.MinMaxScaler().fit_transform(X),
        'Normalizer':preprocessing.Normalizer().fit_transform(X),
        'scale':preprocessing.scale(X),
        'StandardScaler':preprocessing.StandardScaler().fit_transform(X),
        }
        X_normaized  = merge_dicts(X_normaized , normalizers)
    

    ###################################
    #### Set Dimension Reduction #####
    ###################################
    X_rd ={
    'normal':X,    
    }
    n_components = min(n_samples, feature_lenght) 
    n_neighbors = (feature_lenght // min(n_samples, feature_lenght) ) + n_components
    if dimension_reduction_type == 1:
        rd = {
        'TSNE': TSNE(n_components),
        'ICA':FastICA(n_components=n_components),
        'KernelPCA':KernelPCA(n_components=n_components, kernel='rbf', gamma=0.5 , max_iter = 100 , n_jobs = -1),
        'PCA':PCA(n_components=n_components),
        'UMAP':umap.UMAP(n_neighbors=n_neighbors, min_dist = 0.1,metric ='correlation')
        }
        X_rd  = merge_dicts(X_rd , rd)

    ###########################
    #### Models paramters #####    
    ###########################
    # knn paramters
    base_knn = 3
    part_match_knn = 3
    k_neighbors_number = ((feature_lenght  // unique_labels_lenght ) // part_match_knn) + base_knn
    # trees paramters
    max_depth = 100
    n_estimators = 100

    #####################
    #### Set Models #####
    #####################
    models  = {
    'KNeighborsClassifier':KNeighborsClassifier(k_neighbors_number),
    'SVC':SVC(),
    'GaussianProcessClassifier':GaussianProcessClassifier(1.0 * RBF(1.0)),
    'ExtraTreesClassifier':ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state),
    'GaussianNB':GaussianNB(),
    }
    if models_type == 1:
        
        full_models =  {
        'SVC_2':SVC(gamma=2, C=1),
        'GaussianProcessClassifier':GaussianProcessClassifier(1.0 * RBF(1.0)),
        'DecisionTreeClassifier':DecisionTreeClassifier(max_depth=max_depth),
        'RandomForestClassifier':RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, max_features=1),  
        'MLPClassifier':MLPClassifier(alpha=1, max_iter=1000),
        'AdaBoostClassifier':AdaBoostClassifier(),
        'QuadraticDiscriminantAnalysis':QuadraticDiscriminantAnalysis(),
        }
        models = merge_dicts(models , full_models)
    
    #######################
    #### Train Models #####    
    #######################
    train_scores = {}
    test_scores = {}
    i = 0
    for data_type, data_value in X_normaized.items():
        for rd_type, rd_transformer in X_rd.items():
            if verbose  > 0:
                print("""
                data  type {}
                dimension reduction  type {}
                i {}
                """.format(data_type,rd_type , i))
            i = i + 1
            
            if rd_type == 'normal':
                X_ = data_value
            else:
                X_ = rd_transformer.fit_transform(data_value)
            
            X_train, X_test, y_train, y_test = train_test_split( X_, Y, test_size=test_size, random_state=random_state)
            models , model_type = train(models , X_train , y_train)
            
            
            train_score ,test_score = status(models ,X_train , y_train , X_test , y_test,unique_labels , '{} data with {} dimension reduction'.format(data_type , rd_type) , normalize_validation_data , verbose  )
            if verbose > 1:
                print("""
                models performance
                train scores {}
                tset scores {}
                """.format(train_scores , test_scores))
            
            key_dict = '{}_{}'.format(data_type,rd_type)
            test_scores[key_dict] = test_score
            train_scores[key_dict] = train_score
    ##############################
    #### Find best parameter ####
    ##############################
    max_score = 0


    for data_type, logs in test_scores.items():
        for model_type, score in logs.items():
            if score > max_score:
                max_score_info = '{} _ {}'.format(data_type , model_type)
                max_score = score
        
    max_score_info = max_score_info.split('_')
    print("""
    ##############################
    #### Find best parameter ####
    ##############################

    MAX SCORE: {} 

    Preprocessing type: {}

    Dimension reduction type: {}

    Model type: {}
    *** normal mean normal data without any change
    """.format(max_score ,max_score_info[0] , max_score_info[1]  , max_score_info[2]))

        #####################################
        #### Return with best parameters ####
        #####################################
        # # normalizer
        # X = X_normaized[max_score_info[0]]

        # # dimension reduction
        # if max_score_info[1] == 'normal':
        #     X = X
        #     dimension_reduction_transformer = False
        # else:

        #     dimension_reduction_transformer  = rd_transformer[max_score_info[1]].fit(X)
        # # models
        # model = models[max_score_info[2]]
        
    
        
        # return model , X , Y , dimension_reduction_transformer


            


        

def status(models,x_train , y_train, x_test , y_test,lables, data_type = '', normalize = False , verbose = 2 ):
    def plot_confusion_matrix(cm,
                              classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.show()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    
    train_scores = {}
    test_scores = {}
    for key, model in models.items():
        print("""
        model : {}
        """.format(key))
        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)
        test_score = accuracy_score(pred_test,y_test)
        train_score = accuracy_score(pred_train, y_train)
        print("{} - {}: training accuracy={:.2%}, test accuracy={:.2%}".format(data_type,key,
           train_score,
         test_score))
        
        y_pred = pred_test
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)
        plot_confusion_matrix(cnf_matrix, classes=lables,
                      title='Confusion matrix for '.format(key) , normalize = normalize)
        train_scores[key] = train_score
        test_scores[key] = test_score
        
    return train_scores ,test_scores
        
 