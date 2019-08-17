from sklearn.linear_model          import LogisticRegression
from sklearn.linear_model          import RidgeClassifier
from sklearn.linear_model          import SGDClassifier
from sklearn.linear_model          import Perceptron
from sklearn.linear_model          import PassiveAggressiveClassifier
from sklearn.ensemble              import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network        import MLPClassifier
from sklearn.svm                   import LinearSVC
from sklearn.svm                   import NuSVC
from sklearn.svm                   import SVC
from sklearn.tree                  import DecisionTreeClassifier
from sklearn.neighbors             import KNeighborsClassifier
from sklearn.neighbors             import NearestCentroid
from sklearn.neighbors             import RadiusNeighborsClassifier
from sklearn.semi_supervised       import LabelPropagation
from sklearn.semi_supervised       import LabelSpreading


def modelo_familia():
    dic_modelo_famiia = {'LINEAL':{1:'LogisticRegression',
                                   2:'RidgeClassifier',
                                   3:'SGDClassifier',
                                   4:'Perceptron',
                                   5:'PassiveAggressiveClassifier'},
                          'SVM':{1:'LinearSVC',
                                 2:'NuSVC',
                                 3:'SVC'},
                          'TREE':{1:'DecisionTreeClassifier'},
                          'NEIGHBORS':{1:'KNeighborsClassifier',
                                      2:'NearestCentroid',
                                      3:'RadiusNeighborsClassifier'},
                          'SEMI_SUPERVISED':{1:'LabelPropagation',
                                            2:'LabelSpreading'},
                          'ENSEMBLE':{1:'RandomForestClassifier'},
                          'DISCRIMINANT_ANALYSIS':{1:'LinearDiscriminantAnalysis',
                                                  2:'QuadraticDiscriminantAnalysis'},
                          'NEURAL_NETWORK':{1:'MLPClassifier'}}
    return dic_modelo_famiia


def get_modelo(semilla,familia,numero):
    clf=0
    params={}
    if familia =='LINEAL':
        if numero ==1:
            clf = LogisticRegression(random_state=semilla,penalty='l2')
            params = {'multi_class':('ovr','multinomial'),
                      'C':(1e-5,1e-3,1e-1,1e1),
                      'class_weight':(None,'balanced'),
                      'solver':('lbfgs','sag','saga','newton-cg'),
                      'max_iter':(500,1000,10000)
                      }
        if numero ==2:
            clf = RidgeClassifier(random_state=semilla)
            params = {'alpha':(1e-5,1e-3,1e-1,10,100),
                      'class_weight':(None,'balanced'),
                      'solver':('svd','cholesky','lsqr','saga'),
                      'max_iter':(500,1000,10000)
                      }
        if numero ==3:
            clf = SGDClassifier(random_state=semilla)
            params = {'loss':('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),
                      'penalty':['l2'],
                      'learning_rate':('optimal','invscaling','adaptive','constant'),
                      'class_weight':[None],
                      'alpha':(1e-5,1e-3,1e-1,10,100),
                      'epsilon':(1e-5,1e-3,1e-1,10,100),
                      'max_iter':(100,500,1000)}
        if numero ==4:
            clf = Perceptron(random_state=semilla)
            params = {'penalty':('l1','l2','elasticnet'),
                      'class_weight':(None,'balanced'),
                      'alpha':(1e-5,1e-3,1e-1,10,100),
                      'eta0':(1e-5,1e-3,1e-1,10,100),
                      'max_iter':(500,1000,10000)}
        if numero ==5:
            clf = PassiveAggressiveClassifier(random_state=semilla)
            params = {'C':(1e-5,1e-3,1e-1,10,100),
                      'class_weight':(None,'balanced'),
                      'max_iter':(500,1000,10000)}
    if familia =='SVM':
        if numero ==1:
            clf = LinearSVC(penalty='l2',random_state=semilla)
            params = {'loss':['hinge'],
                      'multi_class':['ovr'],
                      'C':(1e-5,1e-3,1e-1,10,100),
                      'class_weight':(None,'balanced'),
                      'max_iter':(500,1000,10000)}
        if numero ==2:
            clf = NuSVC(random_state=semilla)
        if numero ==3:
            clf = SVC(random_state=semilla,)
            params = {'kernel':('linear','poly','rbf','sigmoid'),
                      'degree':(2,3,4,5,6,8,10,15),
                      'C':(1e-5,1e-3,1e-1,10,100),
                      'class_weight':(None,'balanced'),
                      'max_iter':(500,1000,10000)}
    if familia =='TREE':
        if numero ==1:
            clf = DecisionTreeClassifier(random_state=semilla)
            params = {'criterion':('gini','entropy'),
                      'splitter':('best','random'),
                      'max_depth':(None,10,30,50,100),
                      'min_samples_split':(2,10,50,100,200),
                      'min_samples_leaf':(1,2,5,10,50),
                      'max_features':(None,'log2','sqrt'),
                      'class_weight':(None,'balanced')
                     }
    if familia =='NEIGHBORS':
        if numero ==1:
            clf = KNeighborsClassifier()
            params = {'n_neighbors':(1,2,3,4,5,7),
                      'metric':('euclidean','manhattan','chebyshev','minkowski')
                     }
        if numero ==2:
            clf = NearestCentroid()
            params = {'metric':('euclidean','manhattan','chebyshev','minkowski')
                     }
        if numero ==3:
            clf = RadiusNeighborsClassifier()
            params = {'radius':(1,2,3,4,5,10,15),
                      'metric':('euclidean','manhattan','chebyshev','minkowski')
                     }
    if familia =='SEMI_SUPERVISED':
        if numero ==1:
            clf = LabelPropagation()
            params = {'kernel':('rbf','knn'),
                      'n_neighbors':(1,2,3,4,5,10,15),
                      'gamma':(1e-5,1e-3,1e-1,1,20,100),
                      'max_iter':(100,200,300,500,1000)
                     }
        if numero ==2:
            clf = LabelSpreading()
            params = {'kernel':('rbf','knn'),
                      'n_neighbors':(1,2,3,4,5,10,15),
                      'alpha':(1e-5,1e-3,1e-1,1e1,1e2,1e3),
                      'max_iter':(100,200,300,500,1000)
                     }
    if familia =='ENSEMBLE':
        if numero ==1:
            clf = RandomForestClassifier(random_state=semilla)
    if familia =='DISCRIMINANT_ANALYSIS':
        if numero ==1:
            clf = LinearDiscriminantAnalysis()
        if numero ==2:
            clf = QuadraticDiscriminantAnalysis()
    if familia =='NEURAL_NETWORK':
        if numero ==1:
            clf = MLPClassifier(random_state=semilla)
    return clf,params
