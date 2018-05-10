import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.externals.six import StringIO
from pydotplus import graph_from_dot_data
from IPython.display import Image  
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("F:\D drive\data science\DataScience\DataScience-Python3\mammographic_masses.data.txt",na_values=['?'])
df.columns = ['BI_RADS', 'age', 'shape', 'margin', 'density', 'severity']

for rows in df.columns[0:]:
    df[rows] = df[rows].replace('?', np.nan)

df.dropna(inplace=True)    
#print(df.describe(include = 'all'))

array_asmd = df[['age', 'shape', 'margin', 'density']].values

array_severity = df['severity'].values

scaler = preprocessing.StandardScaler().fit(array_asmd)
rescaled_asmd = scaler.transform(array_asmd)
#print(rescaled_asmd)

rescaled_asmd_train,rescaled_asmd_test,severity_train,severity_test = train_test_split(rescaled_asmd,
                                                                      array_severity,test_size=0.25,random_state = 1)
#print(severity_train)

clf_gini = DecisionTreeClassifier()
clf_gini.fit(rescaled_asmd_train, severity_train)

#print(clf_gini.score(rescaled_asmd_test,severity_test))

out = StringIO()
tree.export_graphviz(clf_gini, out_file=out, feature_names=['age', 'shape', 'margin', 'density'])
graph = graph_from_dot_data(out.getvalue())  
Image(graph.create_png()) 

kcv_score = cross_val_score(clf_gini, rescaled_asmd, array_severity, cv=10)
#print(kcv_score.mean())  #0.73

clf_rf = RandomForestClassifier()
clf_rf.fit(rescaled_asmd_train, severity_train)

rf_score = cross_val_score(clf_rf, rescaled_asmd, array_severity, cv=10)
#print(rf_score.mean()) #0.769

clf_svc = svm.SVC(kernel='linear', C=1.0) #kernel='rbf', kernel='sigmoid', kernel='poly'
svm_cv_scores = cross_val_score(clf_svc, rescaled_asmd, array_severity, cv=10)

#print(svm_cv_scores.mean()) # 80.3


for n in range(1, 50):
    clf_knn = neighbors.KNeighborsClassifier(n_neighbors=n)
    knn_cv_scores = cross_val_score(clf_knn, rescaled_asmd, array_severity, cv=10)
    #print(n,knn_cv_scores.mean())

scaler = preprocessing.MinMaxScaler().fit(array_asmd)
rescaled_asmd = scaler.transform(array_asmd)

clf_nb = MultinomialNB()
nb_cv_scores = cross_val_score(clf_nb, rescaled_asmd, array_severity, cv=10)
#print(nb_cv_scores.mean()) #78.42



clf_lr = LogisticRegression()
lr_cv_scores = cross_val_score(clf_lr, rescaled_asmd, array_severity, cv=10)
print(lr_cv_scores.mean())


