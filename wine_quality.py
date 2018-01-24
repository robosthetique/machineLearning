##############################################################################
# 2017-07-14
#
# Importing a dataset to test wine quality- 
# meant as practice for methods used in iris tutorial.
#
# Data grabbed from UCI Machine Learning Repository:
#	'Abstract: Using chemical analysis to determine origin of wines'
#
# +-----------------------------+-------------------+
# | Data Set Characteristics	| Multivariate		|
# | Area						| Physical			|
# | Attribute Characteristics	| Integer, Real		| ---!!! real numbers used
# | Number of Attributes		| 13				|
# | Date Donated				| 1991-07-01		|
# | Associated Tasks			| Classifcaition	|
# | Missing Values?				| No				|
# | Number of Web Hits			| 702114			|
# +-----------------------------+-------------------+
##############################################################################

# load libraries
import pandas # needed to grab dataset
# used for user data analysis:
#from pandas.tools.plotting import scatter_matrix
#import matplotlib.pyplot as plt
from sklearn import model_selection
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
# algorithms used/available
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# used to change real #s to int so we can input them into sklearn libraries
from sklearn import preprocessing
from sklearn import utils

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
names = ['alcohol', 'malic-acid', 'ash', 'ash-alcalinity', 'magnesium', 'total-phenols', 'proanthocyanins', 'color-intensity', 'hue', 'od--dilution', 'proline']
dataset = pandas.read_csv(url, names=names) # read csv (comma seperated) file
		# first parameter is filepath_or_buffer
		# 'names' parameter list of column names to use
	

# following is a list of checks for user understanding of data
# shape
#print(dataset.shape)		# should be (178, 11) [(rows, columns)]
# head
print(dataset.head(20))	# prints first 20 datasets
# descriptions
#print(dataset.describe())	# prints count/mean/std/min/max & whisker plot data
# class distribution
#print(dataset.groupby('class').size())	# !!!! can't get to work - revisit

##############################################################################
# evaluate algorithims:
#	1. Separate out a validation dataset
#	2. 'Setup the test harness to use 10-fold cross validation' - ???
#	3. Build 5 different models to predict attributes
#	4. Select the best model
##############################################################################

# we're ommiting some data from the algorithim to check the accuracy later on
# 8/10ths of the data is for the algorithm, 2/10ths is for quality control
# Seperates into X_train/Y_train and X_validation/Y_validation datasets
array = dataset.values
X = array[:,0:10]
Y = array[:,10]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric:
# splits dataset into 10 parts (train on 9 and test on 1, repeat)
scoring = 'accuracy'
# 'accuracy' = #correct/#total * 100
# i.e., % accurate
# scoring var will be used when we run & eval each model

# Now we'll be checking 6 different algorithms:
# 	1. LR	(logistic regression)					#simple/linear
#	2. LDA	(linear discriminant analysis)			#simple/linear
#	3. KNN	(K-Nearest Neighbors)					#nonlinear
#	4. CART	(Classification and Regression Trees)	#nonlinear
#	5. NB	(Gaussian Naive Bayes)					#nonlinear
#	6. SVM	(Support Vector Machines)				#nonlinear
# these provide a varied mixture (important!!)

#
# Build and Evaluate Each 5 models:
#

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))	
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
######
print('-------')
for Y in array:
	int(Y)	# these functions only work with integers
	print(Y)
##### tag - stuck on above
for name, model in models:
	# reset random number seed before each run to use same data splits
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
#	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#	results.append(cv_results)
#	names.append(name)
#	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#	print(msg) # print accuracy 

