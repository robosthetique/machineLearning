##############################################################################
# 2017-07-13
#
# tutorial from:
# http://machinelearningmastery.com/machine-learning-in-python-step-by-step/
#
# Use 'help('function_name')' for further understanding
#
# Program Summary: Guess species of iris flowers
##############################################################################

# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names) 

##############################################################################
# Load Diagnostic checks to understand dataset:
##############################################################################

# shape
#print(dataset.shape)

# head
#print(dataset.head(20))

# descriptions
#print(dataset.describe())

# class distribution
#print(dataset.groupby('class').size())

# data visualization:
#	1. Univariate plots to better understand each attribute
#	2. Multivariate plots to evaluate relationships between attributes

# box and whisker plots				(Univariate)
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

# histograms						(Univariate)
#dataset.hist()
#plt.show()

# scatter plot matrix				(Multivariate)
#scatter_matrix(dataset)
#plt.show()

##############################################################################
# evaluate algorithims:
#	1. Separate out a validation dataset
#	2. Setup the test harness to use 10-fold cross validation
#	3. Build 5 different models to predict species from flower measurements
#	4. Select the best model
##############################################################################

# we're ommiting some data from the algorithim to check the accuracy later on
# 8/10ths of the data is for the algorithm, 2/10ths is for quality control
# Seperates into X_train/Y_train and X_validation/Y_validation datasets
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric:
# splits dataset into 10 parts (train on 9 and test on 1, repeat)
#seed = 7	# tutorial had this in here? don't know why
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
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))		# un-comment for KNN stats
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
#results = []
#names = []
#for name, model in models:
#	# reset random number seed before each run to use same data splits
#	kfold = model_selection.KFold(n_splits=10, random_state=seed)
#	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#	results.append(cv_results)
#	names.append(name)
#	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#	print(msg) # print accuracy 

# Next step is to pick most accurate example; the terminal gives us:
#	>>LR: 0.966667 (0.040825)
#	>>LDA: 0.975000 (0.038188)
#	>>KNN: 0.983333 (0.033333)
#	>>CART: 0.975000 (0.038188)
#	>>NB: 0.975000 (0.053359)
#	>>SVM: 0.991667 (0.025000)
# in this case, SVM is best algorithm
# but im starting with KNN because that's what the tutorial has set up

# also, if, for some reason, you need some box/whisker plots for above numbers:
#fig = plt.figure()
#fig.suptitle('Algorithm Comparison')
#ax = fig.add_subplot(111)
#plt.boxplot(results)
#ax.set_xticklabels(names)
#plt.show()

# need to watch for overfitting to the training set or data leak
# either occurance results in optimistic accuracy return

##############################################################################
# now we're going to run the model on the validation set directly;
# the results will be summarized as:
#	(1) final accuracy, (2) confusion matrix, (3) classification report
##############################################################################

# Make predictions on validation dataset

# KNN:
#knn = KNeighborsClassifier()
#knn.fit(X_train, Y_train)
#predictions = knn.predict(X_validation)
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))

# SVM:
svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# confusion matrix indicates errors;
# classification report provides a breakdown of each class by:
#	(1) precision, (2) recall, (3) f1-score, (4) support



