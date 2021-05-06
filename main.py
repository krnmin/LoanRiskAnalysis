import pandas as pd							# using pandas to get data from csv file
from numpy import savetxt					# used to save the numpy arrays as a file
from sklearn import tree					# imports the decision tree classifier from sklearn
from sklearn.ensemble import RandomForestClassifier		# random forest classifier
from matplotlib import pyplot as plt					# make pictures of my tree

trainData = pd.read_csv('data/train.csv')					#skipinitialspace=True can be used to remove the whitespace found in CSV files
trainData = trainData[['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','credit']]		# setting the columns that we will be using for the training data (we're gonna be using all of them)
trainData['F10'] = trainData['F10'].map({" White": 0 , " Black": 1 ," Asian-Pac-Islander":2 ," Amer-Indian-Eskimo":3 ," Other":4})		# Have to match exactly how the CSV file has therefore there will be a whitespace before each category.
trainData['F11'] = trainData['F11'].map({" Male": 0 , " Female": 1})		# giving numerical values to race and gender to be processed by 
trainData = trainData.dropna()												# removing any empty instances found in PANDAS documentation

X = trainData.drop('credit',axis =1)										# separating the data from everything except the credit risk
y = trainData['credit']                                                     # separating the data to only hold the credit risk

#creditTree = tree.DecisionTreeClassifier(max_depth = 4)         			# creating a decision tree from sklearn's library can set max_depth and other attributes to prevent overfitting
creditTree = RandomForestClassifier()		# alternatively using random forest classifier
creditTree.fit(X,y)                                                         # fitting the tree with the data and the credit risk rating (0, or 1). we now have a fit tree


testData = pd.read_csv('data/test.csv')                                     # read testing data
testData['F10'] = testData['F10'].map({" White": 0 , " Black": 1 ," Asian-Pac-Islander":2 ," Amer-Indian-Eskimo":3 ," Other":4})			# giving values to race in test data
testData['F11'] = testData['F11'].map({" Male": 0 , " Female": 1})                                                                          # giving values to gender in test Data
test = testData.drop(testData.columns[0], axis = 1)                         # dropping column 0 which has the ids to match training data

testPrediction = creditTree.predict(test)                                   # 
savetxt('out.csv',testPrediction, fmt = '%i')								# output file name ,data, format = integer since we don't want floating point precision with our output

# following lines are used to create a png file that represents the tree. can't be used with random forest classifier
#fig = plt.figure(figsize=(25,20))
#_ = tree.plot_tree(creditTree, 
#                   feature_names=X.columns,
#                   filled=True)
#fig.savefig("tree.png")
