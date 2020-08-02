#Data minding Fundamentals
#cluster analysis (uncivilized analysis = outcome/target is unknown)
#Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) 
#are more similar (in some sense) to each other than to those in other groups (clusters).
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
X = df.drop('estimated_value', axis=1) #make outcome unknown, make feature matrix x
X= X[['bedrooms', 'bathrooms','priorSaleAmount']] #pick out columns
X.fillna(0, inplace=True)   #organize data
kmeans = KMeans(n_clusters=5, random_state=0).fit(X) #number of clusters (need many iterations)
labels = kmeans.labels_ #labels of each point (attribute)
kmeans.cluster_centers_ #coordinates of cluster centers (avg of all points in cluster) (shape is # clusters, # features/columns)
X['cluster'] = labels #makes new column and shows which cluster each point is in
X.grouby('cluster').max() #looks at max of each column in each cluster 
silhouette_score(X, labels) #silhouette score determines how good the cluster is 


#classification and regression (both are supervised learning, labeled outcome)
#classification labels outcomes, can be two classes or more (true false vs red blue green yellow)
#regression is predicting numerical value outcome
from sklearn.model_selection import train_test_split #splits dataset into training and testing for better results
from sklearn.metrics import confusion_matrix #allows visualization of the performance of an algorithm, typically a supervised learning one
#CONFUSION MATRIX: Each row of the matrix represents the instances in a predicted class while each column represents the instances in an actual class (or vice versa).
#X is from before (upper case means matrix, lower case means vector-one line)
#
from sklearn.linear_model import LinearRegression, LogisticRegression #two simplist models (linear=numerical, logistic=class)
#LINEAR REGRESSION
y=df.estimated_value 
lg = LinearRegression() #init. first try with all data used in fitting
lg.fit(X,y) #fit X to y
lg.score(X,y) #fit score (how well the fit is). might be overfitting- means going after the noise
X_train, X_test, y_train, y_test = train_test_split(X,y) #another try with training. first four in specifically this order, splits 80% training 20% testing
lg = LinearRegression()
lg.fit(X_train, y_train) #train model
lg.score(X_test,y_test) #test model on new data so it's not overfit for the specific data
df['estimated_value_bins']= df.estimated_value.apply(lambda x: 'high' if x>500000 else 'low') #lambda function, if estimated is >700000 then considered high, create new column. the bin is a categorical variable
#LOGISTIC REGRESSION (CLASSIFICAITON)
y2 = df.estimated_value_bins
log = LogisticRegression() 
X_train, X_test, y2_train, y2_test = train_test_split(X,y2)
log.fit(X_train, y2_train)
log.score(X_test, y2_test)
y_pred = log.predict(X_test) #shows prediction values with fitted model
confusion_matrix(y2_test,y_pred) #left top and right bottom is right prediction, opposite is wrong
#
from sklearn.svm import SVC, SVR #svm = support vector machine (classifier vs regressor). SVC tries to create a gap between two clusters
#SVR
svr = SVR()
svr.fit(X_train, y_train)
svr.score(X_test, y_test)
#SVC
svc = SVC()
svc.fit(X_train, y2_train)
svc.score(X_test, y2_test)
y2_pred = svc.predict(X_test)
confusion_matrix(y2_test,y2_pred)
#
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor #KNN: fit by looking at neightbors
#KNR
knr = KNeighborsRegressor()
knr.fit(X_train, y_train)
knr.score(X_test, y_test)
#KNC
knc = KNeighborsClassifier()
knc.fit(X_train, y2_train)
knc.score(X_test, y2_test)
y2_pred = knc.predict(X_test)
confusion_matrix(y2_test,y2_pred)


#Association and Correlation (getting rid of outliers)
df.corr() #gives correlation matrix (range from -1 to 1)
sns.heatmap(df.corr()) #heatmap of corr
df.cov() #covariance is unnormalized form of correlation (corr comes from cov with normalization)
y = df.estimated_value
X.columns #outputs column names
#1. three sigma rule: exclude anything outside 3 std dev
for i in X.columns: #prints histogram between one column vs all columns, easy to determine outliers
    print(i)
    X.loc[:,i].hist()
    print ('mean:', X.loc[:, i].mean())
    print ('std:', X.loc[:, i].std())
    plt.show() #using matplotlib to draw graph
#2. boxplot rule: only include data between certain percentiles
sns.boxplot(X) #plots all columns into a single graph with seperate box plots
sns.boxplot(X[['bedrooms', 'bathrooms']])#if one column's scale if off, specify columns want to keep
#3. Mahalanobis Rule 


#Dimensionality Reduction
#curse of dimesionality: as the number of features or dimensions grows, the amount of data we need to genralize accurately grows exponentially
from sklearn.decomposition import PCA #PCA is like a feature compressor (compress 100 features down to 10 new features, but each feature will contain some of old features). new feature loses literal meaning 
pca = PCA(4) #init 4 components (4 is arbituary and needs to trial and error)
X_transformed = pca.fit_transform(X) #X originially have 7 features, turn it into 4 features
pca.components_ #output the components (#newfeatures * #old features). each row represents how much of old features is in new feature
lg = LinearRegression()  #try to fit with new method
X_train, X_test, y_train, y_test = train_test_split(X_transformed,y)
lg.fit(X_train, y_train)
lg.score(X_test,y_test)
#look at elbow plots and eigenvectors/values to determine how many features needed
