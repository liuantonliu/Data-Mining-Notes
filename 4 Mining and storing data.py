#Mining and storing data
#Text mining
import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
data_path = ("data/SMSSpamCollection") # Grab and process the raw data.
sms_raw = pd.read_table('data/SMSSpamCollection',  header=None) #read because it's txt file, and no headers are given
sms_raw.columns = ['spam', 'message'] #choose two columns
sms_raw.spam.value_counts() #frequency count on unique entries
sms_raw.shape
#text analysis
keywords = ['click', 'offer', 'winner', 'buy', 'free', 'cash', 'urgent', 'money'] #create features/columns with keywords as headers. act as flags with true/false values. 
for key in keywords:
    sms_raw[str(key)] = sms_raw.message.str.contains( # Note that we add spaces around the key so that we're getting the word, not just pattern matching.
        ' ' + str(key) + ' ', #if contains one or more of the keyword
        case=False #flag as True, else = False
    )
sms_raw.tail(10) #look at the end
sms_raw['allcaps'] = sms_raw.message.str.isupper() #new column detecting caps or not 
sms_raw[sms_raw.allcaps==True].head(2) #look at the first two flagged by the caps
sms_raw.corr()
sns.heatmap(sms_raw.corr())
data = sms_raw[keywords + ['allcaps']] #grab keyword/flag columns
target = sms_raw['spam']
#
from sklearn.naive_bayes import BernoulliNB # Our data is binary / boolean, so we're importing the Bernoulli classifier.
bnb = BernoulliNB() # Instantiate our model and store it in a new variable.
bnb.fit(data, target) # Fit our model to the data.
y_pred = bnb.predict(data) # Classify, storing the result in a new variable.
print("Number of mislabeled points out of a total {} points : {}".format( # Display our results.
    data.shape[0], #total points
    (target != y_pred).sum() #points got wrong
))
#
#Network mining (ex. getting data from social media)
import networkx as nx #conda install networkx  
G = nx.DiGraph() #Creating a directed graph
G.add_node('You') #Adding nodes.  We could have added them all in one list using .add_nodes_from()
G.add_node('Mom')
G.add_node('Aunt Alice')
G.add_node('Fiancee')
G.add_edges_from([('You','Mom'),('You','Fiancee')]) #Adding edges.  You can also add them individually using .add_edge() 
G.add_edges_from([('Mom','You'),('Mom','Aunt Alice')])
G.add_edges_from([('Aunt Alice','Mom'),('Aunt Alice','You'), ('You', 'Aunt Alice')])
nx.draw_networkx(G, #Drawing the graph
                 pos=nx.circular_layout(G), # Positions the nodes relative to each other
                 node_size=1600, 
                 cmap=plt.cm.Blues, # The color palette to use to color the nodes.
                 node_color=range(len(G)) #The number of shades of color to use.
                 )
plt.axis('off')
#graph properties
print("This graph has {} nodes and {} edges.".format(G.number_of_nodes(),G.number_of_edges()))
print('The "Aunt Alice" node has an in-degree of {} and an out-degree of {}.'.format(G.in_degree('Aunt Alice'),G.out_degree('Aunt Alice')))
print("The nodes are {}.".format(G.nodes()))
print("The edges are {}.".format(G.edges()))
print("The betweenness centrality scores are {}".format(nx.betweenness_centrality(G))) #Compute the shortest-path betweenness centrality for nodes. 
print('The node degrees are {}.'.format(G.degree())) #The node degree is the number of edges adjacent to that node. 
print("Simple paths:",nx.all_pairs_node_connectivity(G),'\n')
print("Shortest paths:", nx.all_pairs_shortest_path(G))
nx.all_pairs_shortest_path(G)
#
#Python matrix library
## Numpy 
## Pandas 

A= np.array([[1,2,3], [4,5,6]]) #init arr NOT matrix
A.T #transpose
B=np.array([[7,8,9], [10,11,12]])
np.dot(A,B.T) #dot product  
C=A**2 #every number in arr squared separately
D= A-1  #every number in A subtracted by 1
C2=np.array([[1,2,3], [1,2,3], [3,4,5]])
C2.diagonal() #outputs [1,2,5]
A=np.matrix(A) #converts A into matrix from array
type(A) #outputs matrix
B=np.matrix(B)
C=np.matrix(C)
A.nonzero() #Return the indices of the elements that are non-zero. 
A-B
A+B
A.dot(B.T) 
print(A.sum(), A.mean(), A.std(), A.var())
A_df=pd.DataFrame(A) #convert matrix into df
#
#mining a database - SQL
## Postgre SQL, SQLite 
# select * from procedure;

# select  specialty, count(id) as physician_count 
# from physician
# group by specialty
# order by 2 desc 
# limit 10 ;


# select  specialty, count(distinct procedure_code)  
# from  physician a 
# join procedure b on a.id=b.physician_id
# group by specialty ;