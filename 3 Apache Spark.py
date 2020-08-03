#Apache Spark
#framework for distributed processing. streamlined alternative to map-reduce
#can analyze large amounts of data fast (uses more RAM and multitasks)
#Ecosystem: Spark SQL+dataframes, streaming, MLlib, GraphX, Core API
#
#IMPORTANT: ADD \ AFTER LINE FOR CONTINUATION. NOT ADDED HERE BECAUSE COMMENT INTERFERES 
#
#coin toss example
import pyspark as ps
spark = ps.sql.SparkSession.builder  #everything below is chained, supposed to be in one line
    .master('local[4]')
    .appName('spark-lecture')
    .getOrCreate() #get app if app name  exists, create app if doesn't exist
sc = spark.sparkContext #creating spark context locally
import random
n=100 #100 coin tosses
heads = (sc.parallelize(xrange(n)) #creates an RDD (resilient distributed dataset), compute all 100 throws in parallel
    .map(lambda _: random.random()) #_ means for every number (0-99), map a random value
    .filter(lambda r: r<0.5) #eliminates numbers under 0.5. can replace lambda with def function such as .filter(is_prime)
    .count()) #counts list length now
#OR break the operation up
numbers = xrange(n)
rdd = sc.parallelize(numbers)
rdd2 = rdd.filter(is_prime) #sets condition
rdd2.collect()  #final operation that performs all operations (uses a lot of computational power)
tails = n-heads
ratio = 1.*heads/n #use 1. to get float
print('heads = ', heads)
print('tails = ', tails)
print('ratio = ', ratio)
#two types of operations: action: produces local object (collect, count, mean) and transformation:produces an RDD (map, filter)
#spark job: sequence of transformations on data with a final action
#Spark application: sequence of spark jobs and other code
#
#Map vs flatmap
#map
sc.textFile('input.txt') #takes content from txt file
    .map(lambda x: x.split()) #splits by spaces. each lline is an arr, whole thing is 2D arr
    .collect()
#flatmap
sc.textFile('input.txt')
    .flatMap(lambda x: x.split()) #splits by spaces. whole thing is 1D arr
    .collect()
#pairRDD (key/value pairs wtih operations such as map and reducebykey)
sc.textFile('sales.txt').top(2) #gives first two rows of data (without column name)
sc.textFile('sales.txt').take(2) #shows first two rows (first row is column name)
sc.textFile('sales.txt')
    .map(lambda x: x.split()) 
    .filter(lambda x: not x[0].startswith('#')) #remove row that starts with # (column headers) because the first row starts with '#'
    .map(lambda x: (x[-3],float(x[-1]))) #keep only the third last and last element of the input list. then convert one of them to float. output key-value pair
    .ReduceByKey(lambda amount1, amount2: amount1 + amount2) #first column is treated as key, then data from second column are aggregated using amount1 and amount2
    .sortBy(lambda state_amount: state_amount[1], ascending = False) #sorts state by their total amount from high to low
    .collect()
#word count example
sc.textFile('input.txt')
    .flatMap(lambda line: line.split()) #splits words
    .map(lambda word: (word, 1)) #assign 1 to every word
    .reduceByKey(lambda count1, count2: count1+count2) #uses word as key, add count up
    .collect() #outputs amount of times each word appeared


#Spark machine learning
import pyspark.sql.functions as F
import pyspark as ps
from pyspark import SQLContext
#
spark = ps.sql.SparkSession.builder \
    .master('local[2]') \
    .appName('spark-ml') \
    .getOrCreate() 
sc = spark.sparkContext
SQLContext = SQLContext(sc) #want to use sql functions
#read CSV
df_aapl = SQLContext.read.csv(r'C:\Users\anton\Desktop\tut\aapl.csv',
    header = True,
    quote = '"',
    sep = ",",
    inferSchema=True)
df_aapl.show(5) #show top five rows of data
print(df_aapl.schema) #show type of data in the df (like df.info)
df_out = df_aapl.select('Date', 'Close').orderBy('Close', ascending=False) #only show to columns, and order it by close
df_out.show(5)
#
from pyspark.ml.feature import MinMaxScaler, VectorAssembler #want to vectorize all features for ML
VectorAssembler = VectorAssembler(inputCols=["Close"], outputCol="Features") #define logic. name output column to features
df_vector = VectorAssembler.transform(df_aapl) #use logic. transform df_aapl to df_vector by adding the one column (value doesn't change, jus adeed [])
df_aapl.show(5)
df_vector.show(5)
scaler = MinMaxScaler(inputCol="Features", outputCol="Scaled Features") #define logic. transform each feature individually such that it is in the given range.
scaler_model = scaler.fit(df_vector) #compute summary stats and generate MinMaxScalerModel
scaled_data = scaler_model.transform(df_vector) # rescale each feature to range [min, max]
scaled_data.select("Features","Scaled Features").show(5) #only show two columns and 5 lines
#Transformers: a generic type in Spark (such as VectorAssembler and Tokenizer). it converts one DF to another, usually by adding columns
#Estimators: abstracts the concept of a learning algorithm or any algorithm that fits or trains on data. Argument is a DF, output is type "model", which is a transformer
#
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import RegexTokenizer, HashingTF
# Prepare training documents from a list of (id, text, label) tuples.
training = spark.createDataFrame([
    (0, "spark is like hadoop mapreduce", 1.0),
    (1, "sparks light fire!!!", 0.0),
    (2, "elephants like simba", 0.0),
    (3, "hadoop is an elephant", 1.0),
    (4, "hadoop mapreduce", 1.0)
], ["id", "text", "label"])
#
regexTokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W")
hashingTF = HashingTF(inputCol="tokens", outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)
tokens = regexTokenizer.transform(training)
hashes = hashingTF.transform(tokens)
logistic_model = lr.fit(hashes) # Uses columns named features/label by default
#
# Prepare test documents, which are unlabeled (id, text) tuples.
test = spark.createDataFrame([
    (5, "simba has a spark"),
    (6, "hadoop"),
    (7, "mapreduce in spark"),
    (8, "apache hadoop")
], ["id", "text"])
# What do we need to do to this to get a prediction?
preds = logistic_model.transform(hashingTF.transform(regexTokenizer.transform(test)))
preds.select('text', 'prediction', 'probability').show()