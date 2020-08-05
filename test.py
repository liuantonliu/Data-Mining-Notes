import pyspark.sql.functions as F
import pyspark as ps
from pyspark import SQLContext

spark = ps.sql.SparkSession.builder \
    .master('local[2]') \
    .appName('spark-ml') \
    .getOrCreate() 
sc = spark.sparkContext