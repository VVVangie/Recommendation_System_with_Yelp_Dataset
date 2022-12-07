#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task2_2.py

import pyspark
import sys
import time
import xgboost
import json
import csv
import math
import numpy as np
from statistics import mean
#business_id: "stars", "review_count" (business.json)
#user_id: "user_id", "review_count", "average_stars":2.0 (user.json)

folder_path = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]
#folder_path = '../resource/asnlib/publicdata/'
#test_file_name = '../resource/asnlib/publicdata/yelp_val.csv'

user_file = folder_path +'user.json'
business_file = folder_path +'business.json'
train_file_name = folder_path +'yelp_train.csv'

time_start = time.time()


sc = pyspark.SparkContext()
sc.setLogLevel("WARN")

user_feature_RDD = sc.textFile(user_file).map(lambda x: json.loads(x)).map(lambda x: (x["user_id"], [x["review_count"], x["average_stars"]])).collectAsMap()
business_feature_RDD = sc.textFile(business_file).map(lambda x :json.loads(x)).map(lambda x:(x["business_id"], [x["review_count"], x["stars"]])).collectAsMap()
#print(user_feature_RDD["lzlZwIpuSWXEnNS91wxjHw"])

users = sc.textFile(user_file).map(lambda x: json.loads(x)).map(lambda x: x["user_id"]).collect()
businesses = sc.textFile(business_file).map(lambda x :json.loads(x)).map(lambda x:x["business_id"]).collect()


data_train = sc.textFile(train_file_name).map(lambda x: x.split(','))
title = data_train.take(1)
dataRDD = data_train.filter(lambda x: x[0] != title[0][0]).map(lambda x: (x[0],x[1],x[2]))
#user_rating = dataRDD.map(lambda x: x[0]).collect()
#business_rated = dataRDD.map(lambda x: x[1]).collect()
#user_business_star_RDD = dataRDD.map(lambda x: (x[0],(x[1],float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
#business_user_star_RDD = dataRDD.map(lambda x: (x[1],(x[2],float(x[2])))).groupByKey().mapValues(dict).collectAsMap()



#def data_dataframe(user, business):
#    if user in users:
#        user_feature = user_feature_RDD[user]
#    else:
#        user_feature = [0,2.5]
#    if business in businesses:
#        business_feature = business_feature_RDD[business]
#    else:
#        business_feature = [0,2.5]
#    data_collection = user_feature + business_feature 
#    return zip(data_collection)

   
training_x_data = dataRDD.map(lambda x: [user_feature_RDD[x[0]],business_feature_RDD[x[1]]]).map(lambda x:[x[0][0],x[0][1],x[1][0],x[1][1]]).collect()
x_train = np.asarray(training_x_data)
y_train = dataRDD.map(lambda x: x[2]).collect()
#print("train:",x_train[:2])
      
model = xgboost.XGBRegressor ()
#n_estimators=20, max_depth=10, learning_rate=0.8
model.fit(x_train, y_train)


user_business_test = sc.textFile(test_file_name).map(lambda x: x.split(','))
title_2 = user_business_test.take(1)
user_business_testRDD = user_business_test.filter(lambda x: x[0] != title[0][0])
output_name = user_business_testRDD.map(lambda x: [x[0],x[1]]).collect()
testing_x_data = user_business_testRDD.map(lambda x: [user_feature_RDD[x[0]],business_feature_RDD[x[1]]]).map(lambda x:[x[0][0],x[0][1],x[1][0],x[1][1]]).collect()
x_test = np.asarray(testing_x_data)

y_predict = model.predict(x_test)
#print("output_name:",output_name[:2])

with open(output_file_name, 'w')as f:
    f.write("user_id, business_id, prediction\n")
    for i in range(len(output_name)):
        f.write(str(output_name[i])[1:-1].replace("'","").replace(" ",""))
        f.write(','+str(y_predict[i]))
        f.write('\n')



time_end = time.time()
print("Duration:{0:.2f}".format(time_end - time_start))


#with open('../resource/asnlib/publicdata/yelp_val.csv') as f1:
#    f_1 = csv.reader(f1)
#    header = next(f_1)
#    f_true = []
#    for row in f_1:
#        f_true.append(float(row[2]))
    
    
#with open('../work/output1.csv') as f2:
#    f_2 = csv.reader(f2)
#    header = next(f_2)
#    f_pred = []
#    for row in f_2:
#        f_pred.append(float(row[2]))
    

#n =len(f_true)
#summary = 0
#for i in range(n):
#    data = f_pred[i]- f_true[i]
#    summary += pow(data, 2)
#Rmse = math.sqrt(summary / n)
#print("RMSE is:", Rmse)










#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task2_2.py '../resource/asnlib/publicdata/' '../resource/asnlib/publicdata/yelp_val.csv' output1.py