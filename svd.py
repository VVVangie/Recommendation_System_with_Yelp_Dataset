#Environment:
# export PYSPARK_PYTHON=python3.6 
# export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G svd.py

import pyspark
import os
import sys
import time
import json
import csv
import math
from statistics import mean
import pandas
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV

folder_path = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]

train_file_name = folder_path + 'yelp_train.csv'
user_file = folder_path +'user.json'
business_file = folder_path +'business.json'


#train_file_name = '../resource/asnlib/publicdata/yelp_train.csv'
#test_file_name = '../resource/asnlib/publicdata/yelp_val_in.csv'

time_start = time.time()

sc = pyspark.SparkContext()
data_raw = sc.textFile(train_file_name).map(lambda x: x.split(','))
title = data_raw.take(1)
#user_id,business_id,stars
trainRDD = data_raw.filter(lambda x: x[0] != title[0][0]).map(lambda x: (x[0],x[1],x[2]))

user_id_column = trainRDD.map(lambda x: x[0]).collect()
bus_id_column = trainRDD.map(lambda x: x[1]).collect()
star_id_column = trainRDD.map(lambda x: x[2]).collect()

user_business_test = sc.textFile(test_file_name).map(lambda x: x.split(','))
title_2 = user_business_test.take(1)
test_RDD = user_business_test.filter(lambda x: x[0] != title_2[0][0]).map(lambda x: (x[0],x[1],float(0)))
testing_dataset = test_RDD.collect()
output_name = testRDD.map(lambda x: [x[0],x[1]]).collect()

train_dataframe = pandas.DataFrame({"u_id":user_id_column, "b_id":bus_id_column, "star":star_id_column})

r = Reader(rating_scale=(1,5))
training_data = Dataset.load_from_df(train_dataframe, r)
training_dataset = training_data.build_full_trainset()

#params = {'n_epochs': [25, 30], 'lr_all': [0.002, 0.005],'reg_all': [0.005, 0.01]}
#gs_model = GridSearchCV(algo_class = SVD, param_grid = params, measures=['rmse'], cv=5)
#gs_model.fit(training_data)
#print("gs.best_score:",gs.best_score['rmse'])
#print("gs.best_params:",gs.best_params['rmse'])

#svd_model = gs_model.best_estimator['rmse']
#svd_model.fit(training_dataset)

svd_model = SVD()
svd_model.fit(training_dataset)

predict_results = svd_model.test(testing_dataset)
#user: wf1GqnKQuvH-V3QN80UOOQ item: fThrN4tfupIGetkrz18JOg r_ui = 0.00   est = 4.12   {'was_impossible': False}
results_output = list(map(lambda x: x.est, predict_results))
                
with open(output_file_name, 'w')as f:
    f.write("user_id, business_id, prediction\n")
    for i in range(len(output_name)):
        f.write(str(output_name[i])[1:-1].replace("'","").replace(" ",""))
        f.write(','+str(results_output[i]))
        f.write('\n')



time_end = time.time()
print("Duration:{0:.2f}".format(time_end - time_start))



#Calculate RMSE:
with open('../resource/asnlib/publicdata/yelp_val.csv') as f1:
    f_1 = csv.reader(f1)
    header = next(f_1)
    f_true = []
    for row in f_1:
        f_true.append(float(row[2]))
    
    
with open('../work/output_svd.csv') as f2:
    f_2 = csv.reader(f2)
    header = next(f_2)
    f_pred = []
    for row in f_2:
        f_pred.append(float(row[2]))

        
    

n =len(f_true)
summary = 0
for i in range(n):
    data = f_pred[i]- f_true[i]
    summary += pow(data, 2)
Rmse = math.sqrt(summary / n)
print("RMSE is:", Rmse)



#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G svd.py '../resource/asnlib/publicdata/' '../resource/asnlib/publicdata/yelp_val.csv' output_svd.csv