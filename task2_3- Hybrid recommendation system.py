#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task2_3.py
import pyspark
import sys
import time
import xgboost
import json
import csv
import math
from statistics import mean
import numpy as np



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
#(business,user,star):[('vxR_YV0atFxIxfOnF9uHjQ', 'gTw6PENNGl68ZPUpYWP50A', '5.0'), ('o0p-iTC5yTBV5Yab_7es4g', 'iAuOpYDfOTuzQ6OPpEiGwA', '4.0'),...]
dataRDD = data_raw.filter(lambda x: x[0] != title[0][0]).map(lambda x: (x[1],x[0],x[2]))


#{id:name}: {9813: 'jIH2fzQfXY7RW3C3W1Z1iQ', 9814: 'kzd8RMZKbDoZF133UjNRVA'...}
user_raw = dataRDD.map(lambda x: x[1]).distinct().zipWithIndex()
user_map = user_raw.collectAsMap()
user_map_broadcast = sc.broadcast(user_map)
user_search_map = user_raw.map(lambda x:(x[1],x[0])).collectAsMap()


business_raw = dataRDD.map(lambda x:x[0]).distinct().zipWithIndex()
business_map = business_raw.collectAsMap()
business_map_broadcast = sc.broadcast(business_map)
business_total = len(business_map)
business_search_map = business_raw.map(lambda x: (x[1], x[0])).collectAsMap()


#[(business_id,{user_id:star,u2..})]: [(12262, {0: 4.0, 5905: 2.0, 7783: 5.0, ...}),...)] change to {business_id,{user_id:star,u2..}}: {12990: {683: 4.0, 4677: 5.0, 9528: 4.0, 7894: 5.0,...}}..
business_user_star_RDD = dataRDD.map(lambda x: (business_map[x[0]],(user_map[x[1]],float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
business_user_star_broadcast = sc.broadcast(business_user_star_RDD)


def pair_filter(business_id1, business_id2):
    user_co1 = set(business_user_star_broadcast.value[business_id1].keys())
    user_co2 = set(business_user_star_broadcast.value[business_id2].keys())
    return len(user_co1 & user_co2) 


def pearson_corelation(business_id1,business_id2):
    user_collect1 = business_user_star_broadcast.value[business_id1]
    user_collect2 = business_user_star_broadcast.value[business_id2]
    user_intersection = set(user_collect1.keys()) & set(user_collect2.keys())
    user_intersection_num = len(user_intersection)
    
    def get_average(user_collect):
        sum_for_average = 0
        for u in user_intersection:
            sum_for_average += user_collect[u]
        average_rating = sum_for_average / user_intersection_num
        return average_rating
    
    upside_data_1 = {}
    upside_data_2 = {}
    for u in user_intersection:
        upside_data_1[u] = user_collect1[u] - get_average(user_collect1)
        upside_data_2[u] = user_collect2[u] - get_average(user_collect2)
    
    upside_data=0
    downside_sum_1 = 0
    downside_sum_2 = 0
    for u in user_intersection:
        upside_data += upside_data_1[u] * upside_data_2[u]
        downside_sum_1 += pow(upside_data_1[u], 2)
        downside_sum_2 += pow(upside_data_2[u], 2)
    if downside_sum_1 == 0 or downside_sum_2 == 0:
        return 0
    pearson_value = upside_data / math.sqrt(downside_sum_1 * downside_sum_2)
    return pearson_value

   
user_rated_business = dataRDD.map(lambda x: (user_map[x[1]],business_map[x[0]])).groupByKey().collectAsMap()

user_business_star_RDD = dataRDD.map(lambda x: (user_map[x[1]],(business_map[x[0]],float(x[2])))).groupByKey().mapValues(dict).collectAsMap()

#chosen_length = []
def rating(user_business_pair):
    if (user_business_pair[0] not in user_map_broadcast.value) or (user_business_pair[1] not in business_map_broadcast.value):
        return 3
    else:
        user_test = user_map[user_business_pair[0]]
        business_test = business_map[user_business_pair[1]]    
        candidate_corelated_buiness = user_rated_business[user_test]
        similarity_map = {}
        for b in candidate_corelated_buiness:
            if pair_filter(b,business_test) < 30: 
                continue
            p_similarity = pearson_corelation(b,business_test)
            if p_similarity <= 0:
                continue
            user_score = user_business_star_RDD[user_test][b]
            similarity_map[p_similarity] = user_score
        chosen_pair = sorted(similarity_map.items(), key=lambda x:x[0])
        #chosen_length.append(len(chosen_pair))
        if len(chosen_pair) == 0:
            user_average_score = mean(list(user_business_star_RDD[user_test].values()))
            #user_average_score_weighted = user_average_score *0.3
            return user_average_score
        else:
            up_data = 0
            down_data = 0
            for p in chosen_pair:
                d = p[0] * p[1]
                up_data += d
                down_data += p[0]
            
            score = up_data / down_data
            #if len(chosen_pair) <10:
                #score_weighted = score * 0.3
            #else:
                #score_weighted = score*0.7
            return score
        #(user_search_map[user_test],business_search_map[business_test],score)

#print("chosen_length:",chosen_length)        
                                  
user_business_test = sc.textFile(test_file_name).map(lambda x: x.split(','))
title_2 = user_business_test.take(1)
user_business_testRDD = user_business_test.filter(lambda x: x[0] != title_2[0][0]).map(lambda x: rating(x))
#only output CF score:
output_CF = user_business_testRDD.collect()
#print("output_CF:", output_CF[:2])







user_feature_RDD = sc.textFile(user_file).map(lambda x: json.loads(x)).map(lambda x: (x["user_id"], [x["review_count"], x["average_stars"]])).collectAsMap()
business_feature_RDD = sc.textFile(business_file).map(lambda x :json.loads(x)).map(lambda x:(x["business_id"], [x["review_count"], x["stars"]])).collectAsMap()


users = sc.textFile(user_file).map(lambda x: json.loads(x)).map(lambda x: x["user_id"]).collect()
businesses = sc.textFile(business_file).map(lambda x :json.loads(x)).map(lambda x:x["business_id"]).collect()
trainRDD = data_raw.filter(lambda x: x[0] != title[0][0]).map(lambda x: (x[0],x[1],x[2]))


training_x_data = trainRDD.map(lambda x: [user_feature_RDD[x[0]],business_feature_RDD[x[1]]]).map(lambda x:[x[0][0],x[0][1],x[1][0],x[1][1]]).collect()
x_train = np.asarray(training_x_data)
y_train = dataRDD.map(lambda x: x[2]).collect()

      
model = xgboost.XGBRegressor ()
#n_estimators=20, max_depth=10, learning_rate=0.8
model.fit(x_train, y_train)

testRDD = user_business_test.filter(lambda x: x[0] != title_2[0][0])
output_name = testRDD.map(lambda x: [x[0],x[1]]).collect()
testing_x_data = testRDD.map(lambda x: [user_feature_RDD[x[0]],business_feature_RDD[x[1]]]).map(lambda x:[x[0][0],x[0][1],x[1][0],x[1][1]]).collect()
x_test = np.asarray(testing_x_data)

y_predict = model.predict(x_test)

weighted_score_output = []
#review_count_list =[]
for i in range(len(output_name)):
    #review_count_list.append(testing_x_data[i][2])
    if testing_x_data[i][2] < 30:
        mol_score = y_predict[i] * 0.5
        cf_score = output_CF[i] * 0.5
        weighted_score = mol_score + cf_score
        weighted_score_output.append(weighted_score)
    else:
        mol_score = y_predict[i] * 0.8
        cf_score = output_CF[i] * 0.2
        weighted_score = mol_score + cf_score
        weighted_score_output.append(weighted_score)
#print("review_count_list:",review_count_list)


with open(output_file_name, 'w')as f:
    f.write("user_id, business_id, prediction\n")
    for i in range(len(output_name)):
        f.write(str(output_name[i])[1:-1].replace("'","").replace(" ",""))
        f.write(','+str(weighted_score_output[i]))
        f.write('\n')


time_end = time.time()
print("Duration:{0:.2f}".format(time_end - time_start))



with open('../resource/asnlib/publicdata/yelp_val.csv') as f1:
    f_1 = csv.reader(f1)
    header = next(f_1)
    f_true = []
    for row in f_1:
        f_true.append(float(row[2]))
    
    
with open('../work/output2.csv') as f2:
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



#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task2_3.py '../resource/asnlib/publicdata/' '../resource/asnlib/publicdata/yelp_val.csv' output2.csv