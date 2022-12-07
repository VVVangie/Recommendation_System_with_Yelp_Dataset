#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task2_1.py

import pyspark
import time
import sys
import math
from numpy import *
import csv

train_file_name = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]

#train_file_name = '../resource/asnlib/publicdata/yelp_train.csv'
#test_file_name = '../resource/asnlib/publicdata/yelp_val_in.csv'

time_start = time.time()

sc = pyspark.SparkContext()
data_raw = sc.textFile(train_file_name).map(lambda x: x.split(','))
title = data_raw.take(1)
#(business,user,star):[('vxR_YV0atFxIxfOnF9uHjQ', 'gTw6PENNGl68ZPUpYWP50A', '5.0'), ('o0p-iTC5yTBV5Yab_7es4g', 'iAuOpYDfOTuzQ6OPpEiGwA', '4.0'),...]
dataRDD = data_raw.filter(lambda x: x[0] != title[0][0]).map(lambda x: (x[1],x[0],x[2]))
#print(dataRDD.collect()[:2])


#{id:name}: {9813: 'jIH2fzQfXY7RW3C3W1Z1iQ', 9814: 'kzd8RMZKbDoZF133UjNRVA'...}
user_raw = dataRDD.map(lambda x: x[1]).distinct().zipWithIndex()
user_map = user_raw.collectAsMap()
user_map_broadcast = sc.broadcast(user_map)
user_search_map = user_raw.map(lambda x:(x[1],x[0])).collectAsMap()
#user_search_map_broadcast = sc.broadcast(user_search_map)

business_raw = dataRDD.map(lambda x:x[0]).distinct().zipWithIndex()
business_map = business_raw.collectAsMap()
business_map_broadcast = sc.broadcast(business_map)
business_total = len(business_map)
business_search_map = business_raw.map(lambda x: (x[1], x[0])).collectAsMap()
#business_search_map_broadcast = sc.broadcast(business_search_map)

#[(business_id,{user_id:star,u2..})]: [(12262, {0: 4.0, 5905: 2.0, 7783: 5.0, ...}),...)] change to {business_id,{user_id:star,u2..}}: {12990: {683: 4.0, 4677: 5.0, 9528: 4.0, 7894: 5.0,...}}..
business_user_star_RDD = dataRDD.map(lambda x: (business_map[x[0]],(user_map[x[1]],float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
business_user_star_broadcast = sc.broadcast(business_user_star_RDD)


#business_pair_for_pearson = sc.parallelize([i for i in range(business_total)]).flatMap(lambda x:[(x,j) for j in range(x+1,business_total)])
#business_pair_total = 305,823,546

def pair_filter(business_id1, business_id2):
    user_co1 = set(business_user_star_broadcast.value[business_id1].keys())
    user_co2 = set(business_user_star_broadcast.value[business_id2].keys())
    return len(user_co1 & user_co2) #################################### Can be improved


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

#business_pair_pearson = business_pair_for_pearson.filter(lambda x: pair_filter(x[0],x[1])).map(lambda x:(x[0], x[1], pearson_corelation(x[0],x[1]))).filter(lambda x: x[2] > 0).map(lambda x: ((x[0],x[1]),x[2])).collectAsMap() #x[0]<x[1]
#business_pair_pearson_broadcast = sc.broadcast(business_pair_pearson)
#print(business_pair_pearson_broadcast.value())

   
user_rated_business = dataRDD.map(lambda x: (user_map[x[1]],business_map[x[0]])).groupByKey().collectAsMap()
#user_rated_business_broadcast = sc.broadcast(user_rated_business)

user_business_star_RDD = dataRDD.map(lambda x: (user_map[x[1]],(business_map[x[0]],float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
#user_business_star_broadcast = sc.broadcast(user_business_star_RDD)


def rating(user_business_pair):
    if (user_business_pair[0] not in user_map_broadcast.value) or (user_business_pair[1] not in business_map_broadcast.value):
        return ((user_business_pair[0],user_business_pair[1],3))
    else:
        user_test = user_map[user_business_pair[0]]
        business_test = business_map[user_business_pair[1]]    
        candidate_corelated_buiness = user_rated_business[user_test]
        similarity_map = {}
        for b in candidate_corelated_buiness:
            if pair_filter(b,business_test) < 30: ############################
                continue
            p_similarity = pearson_corelation(b,business_test)
            if p_similarity <= 0:
                continue
            user_score = user_business_star_RDD[user_test][b]
            similarity_map[p_similarity] = user_score
        chosen_pair = sorted(similarity_map.items(), key=lambda x:x[0])#[:6]##########################Can be improved
        if len(chosen_pair) == 0:
            user_average_score = mean(list(user_business_star_RDD[user_test].values()))
            #return (user_search_map[user_test],business_search_map[business_test],3)
            return (user_search_map[user_test],business_search_map[business_test],user_average_score)
        else:
            up_data = 0
            down_data = 0
            for p in chosen_pair:
                d = p[0] * p[1]
                up_data += d
                down_data += p[0]
            #if up_data == 0 or down_data == 0 :
                #return (user_search_map[user_test],business_search_map[business_test],3)
            score = up_data / down_data
            return (user_search_map[user_test],business_search_map[business_test],score)
        
#def filtering(pair):
#    if (pair[0] not in user_map_broadcast.value) or (pair[1] not in business_map_broadcast.value):
#        return False
#    return True

                                  
user_business_test = sc.textFile(test_file_name).map(lambda x: x.split(','))
title_2 = user_business_test.take(1)
user_business_testRDD = user_business_test.filter(lambda x: x[0] != title[0][0]).map(lambda x: rating(x))

output = user_business_testRDD.collect()
#.map(lambda x: (user_search_map[x[0]], business_search_map[x[1]], x[2]))

#.map(lambda x: (user_map[x[0]],business_map[x[1]]))
#print(user_business_testRDD.collect()[:3])

with open(output_file_name, 'w')as f:
    f.write("user_id, business_id, prediction\n")
    for i in output:
        f.write(str(i)[1:-1].replace("'","").replace(" ",""))
        f.write('\n')

time_end = time.time()
print("Duration:{0:.2f}".format(time_end - time_start))

#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task2_1.py '../resource/asnlib/publicdata/yelp_train.csv' '../resource/asnlib/publicdata/yelp_val.csv' 'output2_1.csv'





#with open('../resource/asnlib/publicdata/yelp_val.csv') as f1:
#    f_1 = csv.reader(f1)
#    header = next(f_1)
#    f_true = []
#    for row in f_1:
#        f_true.append(float(row[2]))
    
    
#with open('../work/output2_1.csv') as f2:
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




                                                             

