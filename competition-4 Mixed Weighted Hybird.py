''' 
Method Description:
I combined four model scores to do a weighted hybird, also contain a switch. The four models are item-based CF, user_based CF, xgboost and SVD. I wish that based on four kinds of models, the accuracy can be more stable when facing the testing data.
I thought the number of the reviews one business got may influence the influence from four models, so I do a switch when the review number is bigger than 30. I also tried to make the number of the reviews one user written or the xgboost model score as the switch basis, but it seems that the one I choose works the best.
I also set weights for different models. The performance of xgboost is the best in general, so it got more weight. Then I tried several times to adjust the parameters. The RMSE just improved a little.

Error Distribution:
>=0 and <1: 101280
>=1 and <2: 33797
>=2 and <3: 6237
>=3 and <4: 730
>=4: 0

RMSE: 
0.9832233614539793

Execution Time:
220.91                                                                                                                                                                   

'''
#Environment:
# export PYSPARK_PYTHON=python3.6 
# export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G competition.py

import pyspark
import sys
import time
import xgboost
import json
import csv
import math
from statistics import mean
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor
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
business_berated_user = dataRDD.map(lambda x: (business_map[x[0]],user_map[x[1]])).groupByKey().collectAsMap()
business_user_star_RDD = dataRDD.map(lambda x: (business_map[x[0]],(user_map[x[1]],float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
business_user_star_broadcast = sc.broadcast(business_user_star_RDD)


user_rated_business = dataRDD.map(lambda x: (user_map[x[1]],business_map[x[0]])).groupByKey().collectAsMap()
user_business_star_RDD = dataRDD.map(lambda x: (user_map[x[1]],(business_map[x[0]],float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
user_business_star_broadcast = sc.broadcast(user_business_star_RDD)


def pair_filter(business_id1, business_id2):
    user_co1 = set(business_user_star_broadcast.value[business_id1].keys())
    user_co2 = set(business_user_star_broadcast.value[business_id2].keys())
    return len(user_co1 & user_co2) 

def u_based_pair_filter(user_id1, user_id2):
    bus_co1 = set(user_business_star_broadcast.value[user_id1].keys())
    bus_co2 = set(user_business_star_broadcast.value[user_id2].keys())
    return len(bus_co1 & bus_co2)

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


def u_based_pearson_corelation(user_id1, user_id2):
    bus_collect1 = user_business_star_broadcast.value[user_id1]
    bus_collect2 = user_business_star_broadcast.value[user_id2]
    bus_intersection = set(bus_collect1.keys()) & set(bus_collect2.keys())
    bus_intersection_num = len(bus_intersection)
    
    def got_average(bus_collect):
        sum_average = 0
        for b in bus_intersection:
            sum_average += bus_collect[b]
        avg_rating = sum_average / bus_intersection_num
        return avg_rating
    
    up_data_1 = {}
    up_data_2 = {}
    for b in bus_intersection:
        up_data_1[b] = bus_collect1[b] - got_average(bus_collect1)
        up_data_2[b] = bus_collect2[b] - got_average(bus_collect2)
        
    up_data = 0
    down_sum_1 = 0
    down_sum_2 = 0
    for b in bus_intersection:
        up_data += up_data_1[b] * up_data_2[b]
        down_sum_1 += pow(up_data_1[b], 2)
        down_sum_2 += pow(up_data_2[b], 2)
    if down_sum_1 == 0 or down_sum_2 == 0:
        return 0
    pea_value = up_data / math.sqrt(down_sum_1 * down_sum_2)
    return pea_value
    


#chosen_length = []
def rating(user_business_pair):
    #if (user_business_pair[0] not in user_map_broadcast.value) or (user_business_pair[1] not in business_map_broadcast.value):
        #return 3
    if (user_business_pair[0] not in user_map_broadcast.value):
        if (user_business_pair[1] in business_map_broadcast.value):
            bus_idx = business_map[user_business_pair[1]]
            business_avg = mean(list(business_user_star_RDD[bus_idx].values()))
            return business_avg
        else:
            return 3
    elif (user_business_pair[1] not in business_map_broadcast.value):
        if (user_business_pair[0] in user_map_broadcast.value):
            user_idx = user_map[user_business_pair[0]]
            user_avg = mean(list(user_business_star_RDD[user_idx].values()))
            return user_avg
        else:
            return 3
            
    else:
        user_test = user_map[user_business_pair[0]]
        business_test = business_map[user_business_pair[1]]    
        candidate_corelated_buiness = user_rated_business[user_test]
        similarity_map = {}
        for b in candidate_corelated_buiness:
            if pair_filter(b,business_test) < 30: 
                continue
            p_similarity = pearson_corelation(b,business_test) # w score
            if p_similarity <= 0:
                continue
            user_score = user_business_star_RDD[user_test][b] #user's rating to similar business
            similarity_map[p_similarity] = user_score # w_score: user_score
        chosen_pair = sorted(similarity_map.items(), key=lambda x:x[0]) #key: w score, value: pair
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
            return score
      

def u_based_rating(user_business_pair):
    if (user_business_pair[0] not in user_map_broadcast.value) or (user_business_pair[1] not in business_map_broadcast.value):
        return 3
    else:
        user_test = user_map[user_business_pair[0]]
        business_test = business_map[user_business_pair[1]]
        candidate_corelated_user = business_berated_user[business_test]
        similar_map = {}
        for u in candidate_corelated_user:
            if u_based_pair_filter(u, user_test) < 60:
                continue
            p_sim = u_based_pearson_corelation(u,user_test) #w score
            if p_sim <= 0:
                continue
            bus_score = business_user_star_RDD[business_test][u] # u's rating to business
            u_avg = mean(list(user_business_star_RDD[u].values()))
            similar_map[p_sim] = [bus_score,u_avg]
        chose_pair = sorted(similar_map.items(), key=lambda x:x[0]) #sorted by w_score: x[0]
        if len(chose_pair) == 0:
            business_avg_score = mean(list(business_user_star_RDD[business_test].values()))
            return business_avg_score
        else:
            up_d = 0
            down_d = 0
            for p in chose_pair:
                d = p[0] * (p[1][0] - p[1][1])
                up_d += d
                down_d += p[0]
            user_avg = mean(list(user_business_star_RDD[user_test].values()))
            score = (up_d / down_d) + user_avg
            return score
    
    
    
    
user_business_test = sc.textFile(test_file_name).map(lambda x: x.split(','))
title_2 = user_business_test.take(1)
user_business_testRDD = user_business_test.filter(lambda x: x[0] != title_2[0][0]).map(lambda x: rating(x))
#only output CF score:
output_CF = user_business_testRDD.collect()#item_based
#print("output_CF:", output_CF[:2])


business_user_testRDD = user_business_test.filter(lambda x: x[0] != title_2[0][0]).map(lambda x: u_based_rating(x))
output_user_based_CF = business_user_testRDD.collect()#user_based




#Build XGRegressor Model:
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


#SVD:

#user_id,business_id,stars
trainRDD = data_raw.filter(lambda x: x[0] != title[0][0]).map(lambda x: (x[0],x[1],x[2]))

user_id_column = trainRDD.map(lambda x: x[0]).collect()
bus_id_column = trainRDD.map(lambda x: x[1]).collect()
star_id_column = trainRDD.map(lambda x: x[2]).collect()


test_RDD = user_business_test.filter(lambda x: x[0] != title_2[0][0]).map(lambda x: (x[0],x[1],float(0)))
testing_dataset = test_RDD.collect()


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



weighted_score_output = []
#review_count_list =[]
for i in range(len(output_name)):
    #review_count_list.append(testing_x_data[i][2])
    if testing_x_data[i][2] < 30:
        mol_score = y_predict[i] * 0.7
        cf_score = output_CF[i] * 0.05
        u_cf_score = output_user_based_CF[i] * 0.05
        svd_score = results_output[i] * 0.2
        weighted_score = mol_score + cf_score + u_cf_score + svd_score
        weighted_score_output.append(weighted_score)
    else:
        mol_score = y_predict[i] * 0.8
        cf_score = output_CF[i] * 0.05
        u_cf_score = output_user_based_CF[i] * 0.05
        svd_score = results_output[i] * 0.1
        weighted_score = mol_score + cf_score + u_cf_score + svd_score
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





                
#Calculate RMSE:
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
level_0_1 = 0
level_1_2 = 0
level_2_3 = 0
level_3_4 = 0
level_4_p = 0
for i in range(n):
    data = abs(f_pred[i]- f_true[i])
    if data >= 0 and data < 1:
        level_0_1 += 1
    elif data >= 1 and data < 2:
        level_1_2 += 1
    elif data >= 2 and data < 3:
        level_2_3 += 1
    elif data >= 3 and data < 4:
        level_3_4 += 1
    else:
        level_4_p += 1
     
    summary += pow(data, 2)
Rmse = math.sqrt(summary / n)
print("RMSE is:", Rmse)
print("level:", "0-1:", level_0_1, "1-2:",level_1_2, "2-3:", level_2_3, "3-4:", level_3_4, "4+:",level_4_p) 

    
    
    
    
    
#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G competition.py '../resource/asnlib/publicdata/' '../resource/asnlib/publicdata/yelp_val.csv' output2.csv