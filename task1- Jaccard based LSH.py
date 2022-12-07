#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task1.py
import pyspark
import sys
import json
import time
import random
input_path = sys.argv[1]
output_path = sys.argv[2]
#input_path = '../resource/asnlib/publicdata/yelp_train.csv'

hash_total = 100  # b*r=hash*total
band_total = 25
row_num = 4
#hash function: f(x)= (ax + b) % m
prime_num = []
i = 2
for i in range(2,551):
    j = 2
    for j in range(2,i):
        if (i % j == 0):
            break
    else:
        prime_num.append(i)
a_list = prime_num[:100]
#for x in range(hash_total):
#    a_list.append(random.randint(0,200))
b_list = []
for y in range(hash_total):
    b_list.append(random.randint(1,500))

time_start = time.time()


sc = pyspark.SparkContext()
sc.setLogLevel("WARN")

data_raw = sc.textFile(input_path).map(lambda x: x.split(','))
title = data_raw.take(1)
dataRDD = data_raw.filter(lambda x: x[0] != title[0][0]).map(lambda x: (x[0],x[1]))

user_raw = dataRDD.map(lambda x: x[0]).distinct().zipWithIndex()
user_map = user_raw.collectAsMap()
user_total = len(user_map)

business_raw = dataRDD.map(lambda x:x[1]).distinct().zipWithIndex()
business_map = business_raw.collectAsMap()
business_total = len(business_map)
#business_search_map = business_raw.map(lambda x: {x[1]:x[0]}).collectAsMap()
business_search_map = business_raw.map(lambda x: (x[1], x[0])).collectAsMap()
business_search_map_broadcast = sc.broadcast(business_search_map)

#b_u_RDD = dataRDD.map(lambda x:(business_map_broadcast.value[x[0]], user_map_broadcast.value[x[1]]))
matrixRDD = dataRDD.map(lambda x:(business_map[x[1]],user_map[x[0]])).groupByKey().mapValues(set)
#matrix_map = matrixRDD.map(lambda x: {x[0]:x[1]}).collectAsMap()
matrix_map = matrixRDD.collectAsMap()
matrix_map_broadcast = sc.broadcast(matrix_map)
#print(matrix_map.collect()[:5])

    
def minhash (user_index):
    hash_index = []
    for i in range(hash_total):
        hash_index.append(set(map(lambda x: (a_list[i] * x + b_list[i])% user_total, user_index)))
    return [min(i) for i in hash_index]

minhash_signature = matrixRDD.mapValues(lambda x: minhash(x))
#print(minhash_signature.collect()[:5])
minhash_signature_map = minhash_signature.map(lambda x: {x[0]:x[1]})

def partition_into_band(minhash_signature):
    bands = []
    for i in range(band_total):
        bands.append(minhash_signature[i*row_num : (i+1)*row_num])
    return bands

partitioned_signature_RDD = minhash_signature.mapValues(lambda x: partition_into_band(x)).collectAsMap()
#print(partitioned_signature_RDD.collect()[:5])
#[business_id,[[],[],[]]

#pair_list =[]
#for i in range(business_total):
#    for j in range(i+1, business_total):
#        pair_list.append(set(i,j))
candidate_pair_raw = sc.parallelize([i for i in range(business_total)]).flatMap(lambda x:[(x,j) for j in range(x+1,business_total)])

#Identify business1,business2 as candidate pair if they are identical in at least one band
def identify_filter(business_1, business_2):
    business_1_signature = partitioned_signature_RDD[business_1]
    business_2_signature = partitioned_signature_RDD[business_2]
    for i in range(band_total):
        if business_1_signature[i] == business_2_signature[i]:
            return True
    return False
                              
candidate_pair = candidate_pair_raw.filter(lambda x: identify_filter(x[0],x[1]))
#print(candidate_pair.collect())                                     

def jaccard_similarity (candi_business_1, candi_business_2):
    users_1 = matrix_map_broadcast.value[candi_business_1]
    users_2 = matrix_map_broadcast.value[candi_business_2]
    sim_rate = len(users_1 & users_2) / len(users_1 | users_2)
    return sim_rate #return sim_rate >= similarity_threshold
                                
    
similarity_threshold = 0.5 
final_pair = candidate_pair.map(lambda x: (x[0], x[1], jaccard_similarity(x[0],x[1]))).filter(lambda x: x[2] >= similarity_threshold).map(lambda x:(business_search_map_broadcast.value[x[0]], business_search_map_broadcast.value[x[1]], x[2])).collect()
#print(output[:5])
output =[]
for i in final_pair:
    b =list(i)
    business = sorted(b[:2])
    business.append(b[2])
    output.append(tuple(business))
   
with open(output_path, 'w')as f:
    f.write("business_id_1, business_id_2, similarity\n")
    for i in sorted(output):
        f.write(str(i)[1:-1].replace("'","").replace(" ",""))
        f.write('\n')


time_end = time.time()
print("Duration:{0:.2f}".format(time_end - time_start))
#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task1.py '../resource/asnlib/publicdata/yelp_train.csv' output1.py