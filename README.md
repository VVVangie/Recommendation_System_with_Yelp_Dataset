# Recommendation_System_with_Yelp_Dataset
Recommendation system: based on Yelp dataset (the given stars rated by a user for a business), to predict a specific user's rating for one specific business. Dataset:https://drive.google.com/drive/folders/1u3BVDSSK3nxEFIQcaCdHnK4HYplRKHFx?usp=sharing

#Task 1:  Implementing the Locality Sensitive Hashing algorithm with Jaccard similarity to calculate the similarity between two businesses in Yelp dataset with pyspark and Python, finally get precision >= 0.99 and recall >= 0.97.

#Task 2_1: Implementing an item-based recommendation system with Pearson similarity using pyspark and Python to predict a specific user's rating for one specific business in Yelp dataset, getting RMSE under 1.09.

#Task 2_2: Using XGBregressor(a regressor based on the decision tree) to train a model with extra features from businesses and users themselves like the average stars rated by a user, the number of reviews and so on, getting RMSE under 1.00.

#Task 2_3: Created a hybird recommendation system with weighted score for Item-based CF and XGBregressor to predict , getting RMSE under 0.99.

#Competition: Improving the recommenddation system to get RMSE under 0.983, with runtime under 230 seconds. Combinded SVD, Item-based CF, content-baed CF and XGBregressor to get a weighted and switech algorithm.
