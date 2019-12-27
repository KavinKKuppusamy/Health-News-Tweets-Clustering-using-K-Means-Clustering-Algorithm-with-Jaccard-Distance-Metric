# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 08:03:40 2019

@author: kavin
"""
import os
import pandas as pd
import numpy as np


def data_cleaning(text):
    #pre-processing the dataset
    remove_at = filter(lambda x: x[0] != '@' and x[:7] != 'http://', text.split())
    remove_other = [w[1:].lower().strip("'") if w[0] == "#" else w.lower().strip("'") for w in remove_at]
    return remove_other


def preprocess(dataset):
    df = pd.read_csv(dataset, sep="|",encoding = "ISO-8859-1", header=None) 
    # Dropping the tweetID and timestamp column,axis-0(rows),1(columns)
    df = df.drop(df.columns[[0, 1]], axis=1)
    #Shuffling the column in the dataframe
    df = df.reindex(np.random.permutation(df.index))
    df.columns = ['tweet']
    df = df['tweet'].apply(data_cleaning)
    df = df.dropna(axis = 0, how ='any')
    return df

def jaccard_distance(t1,t2):
    t1 = set(t1)
    t2 = set(t2)
    union = len(list((t1 | t2)))
    intersection = len(list((t1 & t2)))
    return 1-(intersection/union)

def perform_kmeans(tweet_ds,K,centroids=None):
    tweet_ds = tweet_ds.sample(frac=1).reset_index(drop=True)
    #Initializing the Centroids for the the first Iteration
    if centroids==None:
        centroids = {}        
        for i in range(K):
            if(tweet_ds[i] not in list(centroids.keys())):
                centroids[i] = tweet_ds[i]
                #print("centroid {} is {}".format(i,centroids[i]))
    tweet_cluster = {i:[] for i in range(K)}
    #Assignment step : Clustering the tweets to the centroids
    for tweet in tweet_ds:
        tweet_distance = [jaccard_distance(tweet,centroids[c]) for c in centroids]
        min_distance = tweet_distance.index(min(tweet_distance))
        tweet_cluster[min_distance].append(tweet) 
    new_centroid = update_centroid(tweet_cluster,K)
    converge = False
    centroids_tweet = list(centroids.values())
    new_centroids_tweet = list(new_centroid.values())
    #Converging check - Check if the old_centroid and updated new centroids are equal
    for i in range(K):
        if(centroids_tweet[i] != new_centroids_tweet[i]):
            converge = False
            break
        else:
            converge = True
    if converge == False:
        print("Not Converged...Recomputing the Centroid")
        centroids = new_centroid.copy()
        perform_kmeans(tweet_ds,K,centroids)
    else:
        print("Converge Succeed")
        sse_total = compute_ss_error(tweet_cluster,centroids)
        print("\nThe Sum of Squared Error is ",sse_total)
        for i in range(K):
            #print("\nThe number of tweets in the cluster {0} is {1} ".format(i+1,len(tweet_cluster[i])))

            print("\n{0} : {1} ".format(i+1,len(tweet_cluster[i])))        

def update_centroid(tweet_cluster,K):
    updated_centroid = {i:[] for i in range(K)}
    for i in tweet_cluster:
        cluster = tweet_cluster[i]
        inter_cluster_dist = []
        inter_total_dist = 0
        for tweet in cluster:
            if tweet != []:
                tweet_distance = [jaccard_distance(tweet,c) for c in cluster]
                inter_total_dist = sum(tweet_distance)
                inter_cluster_dist.append(inter_total_dist)
        cluster_tweet_index = inter_cluster_dist.index(min(inter_cluster_dist))
        updated_centroid[i] = cluster[cluster_tweet_index]
    return updated_centroid

def compute_ss_error(tweet_cluster,centroids):
    sse_total  = 0
    for centroid_id in centroids.keys():
        for tweet in list(tweet_cluster[centroid_id]):
            sse_total += jaccard_distance(centroids[centroid_id],tweet)**2
    return sse_total

#Initialize the Number of Cluster 'K'
#print(os.getcwd())
os.chdir("D:/UTDallas/Project-ML/Health-News-Tweets/Health-Tweets")
K=25
dataset = "gdnhealthcare.txt"
#Processing the dataset and removing the delimiter '|' if it occurs more than 2 times in a tweet
f_in = open(dataset,"r+", encoding="utf-8")
line =f_in.readlines()
f_in.truncate(0)
f_in.seek(0)
for i in range (0,len(line)):
    if(line[i].count('|') != 2):
        line_split = line[i].split('|')
        new_line = '|'.join(line_split[:3]) + ' ' + ' '.join(line_split[3:])
        line[i] = new_line
for j in range(0,len(line)):
    l = line[j].strip()
    f_in.write(line[j])  
f_in.close()
cleaned_data = preprocess(dataset)
perform_kmeans(cleaned_data,K,centroids = None)
