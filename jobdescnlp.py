from collections import Counter
import nltk
import string
import math
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer

df = pd.read_csv('data jobs csv/DataEngineer.csv', engine='python')
print("csv imported")

#todo: pandas is slow as fucking balls!!!! switch to numpy

### Keywords
# https://towardsdatascience.com/how-to-use-nlp-in-python-a-practical-step-by-step-example-bd82ca2d2e1e
tool_keywords1 = ['python', 'pytorch', 'sql', 'mxnet', 'mlflow', 'einstein', 'theano', 'pyspark', 'solr', 'mahout', 
 'cassandra', 'aws', 'powerpoint', 'spark', 'pig', 'sas', 'java', 'nosql', 'docker', 'salesforce', 'scala', 'r',
 'c', 'c++', 'net', 'tableau', 'pandas', 'scikitlearn', 'sklearn', 'matlab', 'scala', 'keras', 'tensorflow', 'clojure',
 'caffe', 'scipy', 'numpy', 'matplotlib', 'vba', 'spss', 'linux', 'azure', 'cloud', 'gcp', 'mongodb', 'mysql', 'oracle', 
 'redshift', 'snowflake', 'kafka', 'javascript', 'qlik', 'jupyter', 'perl', 'bigquery', 'unix', 'react',
 'scikit', 'powerbi', 's3', 'ec2', 'lambda', 'ssrs', 'kubernetes', 'hana', 'spacy', 'tf', 'django', 'sagemaker',
 'seaborn', 'mllib', 'github', 'git', 'elasticsearch', 'splunk', 'airflow', 'looker', 'rapidminer', 'birt', 'pentaho', 
 'jquery', 'nodejs', 'd3', 'plotly', 'bokeh', 'xgboost', 'rstudio', 'shiny', 'dash', 'h20', 'h2o', 'hadoop', 'mapreduce', 
 'hive', 'cognos', 'angular', 'nltk', 'flask', 'node', 'firebase', 'bigtable', 'rust', 'php', 'cntk', 'lightgbm', 
 'kubeflow', 'rpython', 'unixlinux', 'postgressql', 'postgresql', 'postgres', 'hbase', 'dask', 'ruby', 'julia', 'tensor',
 'dplyr','ggplot2','esquisse','bioconductor','shiny','lubridate','knitr','mlr','quanteda','dt','rcrawler','caret','rmarkdown',
 'leaflet','janitor','ggvis','plotly','rcharts','rbokeh','broom','stringr','magrittr','slidify','rvest',
 'rmysql','rsqlite','prophet','glmnet','text2vec','snowballc','quantmod','rstan','swirl','datasciencer']

# more than 1 word
tool_keywords2 = set(['amazon web services', 'google cloud', 'sql server'])

skill_keywords1 = set(['statistics', 'cleansing', 'chatbot', 'cleaning', 'blockchain', 'causality', 'correlation', 'bandit', 'anomaly', 'kpi',
 'dashboard', 'geospatial', 'ocr', 'econometrics', 'pca', 'gis', 'svm', 'svd', 'tuning', 'hyperparameter', 'hypothesis',
 'salesforcecom', 'segmentation', 'biostatistics', 'unsupervised', 'supervised', 'exploratory',
 'recommender', 'recommendations', 'research', 'sequencing', 'probability', 'reinforcement', 'graph', 'bioinformatics',
 'chi', 'knn', 'outlier', 'etl', 'normalization', 'classification', 'optimizing', 'prediction', 'forecasting',
 'clustering', 'cluster', 'optimization', 'visualization', 'nlp', 'c#',
 'regression', 'logistic', 'nn', 'cnn', 'glm',
 'rnn', 'lstm', 'gbm', 'boosting', 'recurrent', 'convolutional', 'bayesian',
 'bayes'])

# more than 1 word
skill_keywords2 = set(['random forest', 'natural language processing', 'machine learning', 'decision tree', 'deep learning', 'experimental design',
 'time series', 'nearest neighbors', 'neural network', 'support vector machine', 'computer vision', 'machine vision', 'dimensionality reduction', 
 'text analytics', 'power bi', 'a/b testing', 'ab testing', 'chat bot', 'data mining'])

# education level
degree_dict = {'bs': 1, 'bachelor': 1, 'undergraduate': 1, 
			   'master': 2, 'graduate': 2, 'mba': 2.5, 
			   'phd': 3, 'ph.d': 3, 'ba': 1, 'ma': 2,
			   'postdoctoral': 4, 'postdoc': 4, 'doctorate': 3}


degree_dict2 = {'advanced degree': 2, 'ms or': 2, 'ms degree': 2, '4 year degree': 1, 'bs/': 1, 'ba/': 1,
				'4-year degree': 1, 'b.s.': 1, 'm.s.': 2, 'm.s': 2, 'b.s': 1, 'phd/': 3, 'ph.d.': 3, 'ms/': 2,
				'm.s/': 2, 'm.s./': 2, 'msc/': 2, 'master/': 2, 'master\'s/': 2, 'bachelor\'s/': 1}
degree_keywords2 = set(degree_dict2.keys())

### 

ps = PorterStemmer()

def prepJobDesc(desc):
	tokens = word_tokenize(desc)
	token_tag = pos_tag(tokens)
	include_tags = ['VBN', 'VBD', 'JJ', 'JJS', 'JJR', 'CD', 'NN', 'NNS', 'NNP', 'NNPS']
	filtered_tokens = [tok for tok, tag in token_tag if tag in include_tags]

	stemmed_tokens = [ps.stem(tok).lower() for tok in filtered_tokens]
	return(set(stemmed_tokens))

print("hi")

df['job_desc_wordset'] = df['Job Description'].map(prepJobDesc)
#this takes like 3000 years sheesh
print("jfdsfdafsd")

tool_keywords1_set = set([ps.stem(tok) for tok in tool_keywords1]) 
tool_keywords1_dict = {ps.stem(tok):tok for tok in tool_keywords1}

skill_keywords1_set = set([ps.stem(tok) for tok in skill_keywords1])
skill_keywords1_dict = {ps.stem(tok):tok for tok in skill_keywords1}

degree_keywords1_set = set([ps.stem(tok) for tok in degree_dict.keys()])
degree_keywords1_dict = {ps.stem(tok):tok for tok in degree_dict.keys()}

print("ok1")

# ebin for loop parsing time :DDDDDD

df["tools"] = ""
df["skills"] = ""
df["education"] = ""

tool_list = []
skill_list = []
degree_list = []

tool_ret = []
skill_ret = []
degree_ret = []

n = len(df.index)
for i in range(n):
	print(i)
	job_desc = df.iloc[i]['Job Description'].lower()
	desc_set = df.iloc[i]['job_desc_wordset']

	tool_words = tool_keywords1_set.intersection(desc_set)
	skill_words = skill_keywords1_set.intersection(desc_set)
	degree_words = degree_keywords1_set.intersection(desc_set)
	
	# check if longer keywords (more than one word) are in the job description. Match by substring.
	j = 0
	for tool_keyword2 in tool_keywords2:
		# tool keywords.
		if tool_keyword2 in job_desc:
			tool_list.append(tool_keyword2)
			j += 1
	
	k = 0
	for skill_keyword2 in skill_keywords2:
		# skill keywords.
		if skill_keyword2 in job_desc:
			skill_list.append(skill_keyword2)
			k += 1

	# search for the minimum education.
	min_education_level = 999
	for degree_word in degree_words:
		level = degree_dict[degree_keywords1_dict[degree_word]]
		min_education_level = min(min_education_level, level)
	
	for degree_keyword2 in degree_keywords2:
		# longer keywords. Match by substring.
		if degree_keyword2 in job_desc:
			level = degree_dict2[degree_keyword2]
			min_education_level = min(min_education_level, level)

	# label the job descriptions without any tool keywords.
	if len(tool_words) == 0 and j == 0:
		tool_list.append('nothing specified')
	
	# label the job descriptions without any skill keywords.
	if len(skill_words) == 0 and k == 0:
		skill_list.append('nothing specified')
	
	# If none of the keywords were found, but the word degree is present, then assume it's a bachelors level.
	if min_education_level > 500:
		if 'degree' in job_desc:
			min_education_level = 1

	# df.iloc[i]['tools'] = list(tool_words)
	# df.iloc[i]['skills'] = list(skill_words)
	# df.iloc[i]['education'] = min_education_level
	tool_ret.append(list(tool_words))
	skill_ret.append(list(skill_words))
	degree_ret.append(min_education_level)


print("ok2")

df["tools"] = tool_ret
df["skills"] = skill_ret
df["education"] = degree_ret
df.to_csv("out.csv", index=False)
#todo: this leaves the list brackets in the csv but eh..... im too lazy right now aguahgu