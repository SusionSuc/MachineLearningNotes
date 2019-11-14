import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv('movie_data.csv', encoding='utf-8')

count = CountVectorizer(stop_words='english', max_df=.1, max_features=5000)

X = count.fit_transform(df['review'].values)

lda = LatentDirichletAllocation(n_components=10,
                                random_state=123,
                                learning_method='batch')

X_topics = lda.fit_transform(X)


print(lda.components_.shape)

