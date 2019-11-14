import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
import numpy as np
import nltk

df = pd.read_csv('movie_data.csv', encoding='utf-8')
print(df.head(3))

count = CountVectorizer()
docs = np.array(['The sun is shining',
                 'The weather is sweet',
                 'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)

print(count.get_feature_names())
print(bag.toarray())
# print(count.vocabulary_)
# print(bag.toarray())

np.set_printoptions(precision=2)

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)  # 带有正则化

print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

# is 的 "词频逆反文档频率" 计算。
tf_is = 3
n_docs = 3
idf_is = np.log((n_docs + 1) / (3 + 1))
tfidf_is = tf_is * (idf_is + 1)
print('tf-idf of term "is" = %.2f' % tfidf_is)

# 最后一个文本的 "词频逆反文档频率" 计算
tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
print(raw_tfidf)

# l2 正则化
l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf ** 2))
print(l2_tfidf)

# 清洗文本中的 HTML 标签

df.loc[0, 'review'][-50:]  # 第一行 最后 50个字符


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


cleanedText = preprocessor(df.loc[0, 'review'][-50:])

print(cleanedText)


# 清洗全部文本
# df['review'] = df['review'].apply(preprocessor)


# 把文档处理为令牌

# 波特词干生成算法:

def tokenizer(text):
    return text.split()

#
# def tokenizer_porter(text):
#     porter = PorterStemmer()
#     return [porter.stem(word) for word in text.split()]


print(tokenizer('runners like running and thus they run'))

nltk.download('stopwords')