from nltk import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

text = ['I am a element school student',' And I am a boy.']

text_token_list = []

for str in text:
    text_token_list.append(word_tokenize(str))

print(text_token_list)


str_counter = Counter()
for str in text_token_list:
    str_counter.update(str)

print(str_counter)


text_bag=[]

for key, value in str_counter.items():
    text_bag.append(key)    # 중복 제거된 단어만 리스트에 저장

print(text_bag)

text_count_vector = []

for str in text_token_list:
    str_vector = []
    for word in str:
        str_vector.append(str_counter[word])
    text_count_vector.append(str_counter)

print(text_count_vector)


print("-------------------------")
text = ['I am a element school student And I am a boy.']

count_vector = CountVectorizer()
count_vector_array = count_vector.fit_transform(text).toarray()

print(count_vector_array)
print(count_vector.vocabulary_)


print("-------------------------")

text = ['I am a great great element school student And I am a boy.']

tfidfitm = TfidfVectorizer().fit(text)
tfidfitm_array = tfidfitm.transform(text).toarray()
print(tfidfitm_array)
print(tfidfitm.vocabulary_)
