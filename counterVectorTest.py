from nltk import word_tokenize

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
        str_vector.append(str_counter[str])
    text_count_vector.append(str_counter)

print(text_count_vector)
