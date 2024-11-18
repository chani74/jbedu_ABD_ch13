import warnings
warnings.filterwarnings(action='ignore')# 경고제거

import pandas as pd
import re
from konlpy.tag import Okt  # 형태소 분석
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV



nsmc_train_df = pd.read_csv("data/ratings_train.txt",encoding='utf8', sep="\t", engine='python')
nsmc_train_df.head()

# document. 내용이 null 이 아닌 내용만 찾아서 저장 // 결측치 제거
nsmc_train_df=nsmc_train_df[nsmc_train_df['document'].notnull()]
nsmc_train_df.info()

nsmc_train_df['label'].value_counts()

#
nsmc_train_df['document'] = nsmc_train_df['document'].apply(lambda x: re.sub(r'[^ㄱ-ㅎ|가-힣]+', "", x))
nsmc_train_df.head()

########################################
## nsmc_test_df
nsmc_test_df = pd.read_csv("data/ratings_test.txt",encoding='utf8', sep="\t", engine='python')
nsmc_test_df.head()

# document. 내용이 null 이 아닌 내용만 찾아서 저장 // 결측치 제거
nsmc_test_df=nsmc_test_df[nsmc_test_df['document'].notnull()]
nsmc_test_df.info()

nsmc_test_df['label'].value_counts()

#
nsmc_test_df['document'] = nsmc_test_df['document'].apply(lambda x: re.sub(r'[^ㄱ-ㅎ|가-힣]+', "", x))
nsmc_test_df.head()
########################################




okt = Okt()

def okt_tokenizer(text):    # ㅎ여태소 분석 - > 형태소 단위로 토큰화
    tokens = okt.morphs(text)
    return tokens

tfidf = TfidfVectorizer(tokenizer=okt_tokenizer, ngram_range=(1,2), min_df=3, max_df=0.9,token_pattern=None)
tfidf.fit(nsmc_train_df['document'])
nsmc_train_tfidf =  tfidf.transform(nsmc_train_df['document'])

print(nsmc_train_tfidf)


SA_lr = LogisticRegression(random_state=0,max_iter=500)
SA_lr.fit(nsmc_train_tfidf,nsmc_train_df['label'])

params={'C':[1,3,3.5,4,4.5,5]} #최적화 하이퍼 매개변수 후보군
SA_lr_grid_cv = GridSearchCV(SA_lr, param_grid=params,cv=3,scoring='accuracy',verbose=1)
#최적화 하이퍼 파라미터 찾아서 정확도가 제일 높은 최적 모델로 생성
SA_lr_grid_cv.fit(nsmc_train_tfidf,nsmc_train_df['label'])

print(SA_lr_grid_cv.best_params_ , round(SA_lr_grid_cv.best_score_, 4 ))
SA_lr_best = SA_lr_grid_cv.best_estimator_


#%%
nsmc_test_tfidf = tfidf.transform(nsmc_test_df['document'])
test_predict = SA_lr_best.predict(nsmc_test_tfidf)
#%%
from sklearn.metrics import accuracy_score
print(f'감성 분석 정확도 :' , round(accuracy_score(nsmc_test_df['label'],test_predict),3))

st = "웃자 ^o^ 오늘은 좋은 날이 될 것 같은 예감100%! ^^*"
st= re.compile(r'[ㄱ-ㅎ|가-힣]+').findall(st)
print(st)
st=[" ".join(st)]
print(st)

#1) 입력텍스트 피처 벡터화
st_tfidf = tfidf.transform(st)
#2) 최적 감성 분석 모델에 적용하여 감성 분석 평가
st_predict = SA_lr_best.predict(st_tfidf)
#3) 예측값 출력하기
if(st_predict==0):
    print(st,"->> 부정 감성")
else :
    print(st,"--> 긍정 감성")


