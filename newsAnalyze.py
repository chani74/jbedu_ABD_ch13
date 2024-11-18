import json
import pandas as pd
import re


import warnings
warnings.filterwarnings(action="ignore")  # 경고 제거
from konlpy.tag import Okt  # 형태소 분석


# from konlpy.tag import Okt  # 형태소 분석
#
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression  # 로지스틱 회귀모델
# from sklearn.model_selection import GridSearchCV  # 하이퍼 매개변수 최적값 구하기 모듈
# from sklearn.metrics import accuracy_score  # 정확도 계산 모듈

#### 모델 훈련용 데이터


data_df = pd.read_csv("data/김건희_news_감성분석.csv", encoding="euc-kr",  engine="python")

print(data_df.info())

print(data_df["title_label"].value_counts())
print(data_df["description_label"].value_counts())

#감성 분석 결과를 각각 부정,긍정 결과로 분리하여 저장
NEG_data_df = pd.DataFrame(columns=["title","title_label","description","description_label"])
POS_data_df = pd.DataFrame(columns=["title","title_label","description","description_label"])

for index , data in data_df.iterrows():
    title = data["title"]
    description = data["description"]
    t_label = data["title_label"]
    d_label = data["description_label"]

    if d_label == 0:  # 부정 감성만 추출
        NEG_data_df = pd.concat([NEG_data_df, pd.DataFrame([[title, t_label, description, d_label]],
                                                           columns=["title", "title_label", "description",
                                                                    "description_label"])], ignore_index=True)
    else:
        POS_data_df = pd.concat([POS_data_df, pd.DataFrame([[title, t_label, description, d_label]],
                                                           columns=["title", "title_label", "description",
                                                                    "description_label"])], ignore_index=True)

NEG_data_df.to_csv('data/김건희_NEG.csv',encoding="euc-kr")
POS_data_df.to_csv('data/김건희_POS.csv',encoding="euc-kr")

POS_description = POS_data_df["description"]
POS_description_noun_tk = []

okt = Okt()
for des in POS_description:
    POS_description_noun_tk.append(okt.nouns(des))

POS_description_noun_join = []

for des in POS_description_noun_tk :
    POS_description_noun_temp=[]
    for des2 in des:
        if len(des2) > 1 :
            POS_description_noun_temp.append(des2)
    POS_description_noun_join.append(" ".join(POS_description_noun_temp))
print(POS_description_noun_join)