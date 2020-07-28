# kakao_Recommend

목표 : Brunch를 이용하여 사용자의 취향에 맞는 글을 예측 할 수 있을까?

## Data  구성
- Metadata(643,104) 
- user (310,758) 
- read(3,507,097) 
- magazine(27,967)
- contents 

EDA = https://colab.research.google.com/drive/1wVPIKrDRd2BqPC6h4jBonKMtkMTNoene?usp=sharing

## 아이디어
- 구상 아이디어
  - 유명한 사람의 글을 본다.     
  - 구독 중인 작가의 글을 본다.   
  - 같은 magazine 내의 글을 본다.  
  - 독자와 비슷한 소비 성향의 다른 독자 글을 본다.
  - 독자가 읽은 과거 소비 성향 반영한다.
  - 최신 위주의 글을 본다.

## Embedding
- Doc2Vec ( 128dimension)

자세한 내용은 PPT 및 Code 참조

ppt = https://docs.google.com/presentation/d/1N0uJ-fcEhHJ99gTouH3ZZxCScl59hcnYmCjIKfBX1jA/edit?usp=sharing

출처 - https://arena.kakao.com/c/6

참조

https://arena.kakao.com/c/6/data
https://arena.kakao.com/forum/topics/170
https://github.com/jihoo-kim/BrunchRec
https://github.com/kakao-arena/brunch-article-recommendation
https://lovit.github.io/nlp/representation/2018/03/26/word_doc_embedding/
https://jisoo-coding.tistory.com/27
https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45530.pdf
https://github.com/lsjsj92/recommender_system_with_Python
http://hoondongkim.blogspot.com/2019/03/recommendation-trend.html
