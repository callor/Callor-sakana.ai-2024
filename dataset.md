# 학습용 데이터 셋
## enwik8용 데이터 세트 카드
- http://mattmahoney.net/dc/enwik8.zip
- enwik8 데이터 세트는 2006년 3월 3일 영어 위키백과 XML 덤프의 첫 100,000,000(100M)바이트이며 일반적으로 모델의 데이터 압축 기능을 측정하는 데 사용됩니다.


## 작은 셰익스피어, 캐릭터 수준
- https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- 다운로드 불가
- 옛날의 찰리 셰익스피어로 유명한 작은 셰익스피어 :) 캐릭터 수준에서 다루었습니다.
- train.bin에는 1,003,854개의 토큰이 있습니다
- val.bin에는 111,540개의 토큰이 있습니다

## Text8
- http://mattmahoney.net/dc/text8.zip
- word2vec 알고리즘에서 사용된 공식 데이터 셋
- LSA, LDA 에 비해 최근에 발표된 알고리즘(2016년)
- LSA, LDA 는 Vector Space Model 또는 미리 구성된 Corpus 를 사용한 알고리즘인데 비해 word2vec 는 deep learning 기술을 이용하여 구현한 모델
- python 의 gensim 패키지를 이용하여 구현 테스트 가능
- Skip Gram 과 CBOW 2가지 모델방식으로 구분된다

- skip gram : 한개의 단어에서 여러개의 단어를 유추하는 방식
- CBOW : 여러개의 단어에서 한개의 단어를 유추하는 방식

- 예) 왕 - 남자 + 여자 = ? , 결과로 `여왕`을 유추
- 참고1 : https://www.tensorflow.org/tutorials/text/word2vec?hl=ko
- 참고2 : https://wikidocs.net/22660
- 참고3 : https://velog.io/@aqaqsubin/Word2Vec-Word-to-Vector
- 참고4 : https://word2vec.kr/search/
- 참고5 : https://goldenplanet.co.kr/our_contents/blog?number=859&pn=2
- 참고6 : https://ko.wikipedia.org/wiki/Word2vec