# StockPrediction
네이버 주식 데이터를 Crawler를 통해 수집하여 StockDataSet을 구축하고 이를 통해 주식 가격을 예측하였음.

## Crawler를 통한 주식 데이터 수집
본 프로젝트에서는 Naver 금융 웹에서 특정 종목의 가격을 받아오게 하였다.

해당 프로젝트에서 선정한 종목은 "삼성전자"이다.

따라서 해당 크롤러를 통해 만들어지는 StockDataSet은 "삼성전자"에 대한 데이터셋이다.

### Crawler 작동을 위한 소프트웨어 환경
* Software
    - python 3.6 (anaconda recommended)
* Packages
    - `pip install pandas` OR `conda install pandas`
    - `pip install lxml`
    
### Crawler 작동 방법
crawler 디렉토리에서 다음과 같은 명령어를 통해 작동시킬 수 있다.

`python crawler_naver_stock.py`

StockDataSet에 수집된 데이터가 .csv 파일의 형태로 만들어진다.

## Stock Predictor 훈련 방법
본 프로젝트에서는 linear regresion, lstm을 활용하여 주식 가격을 예측하도록 하였다.

### 모델 작동을 위한 소프트웨어 환경
* Software
    - python 3.6 (anaconda recommended)
* Packages
    - `pip install pandas` OR `conda install pandas`
    - `pip install lxml`
    - `pip install tqdm` OR `conda install tqdm`
    - `pip install pytorch` OR `conda install pytorch`
    
### 모델 훈련 방법


## 모델의 Output 시각화