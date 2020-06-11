# StockPrediction
네이버 주식 데이터를 Crawler를 통해 수집하여 StockDataSet을 구축하고 이를 통해 주식 가격을 예측하였음.

## Crawler를 통한 주식 데이터 수집
본 프로젝트에서는 Naver 금융 웹에서 특정 종목의 가격을 받아오게 하였다.

### Crawler 작동을 위한 소프트웨어 환경
* Software
    - python 3.6 (anaconda recommended)
* Packages
    - `pip install pandas` OR `conda install pandas`
    - `pip install lxml`
    
### Crawler 작동 방법
crawler 디렉토리에서 다음과 같은 명령어를 통해 작동시킬 수 있다.

`python crawler_naver_stock.py --code 091990`

이 때, code Argument는 크롤링하고 싶은 주식 종목 코드를 의미한다.

StockDataSet에 수집된 데이터가 .csv 파일의 형태로 만들어진다.

## Stock Predictor 훈련 방법
본 프로젝트에서는 lstm을 활용하여 주식 가격을 예측하도록 하였다.

### 모델 훈련을 위한 Train Data와 Test Data 나누기
본 프로젝트에서는 가장 최근 2년의 데이터를 Test Data로 설정하고 나머지 데이터를 Train Data로 설정하였다.

우선, Train Data와 Test Data를 나누기 위해서는 "./data"에 크롤링을 통해 수집된 데이터가 .csv 파일 형태로 준비되어 있어야 한다.

다음과 같은 명령어를 통해 Train Data와 Test Data를 얻을 수 있다.

`python build_train_eval.py`

이 명령을 통해 생성되는 파일은 다음과 같다.

* train.csv
* test.csv

### 모델 작동을 위한 소프트웨어 환경
* Software
    - python 3.6 (anaconda recommended)
* Packages
    - `pip install pandas` OR `conda install pandas`
    - `pip install tqdm` OR `conda install tqdm`
    - `pip install pytorch` OR `conda install pytorch`
    - `pip install scikit-learn` OR `conda install scikit-learn`
    - `pip install matplotlib` OR `conda install matplotlib`
    - `pip install numpy` OR `conda install numpy`
    
### 모델 훈련 방법

다음과 같은 명령어를 통해 수행할 수 있다.

`python run_stock_predict.py --model_type lstm --do_train --do_eval --data_dir ./data --per_gpu_train_batch_size=8 --per_gpu_eval_batch_size=8 --learning_rate 1e-3 --num_train_epochs 8.0 --output_dir ./output --output_mode regression --target volume --domain samsung --window_size 10`

훈련을 완료하고 나면 "./output"에 model.out 파일이 생성되고 이 파일은 학습된 모델의 Parameter들을 저장하고 있다.

## 모델의 Output 시각화

학습이 완료되면 plot이 생성되어 실제 Test Set에 대한 주가 그래프와 모델의 예측 주가 그래프가 그려질 것이다.