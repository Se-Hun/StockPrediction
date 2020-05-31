import os
import time
import random

import pandas as pd

def get_url(code):
    url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code) # url

    return url

def preprocess_dataset(dataset):
    # Remove NaN Values
    dataset = dataset.dropna()

    # Retype Float -> Int
    dataset[['종가', '전일비', '시가', '고가', '저가', '거래량']] = \
        dataset[['종가', '전일비', '시가', '고가', '저가', '거래량']].astype(int)

    # Retype String -> Datetime
    dataset['날짜'] = pd.to_datetime(dataset['날짜'])

    # Sort in ascending order by 날짜
    dataset = dataset.sort_values(by=['날짜'], ascending=True)

    # Sort Index For Row
    dataset.index = [ i for i in range(len(dataset.index))]

    return dataset

def data_save(output_dir, filename, dataset):
    path = os.path.join(output_dir, filename)

    dataset.to_csv(path, index=False)

def random_time():
    return random.randrange(3, 4) + random.random()

def main():
    output_dir = "./StockDataSet"

    code = '005930'  # code for Samsung Electronics
    target_bundle_num = 2 #200

    dataset = pd.DataFrame()

    url = get_url(code)

    count = 0
    while count < target_bundle_num:
        # 1페이지부터 20페이지를 한 묶음(bundle)으로 봄
        start_page = count * 20 + 1  # 1 page 부터
        end_page = count * 20 + 20  # 20 page 까지

        for page in range(start_page, end_page+1):
            page_url = '{url}&page={page}'.format(url=url, page=page)

            dataset = dataset.append(pd.read_html(page_url, header=0)[0], ignore_index=True)

        count = count + 1
        time.sleep(random_time()) # => Naver가 Crawling을 막아서 요청을 조금 느리게 해야함..

    # preprocessing dataset
    dataset = preprocess_dataset(dataset)

    start_date = dataset['날짜'][0]
    start_date = str(start_date).split()[0]
    end_date = dataset['날짜'][len(dataset.index) - 1]
    end_date = str(end_date).split()[0]

    # DataFrame -> CSV File
    filename = "data_" + str(start_date) + "_" + str(end_date) + ".csv"
    data_save(output_dir, filename, dataset)

    print("Start Date : {}".format(start_date))
    print("End Date : {}".format(end_date))

    # print(dataset.head())

    return None

if __name__ == '__main__':
    main()
