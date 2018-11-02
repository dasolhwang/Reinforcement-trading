import logging
import os
import settings
import pandas as pd
import data_manager
from policy_learner import PolicyLearner

if __name__ == '__main__':
#    coin_code = 'bitcoin-2018-01-12-2018-07-15'
    coin_code = '2017-10-01-min'
    log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % coin_code)
    timestr = settings.get_time_str()
    if not os.path.exists('logs/%s' % coin_code):
        os.makedirs('logs/%s' % coin_code)
    file_handler = logging.FileHandler(filename=os.path.join(
        log_dir, "%s_%s.log" % (coin_code, timestr)), encoding='utf-8')
    stream_handler = logging.StreamHandler()
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
                        handlers=[file_handler, stream_handler], level=logging.DEBUG)

    coin_chart = data_manager.load_chart_data(
        os.path.join(settings.BASE_DIR,'data/chart_data/{}.csv'.format(coin_code)))
    print("coin chart get")
#    print(len(coin_chart))
    prep_data = data_manager.preprocess_min(coin_chart)
#    print(len(prep_data))
    training_data = data_manager.build_training_data(prep_data)
<<<<<<< HEAD
#    print(len(training_data))
    training_data = training_data[(training_data['date'] >= '2018-05-24 00:00:00')&(training_data['date'] <= '2018-07-10 00:00:00')]
=======

    training_data = training_data[(training_data['date'] > '2018-10-13 00:00:00')]
>>>>>>> 10b363ce06767c8441b9160102f77d7e0ccf3c5f
    training_data = training_data.dropna()

    features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
    coin_chart = training_data[features_chart_data]

    features_training_data = [
        'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        'close_ma5_ratio', 'volume_ma5_ratio',
        'close_ma10_ratio', 'volume_ma10_ratio',
        'close_ma20_ratio', 'volume_ma20_ratio',
        'close_ma60_ratio', 'volume_ma60_ratio',
        'close_ma120_ratio', 'volume_ma120_ratio'
    ]
    training_data = training_data[features_training_data]

    print("train data get")
    print("coin_len", len(coin_chart))
    print("train_len", len(training_data))

    policy_learner = PolicyLearner(
        coin_code=coin_code, coin_chart=coin_chart, training_data=training_data,
        min_trading_unit=0.001, max_trading_unit=0.01, delayed_reward_threshold=.1, lr=.001)
    print("policy learner start")

    policy_learner.fit(balance=1000000, num_epoches=100,
                       discount_factor=0, start_epsilon=.5)

    model_dir = os.path.join(settings.BASE_DIR, 'models/%s' % coin_code)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'model_%s.h5' % timestr)
    policy_learner.policy_network.save_model(model_path)
