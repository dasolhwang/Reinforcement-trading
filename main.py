import logging
import os
import settings
import data_manager
from policy_learner import PolicyLearner

if __name__ == '__main__':
    coin_code = 'bitcoin_min'

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
    prep_data = data_manager.preprocess(coin_chart)
    training_data = data_manager.build_training_data(prep_data)

    training_data = training_data[(training_data['date'] >= '2017-01-01 00:00:00')&(training_data['date'] <= '2018-01-01 00:00:00')] # 추세를 사람이 판단해서 (상승, 하락, 보합)
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

    policy_learner = PolicyLearner(
        coin_code=coin_code, coin_chart=coin_chart, training_data=training_data,
        min_trading_unit=0.01, max_trading_unit=3, delayed_reward_threshold=.1, lr=.001)

    policy_learner.fit(balance=10000000, num_epoches=5,
                       discount_factor=0, start_epsilon=.5)

    model_dir = os.path.join(settings.BASE_DIR, 'models/%s' % coin_code)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'model_%s.h5' % timestr)
    policy_learner.policy_network.save_model(model_path)
