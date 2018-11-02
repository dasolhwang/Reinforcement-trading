import numpy as np

class Environment:

    PRICE_IDX = 4

    def __init__(self, coin_chart=None):
        self.coin_chart = coin_chart
        self.observation = None
        self.idx = -1

    def reset(self):
        self.observation = None
        self.idx = -1

    def observe(self):
        if len(self.coin_chart) > self.idx + 1:
            self.idx += 1
            self.observation = self.coin_chart.iloc[self.idx]
            return self.observation
        return None

    def get_price(self):
        if np.isnan(self.observation.close):
            return None
        if (self.observation is not None):
            return self.observation[self.PRICE_IDX]
        return None

    def set_chart_data(self, coin_chart):
        self.coin_chart = coin_chart
