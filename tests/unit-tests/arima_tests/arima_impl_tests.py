import unittest
from dsmodels.arima.arima_impl import ARIMA
import pandas

class TestArimaImpl(unittest.TestCase):

    def setUp(self):
        print("one time initialised")
        self.path="dsmodels/test/files"
        self.filename="sarimax"
        self.version="0.0.0.1.test"

    # def test_Arima_imple_paraMs(self):
    #     arima_model=ARIMA(q=0,d=0,p=0,trend='c',method=None,start_params=None)
    #
    #     assert arima_model.p == 0

    def test_train(self):
        pd_data=pandas.read_csv('tests/files/international-airline-passengers.csv', sep=';', error_bad_lines=False, header=0)
        # A bit of pre-processing to make it nicer
        print(pd_data.to_string())
        pd_data['Month'] = pandas.to_datetime(pd_data['Month'], format='%Y-%m-%d')
        pd_data.set_index(['Month'], inplace=True)

        train_data = pd_data['1949-01-01':'1959-12-01']
        arima_model = ARIMA(q=range(0,2), d=range(0,2), p=range(0,4), trend='c', method=None, start_params=None)
        result=arima_model.fit(train_data)
        arima_model.save_model(result)
