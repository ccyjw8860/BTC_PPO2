from db.mongoConnect import MongoConnect


class DBClass:
    def __init__(self):
        mongo = MongoConnect()
        self.client = mongo.client
        self.CoinDB = self.client['Coin']
        self.XRPCollection = self.CoinDB['XRP']
        self.XRP1HourCollection = self.CoinDB['XRP_1Hour']
        self.XRPUSDTPCollection = self.CoinDB['XRPUSDT.P']
        self.XRPBackupCollection = self.CoinDB['XRP_Backup']
        self.XRPDollarBarCollection = self.CoinDB['XRP_DollarBar_1M']
        self.BTCUSDTPCollection = self.CoinDB['BTCUSDT.P']
        self.BTCUSDTP_5MinCollection = self.CoinDB['BTCUSDT.P_5Min']
        self.INVERSED_BTCUSDTP_5MinCollection = self.CoinDB['INVERSED_BTCUSDT_P_5M']
        self.BTCUSDTP_input = self.CoinDB['BTCUSDT.P_input']
        self.BTCUSDTP_5Min_Aggregated = self.CoinDB['BTCUSDTP_5Min_Aggregated']
        self.BTCUSDTP_input2 = self.CoinDB['BTCUSDT.P_input2']
        self.BTCUSDTP_input3 = self.CoinDB['BTCUSDT.P_input3']
        self.INVERSED_BTCUSDTP_5Min_Aggregated = self.CoinDB['INVERSED_BTCUSDTP_5Min_Aggregated']