import pandas as pd

class TessDataFetcher:
    def __init__(self):
        self.toi_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv"
        self.koi_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"

    def fetch_data(self):
        toi_df = pd.read_csv(self.toi_url)
        koi_df = pd.read_csv(self.koi_url)
        print("toi_df.columns",list(toi_df.columns))
        print("koi_df.columns",list(koi_df.columns))
        print("toi_df.shape",toi_df.shape)
        print("koi_df.shape",koi_df.shape)
        return toi_df, koi_df

if __name__ == "__main__":
    fetcher = TessDataFetcher()
    df = fetcher.fetch_data()


