"""Download the RetailRocket dataset from Kaggle into data/raw.

Kaggle serves this public dataset without a login. If the request is refused,
download it manually from https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
and unzip it into data/raw.
"""
import urllib.request
import zipfile

from config import RAW

URL = "https://www.kaggle.com/api/v1/datasets/download/retailrocket/ecommerce-dataset"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    if (RAW / "events.csv").exists():
        print("data/raw/events.csv already present")
        return
    archive = RAW / "retailrocket.zip"
    print("Downloading (about 300 MB)...")
    urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(RAW)
    archive.unlink()
    print("Done:", sorted(p.name for p in RAW.iterdir()))


if __name__ == "__main__":
    main()
