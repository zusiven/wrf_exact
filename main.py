from load_wrf.core import LoadWrfout
from wztools import get_logger
import logging
import numpy as np


logger = get_logger(level=logging.DEBUG)

def main():
    load_wrfout = LoadWrfout(filepath = "data/wrfout_d01_2025-09-24_00:00:00")
    # df = load_wrfout.read_var("SWDOWN", lons=[114], lats=[38]) # T2 rh2 slp PBLH RAIN
    # df = load_wrfout.read_var("SWDOWN", lons=[114.5044], lats=[37.0706]) # T2 rh2 slp PBLH RAIN
    # df = load_wrfout.read_var("SWDOWN", lons=[114.5044], lats=[37.0706]) # T2 rh2 slp PBLH RAIN
    df = load_wrfout.read_var("HFX", lons=[114.5044], lats=[37.0706]) # T2 rh2 slp PBLH RAIN
    print(df)
    # ds = load_wrfout.ds
    df.write_csv('data/a.csv')


if __name__ == "__main__":
    main()
