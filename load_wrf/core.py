from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime
import datetime as dt
from wztools import error_info
import logging
import numpy as np
import polars as pl
from scipy.spatial import KDTree
from wrf import getvar, ALL_TIMES, extract_times, latlon_coords



logger = logging.getLogger(name="loadwrfout")

class LoadWrfout:
    def __init__(self, filepath: str | Path) -> None:
        # 1. 检查文件是否存在
        if not Path(filepath).exists:
            raise FileNotFoundError(f"{filepath} 不存在")
        
        # 2. 加载文件
        self.ds = Dataset(filepath)

        # 3. 初始化变量
        self.lons = None
        self.lats = None
        self.start_time = None
        self.end_time = None

        # 4. 加载通用变量
        # 4.1 加载时间
        all_times = extract_times(self.ds, timeidx=ALL_TIMES)
        self.all_times = [(dt.datetime.strptime(str(time), "%Y-%m-%dT%H:%M:%S.%f000") + dt.timedelta(hours=8)).strftime("%Y-%m-%d %H:00") for time in all_times]

        # 4.2 通过温度变量加载 经纬度
        tmp_data = getvar(self.ds, varname="T2", timeidx=0)
        self.ds_lats, self.ds_lons = latlon_coords(tmp_data)
        self.lats_flat = np.array(self.ds_lats).flatten()
        self.lons_flat = np.array(self.ds_lons).flatten()

    def read_var(self, varname: str,
                    lons: list | None = None, lats: list | None = None, # 地点
                    start_time: datetime = None, end_time: datetime = None, # 时间
                ):
        """ 
        ## 说明
        - 变量: https://wrf-python.readthedocs.io/en/latest/diagnostics.html#diagnostic-table
        - 默认参数:
            - 温度 T2
            - 湿度 rh2 
            - 海平面气压 slp
            - 风速
            - 边界层高度 PBLH 
            - 地表显热通量 hfx
            - 降雨量 rain
        """
        try:
            # 1. 加载配置
            lons = lons if lons is not None else self.lons
            lats = lats if lats is not None else self.lats

            start_time = start_time if start_time is not None else self.start_time
            end_time = end_time if end_time is not None else self.end_time

            # 2. 读取数据
            logger.debug(f"正在加载 [{varname}] 变量")
            
            # 2.1 判断纬度 有没有高度概念
            if varname != "rain":
                data = getvar(self.ds, varname=varname, timeidx=ALL_TIMES) # [:, :, :] # Time: 85, south_north: 106, west_east: 124


            else:
                logger.info("降雨: ")
                data_rainc = getvar(self.ds, varname="RAINC", timeidx=ALL_TIMES)
                data_rainnc = getvar(self.ds, varname="RAINNC", timeidx=ALL_TIMES)
                data_rainsh = getvar(self.ds, varname="RAINSH", timeidx=ALL_TIMES)

                # 原始总降雨
                data_cumulative = data_rainc + data_rainnc + data_rainsh
                # 差分得出瞬时降雨
                data_instant = np.diff(data_cumulative, axis=0)

                # 在前面加上 0 时刻的累积值，保持长度一致, (最开始时间是 wrf 预热结果，反正都要舍弃的)
                data = np.concatenate(([data_cumulative[0]], data_instant), axis=0)

            shape_num = len(data.shape)
            logger.debug(f"{data.shape = }, {shape_num = }")
            if shape_num == 3:
                data = np.array(data[:, :, :])
            elif shape_num == 4:
                data = np.array(data[:, 0, :, :])

            nt, ny, nx = data.shape
            logger.debug(f"{nt = }, {nx = }, {ny = }")
            data_flat = data.reshape(nt, -1)

            dfs = []
            for i in range(nt):
                df_sub = pl.DataFrame(
                    {
                        "time": [self.all_times[i]] * (ny * nx),
                        "lon": self.lons_flat,
                        "lat": self.lats_flat,
                        varname: data_flat[i, :],
                    }
                )
                dfs.append(df_sub)
            
            df = pl.concat(dfs)
            round_cols = [varname, "lon", "lat"]

            df = df.with_columns(
                *[pl.col(col).cast(pl.Float64).round(4) for col in round_cols],
                pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M")
            )

            # logger.info(df)
            df = self.filter_data(lons, lats, start_time, end_time, df)

            return df

        except:
            error_infos = error_info(show_details=True)
            logger.error(error_infos)

    def load_config(self,
                    lons: list | None = None, lats: list | None = None, # 地点
                    start_time: datetime = None, end_time: datetime = None, # 时间
                ):
        self.lons = lons
        self.lats = lats
        self.start_time = start_time
        self.end_time = end_time    

    def filter_data(self, lons, lats, start_time, end_time, df: pl.DataFrame):
        # 按时间处理
        if start_time is not None and end_time is not None:
            df = df.filter(
                pl.col("time").is_between(start_time, end_time)
            )

        if lons is not None and lats is not None:
            lonlats = [f"{lons[i]}-{lats[i]}" for i in range(len(lons))]
            # print(lonlats)
            
            _time = df["time"][-1]
            _df = df.filter(
                pl.col("time") == _time
            )
            _lons = _df["lon"]
            _lats = _df["lat"]
            _points = np.column_stack((_lons, _lats))

            # 构建KDTree
            tree = KDTree(_points)

            # 计算 res 
            _lon0 = _lons[0]
            _lat0 = _lats[0]

            distance, index = tree.query([_lon0, _lat0], 2)
            _res = distance[1] * 3 / 2

            dfs = []
            for i in range(len(lons)):
                # 循环每一个点, 可以先最大最小筛选一遍，这个数据小，没有必要
                target_lon = lons[i]
                target_lat = lats[i]
                lon_max = target_lon + _res
                lon_min = target_lon - _res
                lat_max = target_lat + _res
                lat_min = target_lat - _res

                _df_fix = _df.filter(
                    (pl.col("lon").is_between(lon_min, lon_max))
                    &
                    (pl.col("lat").is_between(lat_min, lat_max))
                )

                _df_lons = _df_fix["lon"]
                _df_lats = _df_fix["lat"]
                _df_points = np.column_stack((_df_lons, _df_lats))
                tree2 = KDTree(_df_points)

                distance2, index2 = tree2.query([target_lon, target_lat])

                lon_nearest = _df_lons[int(index2)]
                lat_nearest = _df_lats[int(index2)]

                _df_nearest = df.filter(
                    (pl.col("lon") == lon_nearest) 
                    &
                    (pl.col("lat") == lat_nearest) 
                ).drop("lon", "lat").with_columns(
                    pl.lit(target_lon).cast(pl.Float64).alias("lon"),
                    pl.lit(target_lat).cast(pl.Float64).alias("lat"),
                )

                dfs.append(_df_nearest)

        res_df = pl.concat(dfs)

        return res_df


