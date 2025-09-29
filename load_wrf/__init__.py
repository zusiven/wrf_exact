"""
>>> load_wrfout = LoadWrfout(filepath = "data/wrfout_d01_2025-09-24_00:00:00")
>>> df = load_wrfout.read_var("HFX", lons=[114.5044], lats=[37.0706])
"""