import time
import requests
import duckdb
import pandas as pd
import numpy as np
from load_bts_data import bts_data

df = bts_data #bts_data is just an empty df right now, fix this!

# Base API url
API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Same start and end date for every airport
DATE_START, DATE_END = min(pd.to_datetime(df["FlightDate"])), max(pd.to_datetime(df["FlightDate"]))

# Same weather variables for every airport
WEATHER_VARS = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "apparent_temperature_mean",
    "apparent_temperature_max", "apparent_temperature_min", "wind_speed_10m_max", "wind_gusts_10m_max",
    "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration", "precipitation_sum", "rain_sum",
    "snowfall_sum", "precipitation_hours", "weather_code"
]

# To prevent hitting a rate limit, set a batch size to pass (lat,long) values at
BATCH_SIZE = 10

# Need a list of the unique (lat,long) pairs as a proxy for airport code to enter into API
lat_long_pairs = list(df["lat_long"].unique()) # .unique() has consistent output so shouldn't need to sort unless df changes

# Initialize an empty list to hold weather data for each (lat,long) pair across time range
weather_dfs = []

def unpack_response():
# For a given batch (list of BATCH_SIZE pairs of lat and long) and a response
# load each df, add lat_long column, append to intermediate storage list, concat intermediate storage list,
# return resulting concated df -> (append that concated df to high level storage list)
    pass


def call_open_meteo(lat_long_pairs = lat_long_pairs, batch_size = BATCH_SIZE, 
                    base_url = API_URL, start_date = DATE_START, end_date = DATE_END):
    for i in range(0, len(lat_long_pairs), batch_size):
        current_batch = lat_long_pairs[i : i + BATCH_SIZE]
        latitudes = ",".join(str(pair[0]) for pair in current_batch)
        longitudes = ",".join(str(pair[1]) for pair in current_batch)
        
        params = {
            "latitude":latitudes,
            "longitude":longitudes,
            "start_date":DATE_START.strftime("%Y-%m-%d"),
            "end_date":DATE_END.strftime("%Y-%m-%d"),
            "daily":WEATHER_VARS
        }
    
    pass

def resume_call_open_meteo():
    pass

# For each unique (lat,long) pair, get weather data for set time period, store data frame with (lat,long) in a list
# There's a good chance this hits a rate limit -> I'll have to maybe make a function that stores what we do have 
# and maybe remembers where to pick up from
for i in range(0,len(lat_long_pairs), BATCH_SIZE):
    current_batch = lat_long_pairs[i : i + BATCH_SIZE]
    latitudes = ",".join(str(pair[0]) for pair in current_batch)
    longitudes = ",".join(str(pair[1]) for pair in current_batch)
    
    params = {
        "latitude":latitudes,
        "longitude":longitudes,
        "start_date":DATE_START.strftime("%Y-%m-%d"),
        "end_date":DATE_END.strftime("%Y-%m-%d"),
        "daily":WEATHER_VARS
    }
    
    print(f"Fetching batch {i//BATCH_SIZE + 1} / {((len(lat_long_pairs)-1) // BATCH_SIZE) + 1}")
    response = requests.get(API_URL, params = params)
    
    if response.status_code == 200:
        batch_len = len(current_batch)
        lat_long_weather_list = []
        
        for j in range(0, batch_len):
            weather_df_j = pd.DataFrame(response.json()[j]["daily"])
            weather_df_j["lat_long"] = [current_batch[j]] * len(weather_df_j)
            lat_long_weather_list.append(weather_df_j)
        
        lat_long_weather_df = pd.concat(lat_long_weather_list)
        weather_dfs.append(lat_long_weather_df)
        
    elif response.status_code == 429:
        print("429 Status -- Retrying after 60 seconds")
        time.sleep(60)
        response = requests.get(API_URL, params = params)
                
        if response.status_code != 200:
            print(response.text)
            raise Exception(f"Request still failed with status code: {response.status_code}")
                
        batch_len = len(current_batch)
        lat_long_weather_list = []
                
        for j in range(0, batch_len):
            weather_df_j = pd.DataFrame(response.json()[j]["daily"])
            weather_df_j["lat_long"] = [current_batch[j]] * len(weather_df_j)
            lat_long_weather_list.append(weather_df_j)
                
        lat_long_weather_df = pd.concat(lat_long_weather_list)
        weather_dfs.append(lat_long_weather_df)
   
    else:
        print(response.text)
        raise Exception(f"Request failed with status code: {response.status_code}")

    time.sleep(60)

