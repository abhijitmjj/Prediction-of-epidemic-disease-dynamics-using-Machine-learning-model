# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:48:35 2018

@author: Abhijit
"""

#from api_keys import POSTGRES_PWD
import dill
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import NVARCHAR

def csv_it(dataframe, filebase):
    dataframe.to_csv('./csv/'+filebase+'.csv', sep=',', index=False, encoding='utf-8')
    return

def pkl_it(dataframe, filebase):
    with open('./pkl/'+filebase+'.pkl', 'wb') as fh:
        dill.dump(dataframe, fh)

    return

def sql_it(dataframe, filebase, dbname='zika'): #check!
    postgres_str = 'postgresql://abhijit:{}@127.0.0.1:5432/{}'.format(POSTGRES_PWD, dbname)
    engine = create_engine(postgres_str)
    dataframe.to_sql(filebase, engine, if_exists='replace', dtype={col_name: NVARCHAR for col_name in filebase})
    return

def save_it(dataframe, filebase, dbname='zika'):
    csv_it(dataframe, filebase)
    pkl_it(dataframe, filebase)
    #sql_it(dataframe, 'sql_' + filebase, dbname=dbname)
    return
import pandas as pd
import numpy as np
from glob import glob
#from csv_pkl_sql import save_it

pd.options.mode.chained_assignment = None
location_files = glob('./zika/*/*Places.csv')
locations = pd.concat([pd.read_csv(x, encoding = 'ANSI') 
                       for x in location_files], axis=0).reset_index(drop=True)

data_file_locations = glob('./zika/*/*/data/*.csv')
data_locations = pd.concat([pd.read_csv(x, usecols=[1], encoding='ANSI').drop_duplicates() 
                            for x in data_file_locations], axis=0).drop_duplicates().reset_index(drop=True)

# Drop the locations that don't exist in any data files
mask = locations.location.isin(data_locations.location)
locations = locations[mask]

# District data will be difficult to incorporate into the model, so drop for now
mask = locations.location_type.isin(['country', 'region', 'district']).pipe(np.invert)
locations = locations.loc[mask]

locations = locations.dropna(axis=1, how='all')

location_key = locations[['location', 'location_type']]
location_key[['country', 'province', 'county']] = location_key.location.str.split(r"""-""", expand=True)
location_key['city'] = location_key.county
def map_locations(x):
    location_mapper = {'state':'province',
                       'municipality':'city',
                       'department':'province',
                       'Region':'province',
                       'Collectivity':'province',
                       'territory':'province'
                      }
    if x in location_mapper.keys():
        return location_mapper[x]
    else:
        return x
    
location_key['location_type'] = location_key.location_type.apply(lambda x: map_locations(x))

# Fix the US Virgin Islands entries
mask = ( location_key.county.isnull() & 
         (location_key.location_type=='county') &
         (location_key.country=='United_States_Virgin_Islands')
        )

location_key.loc[mask, 'county'] = location_key.loc[mask, 'province']
location_key.loc[mask, 'province'] = 'Virgin Islands'
location_key.loc[mask, 'country'] = 'United States'


mask = ( location_key.province.isnull() & 
         (location_key.location_type=='province'))
location_key.loc[mask, 'province'] = 'Virgin Islands'
location_key.loc[mask, 'country'] = 'United States'


mask = (location_key.location=='United_States-US_Virgin_Islands')
location_key.loc[mask, 'province'] = 'Virgin Islands'

# Fix remaining counties (mainly in Ecuador and Panama)
mask = ( location_key.county.isnull() & 
         (location_key.location_type=='county'))

location_key.loc[mask, 'county'] = location_key.loc[mask, 'province']
location_key.loc[mask, 'province'] = None

# Move cities to correct column
mask = ( location_key.city.isnull() & 
         (location_key.location_type=='city'))

location_key.loc[mask, 'city'] = location_key.loc[mask, 'county']
location_key.loc[mask, 'county'] = None

# More fixes for cities
mask = ( location_key.city.isnull() & 
         (location_key.location_type=='city'))

location_key.loc[mask, 'city'] = location_key.loc[mask, 'province']
location_key.loc[mask, 'province'] = None

# Drop unknown cities
location_key = location_key[location_key.city.isin(['Unknown','Not_Reported']).pipe(np.invert)]

# Fix for Dade County Florida
mask = location_key.location=='United_States-Florida-Miami-Dade_County'
location_key.loc[mask, 'county'] = 'Dade_County'
location_key.loc[mask, 'city'] = 'Miami'
# Fix for Santiago Del Estero Argentina
location_key.loc[location_key.location=='Argentina-Sgo_Del_Estero', 'province'] = 'Santiago Del Estero'
location_key.loc[location_key.location=='Argentina-CABA', 'province'] = 'Ciudad de Buenos Aires'

# Remove county name
location_key['county'] = location_key.county.str.replace('_County','')

location_key = location_key[location_key.county.isin(['Unknown','Not_Reported']).pipe(np.invert)]

# Remove all underscores
for col in ['country', 'province', 'county', 'city']:
    location_key[col] = location_key[col].str.replace('_', ' ')
    
# For checking the data 50 lines at a time
i=32
nsize = 50
location_key.iloc[i*nsize:(i+1)*nsize]
save_it(location_key, '00_cleaned_city_names')
os.mkdir("./csv")
os.mkdir("./pkl")
save_it(location_key, '00_cleaned_city_names')
#%%
coord = pd.read_csv("./csv/World_Cities_Location_table.csv",delimiter=";", header=None,
                    names="id, country, city, latitude, longitude, altitude".split(", "))

import pandas as pd
import numpy as np
import requests
import json
import re
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options




GOOGLE_API_KEY="**********" #insert API Key here


pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt
def get_latitude_longitude(df_row):
    subtype = df_row.location_type
    second_str = df_row[subtype].replace(' ', '+')
    country_str = df_row.country.replace(' ', '+')
    
    url = 'https://maps.googleapis.com/maps/api/geocode/json?address=+{},+{}&key={}'.format(second_str, 
                                                                                            country_str, 
                                                                                            GOOGLE_API_KEY)
    
    try:
        response = requests.get(url).text
        lat_lng = json.loads(response)['results'][0]['geometry']['location']
        lat_lng_df = pd.Series({'lat':lat_lng['lat'], 'lng':lat_lng['lng']})
    except:
        lat_lng_df = pd.Series({'lat':np.NaN, 'lng':np.NaN})
        
    time.sleep(1)
    return lat_lng_df
location_key[['latitude','longitude']] = location_key.apply(lambda x: get_latitude_longitude(x), axis=1)
save_it(location_key, '01_latitude_longitude_google')
import copy
location_key_out = copy.deepcopy(location_key)
mask = ((location_key_out.longitude<-140)|(location_key_out.longitude>-20)).pipe(np.invert)
new_location_key=location_key_out.loc[mask]
save_it(new_location_key, '01_latitude_longitude_google')
#%%
import pandas as pd
import numpy as np
import requests
import re
from io import StringIO

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
location_key_plain = pd.read_pickle('./pkl/00_cleaned_city_names.pkl')
#from csv_pkl_sql import save_it
data_countries = new_location_key.country.unique()

tables = pd.read_html(requests.get('http://www.fallingrain.com/world/index.html').text)
countries = pd.DataFrame({'full':tables[0].values.ravel()}).dropna()
countries[['abbrev', 'name']] = (countries.full
                                 .str.extract(r"""([A-Z]{2}) (.+)""", expand=True)
                                 )

mask = countries.name.isin(data_countries)
countries = countries[mask].reset_index(drop=True)
assert mask.sum() == len(data_countries)

countries['url'] = countries.abbrev.apply(lambda x: 'http://www.fallingrain.com/world/{}/'.format(x))
state_abbreviations = pd.read_csv(StringIO(requests.get('http://www.fonz.net/blog/wp-content/uploads/2008/04/states.csv').text))
countries['airports'] = countries.loc[countries.abbrev!='US','url'].apply(lambda x: x+'airports.html')
df_list = list()
for url in countries.airports:
    if url is not np.NaN:
        table = pd.read_html(url)[0]
        table.columns = table.iloc[0]
        table = table.iloc[1:]
        
        country_code = re.search(r"""world\/([A-Z]{2})\/""",url).group(1)
        country_name = countries.loc[countries.abbrev==country_code, 'name'].values[0]
        table['country'] = country_name
        df_list.append(table)
        
        
airports_df_1 = pd.concat(df_list, axis=0)
states = pd.DataFrame({'states':new_location_key.loc[new_location_key.country=='United States','province'].unique()})
states = pd.merge(states, state_abbreviations, left_on='states', right_on='State', how='left')

states.loc[states.states=='Puerto Rico', 'Abbreviation'] = 'PR'
states.loc[states.states=='American Samoa', 'Abbreviation'] = 'AS'
states.loc[states.states=='Virgin Islands', 'Abbreviation'] = 'VI'
states.drop(['State'], axis=1, inplace=True)
states['airports'] = states.Abbreviation.apply(lambda x: 'http://www.fallingrain.com/world/US/{}/airports.html'.format(x))

df_list = list()
for url in states.drop_duplicates(subset=['Abbreviation']).airports:
    if url is not np.NaN:
        table = pd.read_html(url)[0]
        table.columns = table.iloc[0]
        table = table.iloc[1:]
        
        state_code = re.search(r"""world\/US\/([A-Z]{2})\/""",url).group(1)
        state_name = states.loc[states.Abbreviation==state_code, 'states'].values[0]
        table['state'] = state_name
        table['country'] = 'United States'
        df_list.append(table)
        
airports_df_2 = pd.concat(df_list, axis=0)

airports = pd.concat([airports_df_1, airports_df_2], axis=0).reset_index(drop=True)

# Clean up column names
name_mapper = dict([(x,x.lower().replace(' ','_')) 
               for x in ['Kind','City','Name','Latitude','Longitude','Max Runway']])

airports = airports.rename(columns=name_mapper)

# Column formatting
airports['latitude'] = airports.latitude.str.replace(r"""\([NS]\)""", '').astype(float)
airports['longitude'] = airports.longitude.str.replace(r"""\([EW]\)""", '').astype(float)
airports['max_runway'] = airports.max_runway.str.replace(r""" ft""", '').astype(float)

# Extract just the medium and large airports
mask = airports.kind.isin(['Medium','Large'])
airports = airports[mask]

airports.head()

save_it(airports, '02_airport_information_fallingrain')
#%%
import pandas as pd
import numpy as np
import re
import time
from glob import glob

import matplotlib.pyplot as plt
data = pd.read_csv('cdc_zika.csv', encoding = 'utf-8')
data.drop(['time_period','time_period_type'], axis=1, inplace=True)

data['report_date'] = data.report_date.str.replace('_','-')       
data['report_date'] = pd.to_datetime(data.report_date)
data = data.loc[data.unit!='municipalities']
data = (data[['report_date', 'location', 'value', 'data_field']]
             .rename(columns={'report_date':'date','value':'zika_cases'}))
for x in data.zika_cases.iteritems():
    try:
        float(x[1])
    except:
        print(x)
data.loc[2414, 'zika_cases'] = 0
data.loc[2783, 'zika_cases'] = 0
data.loc[5192, 'zika_cases'] = 0
data['zika_cases'] = data.zika_cases.fillna(0)
data['zika_cases'] = data.zika_cases.astype(int)

data.query("zika_cases>0").shape, data.shape

excluded_fields = ['cumulative_cases_discarded',
'microcephaly_not',
'gbs_reported',
'zika_not',
'confirmed_acute_fever',
'confirmed_arthralgia',
'confirmed_arthritis', 
'confirmed_rash', 
'confirmed_conjunctivitis',
'confirmed_eyepain', 
'confirmed_headache', 
'confirmed_malaise',
'zika_reported_travel',
'yearly_reported_travel_cases']

mask = data.data_field.isin(excluded_fields)
print(mask.sum(), data.loc[mask, 'zika_cases'].sum(), data.zika_cases.sum())
data = data.loc[mask.pipe(np.invert)]

save_it(data, '03_infection_data_final')
#%%
import pandas as pd
import numpy as np
import re
import time
import dill
from datetime import timedelta
airport_info = pd.read_pickle('./pkl/02_airport_information_fallingrain.pkl')
airport_info.head(1)
lat_long_data = pd.read_pickle('./pkl/01_latitude_longitude_google.pkl')
lat_long_data.head(1)

airport_coords = airport_info[['latitude', 'longitude']].values[np.newaxis, :]
places_coords = np.rollaxis(lat_long_data[['latitude','longitude']].values[np.newaxis, :], 0, -1)

dist_coords = ((places_coords - airport_coords)**2).sum(axis=-1)
min_coords = dist_coords.argmin(axis=1)

merge_data = lat_long_data.copy()

print (merge_data.shape)

merge_data['airport_index'] = airport_info.index[min_coords]

# Now grap the airport and location info
df = airport_info.loc[merge_data.airport_index, ['country','name','FAA','IATA','ICAO']]
merge_data[['country','name','FAA','IATA','ICAO']] = df.set_index(merge_data.index)
pkl_it(merge_data, '04_merged_latitude_longitude_airport_checkpoint')

infection_data = pd.read_pickle('./pkl/03_infection_data_final.pkl')
infection_data = infection_data[['date','location']]
infection_data.head(1)

merge_all = pd.merge(infection_data, 
                     merge_data[['location','country','FAA','IATA','ICAO']], 
                     on='location', 
                     how='left').drop_duplicates()

print (merge_all.shape)

merge_all.head()

weather_scrape = (merge_all[['date','country','IATA','ICAO']]
                  .drop_duplicates()
                  .set_index(['country','IATA','ICAO'])
                  )

weather_scrape['date1'] = weather_scrape.date - timedelta(days=7)
weather_scrape['date2'] = weather_scrape.date - timedelta(days=14)

weather_scrape = (weather_scrape
                  .stack()
                  .reset_index(level=-1, drop=True)
                  .reset_index()
                  .rename(columns={0:'date'})
                  .dropna(subset=['IATA','ICAO'], how='all')
                 )

weather_scrape.shape

def scrape_weekly_weather(date, df_row):
    # Scrape the weekly data table
    url_fmt = 'https://www.wunderground.com/history/airport/{}/{}/{}/{}/WeeklyHistory.html?hdf=1'
    #Date=str(date.year)+'-'+str(date.month)+'-'+str(date.day)
    try:
        url = url_fmt.format(df_row.ICAO,date.year, date.month, date.day)
    except:
        url = url_fmt.format(df_row.IATA, date.year, date.month, date.day)
    
    try:
        table = pd.read_html(url)[0].dropna(subset=['Max','Avg','Min','Sum'], how='all')
        table.columns = ['Measurement','Max','Avg','Min','Sum']
        table.set_index('Measurement', inplace=True)
        table = table.stack()
        time.sleep(0.1)
    except:
        table = pd.Series({'NULL':np.NaN}, index=pd.Index([0]))
    
    return table
date_list = weather_scrape['date'].drop_duplicates()
airport_list = weather_scrape[['country','IATA','ICAO']].drop_duplicates()
os.mkdir("./pkl/04_scrape_weekly_weather_data")
for ndate, date in enumerate(date_list):
    
    print (ndate)
    df_list = list()
    
    for num,(row,dat) in enumerate(airport_list.iterrows()):
        
        try:
            df = scrape_weekly_weather(date, dat)
        except:
            df = pd.Series({'NULL':np.NaN}, index=pd.Index([row]))

        df_list.append((date, dat.name, df))
        
    with open('./pkl/04_scrape_weekly_weather_data/df_list{}.pkl'.format(ndate+69),'wb') as fh:
        dill.dump(df_list, fh)
def clean_weather_data(entry):
    index = pd.MultiIndex.from_tuples([(entry[0],
                                        entry[1])]*len(entry[2]),
                                      names=['date','index'])
    
    df = pd.DataFrame(entry[2].reset_index().values, 
                      index=index, 
                      columns=['measurement','type','value'])

    mask = (df.measurement.isin(['Max Temperature','Mean Temperature',
                                   'Min Temperature','Dew Point','Precipitation','Wind']))
    df = df.loc[mask]
    
    mask = ((((df.measurement=='Precipitation')&(df.type=='Sum'))|(df.type=='Avg')) & 
            ((df.measurement=='Precipitation')&(df.type=='Avg')).pipe(np.invert))
    df = df.loc[mask].drop(['type'], axis=1)
    
    df['value'] = (df.value
                   .str.replace('-', '')
                   .str.extract(r"""([0-9.-]+)""", expand=True)
                   .astype(float)
                   )
    
    return df

df_clean = list()


for i in range(131):
    with open('./pkl/04_scrape_weekly_weather_data/df_list{}.pkl'.format(i), 'rb') as fh:
        df_list = dill.load(fh)
    
    for df in enumerate(df_list):
        if not df[1][2].isnull().all():
            df_clean.append(clean_weather_data(df[1]))

weather_combined = pd.concat(df_clean, axis=0)
weather_combined.head()
weather_combined = pd.merge(weather_combined.reset_index(level=-1), 
                            airport_list, 
                            left_on='index', 
                            right_index=True).drop(['index'], axis=1).reset_index()

def time_shift(df, feature, week=1):
    new_df = (pd.merge(df[['date', feature]].reset_index(),
                       df[['date'+str(week), feature]].reset_index(),
                       left_on=df.index.names + ['date'], 
                       right_on=df.index.names + ['date'+str(week)],
                       suffixes=('',str(week)), 
                       how='inner')
              .drop(['date'+str(week)] + df.index.names, axis=1)
              .reset_index(level=-1, drop=True))
        
    return new_df

def create_weather_feature(df, feature):
    df_new = (df.loc[df.measurement==feature]
             .set_index(['IATA','ICAO','country','date','measurement'])
             .unstack())
    
    df_new = df_new.reset_index(level=-1)
    df_new.columns = ['date', feature]

    df_new['date1'] = df_new.date + timedelta(days=7)
    df_new['date2'] = df_new.date + timedelta(days=14)

    df_new1 = (df_new
            .groupby(level=[0,1])
            .apply(lambda x: time_shift(x,feature, 1))
            .reset_index(level=-1,drop=True))
    
    df_new2 = (df_new
            .groupby(level=[0,1])
            .apply(lambda x: time_shift(x, feature, 2))
            .reset_index(level=-1,drop=True))
    
    df_new = pd.merge(df_new1.reset_index(),
                      df_new2.reset_index().drop([feature], axis=1),
                      on=df_new1.index.names + ['date']).set_index(df_new1.index.names)
    
    return df_new

# Shift the one and two week prior data

max_temp = create_weather_feature(weather_combined, 'Max Temperature').set_index('date',append=True)
mean_temp = create_weather_feature(weather_combined, 'Mean Temperature').set_index('date',append=True)
min_temp = create_weather_feature(weather_combined, 'Min Temperature').set_index('date',append=True)
dew_point = create_weather_feature(weather_combined, 'Dew Point').set_index('date',append=True)
precipitation = create_weather_feature(weather_combined, 'Precipitation').set_index('date',append=True)
wind = create_weather_feature(weather_combined, 'Wind').set_index('date',append=True)

max_temp = max_temp.interpolate(method='linear', limit_direction='both')
mean_temp = mean_temp.interpolate(method='linear', limit_direction='both')
min_temp = min_temp.interpolate(method='linear', limit_direction='both')
dew_point = dew_point.interpolate(method='linear', limit_direction='both')
precipitation = precipitation.interpolate(method='linear', limit_direction='both')
wind = wind.interpolate(method='linear', limit_direction='both')

airport = pd.read_pickle('./pkl/04_merged_latitude_longitude_airport_checkpoint.pkl')
airport.head(1)

max_temp = pd.merge(max_temp.reset_index(),
         airport[['ICAO','IATA','location']],
         on=['ICAO','IATA'],
         how='left')#.drop_duplicates(subset=['location'])

mean_temp = pd.merge(mean_temp.reset_index(),
         airport[['ICAO','IATA','location']],
         on=['ICAO','IATA'],
         how='left')


min_temp = pd.merge(min_temp.reset_index(),
         airport[['ICAO','IATA','location']],
         on=['ICAO','IATA'],
         how='left')


dew_point = pd.merge(dew_point.reset_index(),
         airport[['ICAO','IATA','location']],
         on=['ICAO','IATA'],
         how='left')

precipitation = pd.merge(precipitation.reset_index(),
         airport[['ICAO','IATA','location']],
         on=['ICAO','IATA'],
         how='left')

wind = pd.merge(wind.reset_index(),
         airport[['ICAO','IATA','location']],
         on=['ICAO','IATA'],
         how='left')
max_temp = max_temp.drop(['ICAO','IATA'], axis=1).drop_duplicates(subset=['location','date'])
max_temp.columns = [x.lower().replace(' ', '_').replace('erature','') for x in max_temp.columns]

mean_temp = mean_temp.drop(['ICAO','IATA'], axis=1).drop_duplicates(subset=['location','date'])
mean_temp.columns = [x.lower().replace(' ', '_').replace('erature','') for x in mean_temp.columns]

min_temp = min_temp.drop(['ICAO','IATA'], axis=1).drop_duplicates(subset=['location','date'])
min_temp.columns = [x.lower().replace(' ', '_').replace('erature','') for x in min_temp.columns]

dew_point = dew_point.drop(['ICAO','IATA'], axis=1).drop_duplicates(subset=['location','date'])
dew_point.columns = [x.lower().replace(' ', '_').replace('erature','') for x in dew_point.columns]

precipitation = precipitation.drop(['ICAO','IATA'], axis=1).drop_duplicates(subset=['location','date'])
precipitation.columns = [x.lower().replace(' ', '_').replace('erature','') for x in precipitation.columns]

wind = wind.drop(['ICAO','IATA'], axis=1).drop_duplicates(subset=['location','date'])
wind.columns = [x.lower().replace(' ', '_').replace('erature','') for x in wind.columns]

weather_final = pd.merge(max_temp, mean_temp, on=['date','location'], how='inner')
weather_final = pd.merge(weather_final, min_temp, on=['date','location'], how='inner')
weather_final = pd.merge(weather_final, dew_point, on=['date','location'], how='inner')
weather_final = pd.merge(weather_final, precipitation, on=['date','location'], how='inner')
weather_final = pd.merge(weather_final, wind, on=['date','location'], how='inner')

save_it(weather_final, '04_weekly_weather')
#%%
import pandas as pd
import numpy as np
import re

mosquito = pd.read_csv('./csv/aegypti_albopictus.csv')

mosquito.columns = [x.lower() for x in mosquito.columns]

# according to the code book, x is longitude and y is latitude
mosquito.rename(columns={'x':'longitude', 'y':'latitude'}, inplace=True)
mosquito_clean = pd.concat([mosquito.loc[mosquito.year=='2006-2008'].assign(year=x) 
                      for x in ['2006', '2007', '2008']])
mosquito_clean = pd.concat([mosquito.loc[mosquito.year!='2006-2008'], mosquito_clean])
mosquito_clean = mosquito_clean.loc[mosquito_clean.year.notnull()]
mosquito_clean['year'] = mosquito_clean.year.astype(int)
mosquito_clean.sort_values('year',inplace=True)
mosquito_clean = mosquito_clean.loc[mosquito_clean.year>=2006]

airport = pd.read_pickle('./pkl/02_airport_information_fallingrain.pkl')
airport_list = airport.country.unique()
mosquito_list = mosquito_clean.country.unique()

np.in1d(airport_list, mosquito_list)
mask = mosquito_clean.country=='United States of America'
mosquito_clean.loc[mask,'country'] = 'United States'
save_it(mosquito_clean, '05_mosquito_sightings')
#%%
import pandas as pd
import numpy as np

#from csv_pkl_sql import csv_it, sql_it
#import cPickle as pickle
#import os
from osgeo import gdal
#import gdal
# Import the latitude and longitude data
lat_long_data = pd.read_csv('./csv/01_latitude_longitude_google.csv')


# Import the geoTIF map
ds = gdal.Open('./gpw_population_density/gpw-v4-population-density_2020.tif')

rows = ds.RasterYSize
cols = ds.RasterXSize

transform = ds.GetGeoTransform()
xOrigin = transform[0]
yOrigin = transform[3]
pixelWidth = transform[1]
pixelHeight = transform[5]

# My data has one band, otherwise would have to iterate through bands
band = ds.GetRasterBand(1)


def get_population_density(latitude, longitude, 
                           xOrigin=xOrigin, yOrigin=yOrigin,
                           pixelWidth=pixelWidth, pixelHeight=pixelHeight,
                           band=band):

    # # Single point, x=longitude, y=latitude
    # x = -155.662499999999824
    # y = 19.0041666666754416
    x = longitude
    y = latitude

    # This reads three pixels in x- and y- direction
    try:
        # Subtract one off the end because I want to read 3 x 3 region
        size = 100

        dist_matrix = np.meshgrid(np.arange(-size, size+1), 
                                  np.arange(-size, size+1))
        dist_matrix = np.sqrt((dist_matrix[0]**2 + dist_matrix[1]**2))
        sort_order = dist_matrix.ravel().argsort()

        xOffset = int((x - xOrigin) / pixelWidth) - size
        yOffset = int((y - yOrigin) / pixelHeight) - size

        data = band.ReadAsArray(xOffset, yOffset, 2*size+1, 2*size+1)
        data_sort = data.ravel()[sort_order]

        density = data_sort[data_sort>0][:9].mean()
    except:
        density = np.NaN

    return density


lat_long_data['density_per_km'] = lat_long_data.apply(lambda x: get_population_density(x.latitude, x.longitude), axis=1)

#lat_long_data[['location','density_per_km']].to_csv('../csv/06_population_density.csv')
csv_it(lat_long_data[['location','density_per_km']], '06_population_density')

with open('./pkl/06_population_density.pkl', 'wb') as fh:
    pickle.dump(lat_long_data[['location','density_per_km']], fh)

#sql_it(lat_long_data[['location','density_per_km']], '06_population_density')
#%%
import pandas as pd
import numpy as np
import dill
from datetime import timedelta
#from csv_pkl_sql import save_it, csv_it, pkl_it

import matplotlib.pyplot as plt
import seaborn as sns

with open('./pkl/00_cleaned_city_names.pkl', 'rb') as fh:
    location_key = dill.load(fh)
location_key.head(1)

with open('./pkl/01_latitude_longitude_google.pkl', 'rb') as fh:
    lat_long = dill.load(fh)
lat_long.head(1)
lat_long = lat_long[['location','latitude','longitude']]
location_key.shape, lat_long.shape
location = pd.merge(location_key, lat_long, on='location', how='inner')
location.head(1)
with open('./pkl/02_airport_information_fallingrain.pkl', 'rb') as fh:
    airport = dill.load(fh)
airport.head(1)
airport = airport[['city', 'FAA', 'IATA', 'ICAO', 'kind', 'latitude',
       'longitude', 'max_runway', 'name', 'country', 'state']]
with open('./pkl/04_merged_latitude_longitude_airport_checkpoint.pkl', 'rb') as fh:
    airport2 = dill.load(fh)
airport2.head(1)
airport2 = airport2[['location', 'latitude', 'longitude', 'airport_index', 'country', 'name', 'FAA', 'IATA',
       'ICAO']]
airport.shape, airport2.shape
airport.kind.unique()
# Closest medium or large airport
airport_coords = airport[['latitude', 'longitude']].values[np.newaxis, :]
places_coords = np.rollaxis(lat_long[['latitude','longitude']].values[np.newaxis, :], 0, -1)
dist_coords = ((places_coords - airport_coords)**2).sum(axis=-1)
min_dist = dist_coords.min(axis=1)

airport_distance = lat_long[['location']].copy()
airport_distance['airport_dist_any'] = min_dist

# Closest large airport
airport_coords = airport.loc[airport.kind=='Large', 
                             ['latitude', 'longitude']].values[np.newaxis, :]
places_coords = np.rollaxis(lat_long[['latitude','longitude']].values[np.newaxis, :], 0, -1)
dist_coords = ((places_coords - airport_coords)**2).sum(axis=-1)
min_dist = dist_coords.min(axis=1)

airport_distance['airport_dist_large'] = min_dist
with open('./pkl/04_weekly_weather.pkl', 'rb') as fh:
    weather = dill.load(fh)

weather.head(2)

with open('./pkl/05_mosquito_sightings.pkl', 'rb') as fh:
    mosquito = dill.load(fh)
mosquito.head(1)

# Closest mosquito sighting
mosquito_coords = mosquito[['latitude', 'longitude']].values[np.newaxis, :]
places_coords = np.rollaxis(lat_long[['latitude','longitude']].values[np.newaxis, :], 0, -1)
dist_coords = ((places_coords - mosquito_coords)**2).sum(axis=-1)
min_dist = dist_coords.min(axis=1)

mosquito_distance = lat_long[['location']].copy()
mosquito_distance['mosquito_dist'] = min_dist

mosquito_distance.head()

with open('./pkl/06_population_density.pkl', 'rb') as fh:
    population = dill.load(fh)
population.head(1)

with open('./pkl/03_infection_data_final.pkl', 'rb') as fh:
    infection = dill.load(fh)
infection.head(1)
infection = (infection
             .groupby(['location','date']).sum()
             .reset_index()
            )

infection.sort_values('zika_cases',ascending=False).head(20)

model = pd.merge(weather,
                 infection,
                 on=['date','location'],
                 how='left')

model['zika_cases'] = model.zika_cases.fillna(0)

print (model.shape, model.isnull().sum().max())

model = pd.merge(model,
                 population,
                 on='location', 
                 how='left')

print (model.shape, model.isnull().sum().max())

model = pd.merge(model,
                 airport_distance,
                 on='location',
                 how='left')

print (model.shape, model.isnull().sum().max())

model = pd.merge(model,
                 mosquito_distance,
                 on='location',
                 how='left')

print (model.dropna().shape, model.isnull().sum().max())

save_it(model, '07_feature_engineering_and_cleaning')
#%%
