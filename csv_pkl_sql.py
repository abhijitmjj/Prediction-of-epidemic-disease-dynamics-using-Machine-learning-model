
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

def sql_it(dataframe, filebase, dbname='zika'):
    postgres_str = 'postgresql://abhijit:{}@127.0.0.1:5491/{}'.format(POSTGRES_PWD, dbname)
    engine = create_engine(postgres_str)
    dataframe.to_sql(filebase, engine, if_exists='replace', dtype={col_name: NVARCHAR for col_name in filebase})
    return

def save_it(dataframe, filebase, dbname='zika'):
    csv_it(dataframe, filebase)
    pkl_it(dataframe, filebase)
    #sql_it(dataframe, 'sql_' + filebase, dbname=dbname)
    return
