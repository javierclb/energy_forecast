import pickle
import numpy as np
import pandas as pd
from datetime import datetime as dt
from scipy.stats import zscore

# NERC6 holidays with inconsistent dates. Created with python holidays package
# years 1990 - 2024
with open('serialized/holidays.pickle', 'rb') as f:
	nerc6 = pickle.load(f)

def isHoliday(holiday, df):
	# New years, memorial, independence, labor day, Thanksgiving, Christmas
	m1 = None
	if holiday == "New Year's Day":
		m1 = (df["dates"].dt.month == 1) & (df["dates"].dt.day == 1)
	if holiday == "Independence Day":
		m1 = (df["dates"].dt.month == 7) & (df["dates"].dt.day == 4)
	if holiday == "Christmas Day":
		m1 = (df["dates"].dt.month == 12) & (df["dates"].dt.day == 25)

	if holiday == "Fiestas Patrias":
		m1 = (df["dates"].dt.month == 9) & (df["dates"].dt.day == 18)
	
	if holiday == "Glorias Ejercito":
		m1 = (df["dates"].dt.month == 9) & (df["dates"].dt.day == 19)

	if holiday == "Glorias navales":
		m1 = (df["dates"].dt.month == 5) & (df["dates"].dt.day == 21)	

	m1 = df["dates"].dt.date.isin(nerc6[holiday]) if m1 is None else m1
	m2 = df["dates"].dt.date.isin(nerc6.get(holiday + " (Observed)", []))
	return m1 | m2

def prepare_data(df, noise=2.5, hours_prior=8760):
		
	if 'dates' not in df.columns:
		df['dates'] = df.apply(
			lambda x: dt(
				int(x['year']), 
				int(x['month']), 
				int(x['day']), 
				int(x['hour'])), 
			axis=1
		)
    
	out_df = pd.DataFrame()
	out_df["load_n"] = zscore(df["load"])

	# create day of week vector
	out_df["day"] = df["dates"].dt.dayofweek  # 0 is Monday.
	w = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
	for i, d in enumerate(w):
		out_df[d] = (out_df["day"] == i).astype(int)
		
	# create hour of day vector
	out_df["hour"] = df["dates"].dt.hour
	d = [("h" + str(i)) for i in range(24)]
	for i, h in enumerate(d):
		out_df[h] = (out_df["hour"] == i).astype(int)
	
	# create month vector
	out_df["month"] = df["dates"].dt.month
	y = [("m" + str(i)) for i in range(1, 13)]
	for i, processed in enumerate(y):
		out_df[processed] = (out_df["month"] == i+1).astype(int)
	
	for i in range (hours_prior):
		out_df["load"+str(i)] = out_df["load_n"].shift(i+1)
		out_df["load"+str(i)].bfill(inplace=True)
	
		# create holiday booleans
	out_df["isNewYears"] = isHoliday("New Year's Day", df).astype(int)
	out_df["isLaborDay"] = isHoliday("Labor Day", df).astype(int)
	out_df["isChristmas"] = isHoliday("Christmas Day", df).astype(int)
	out_df["is18"] = isHoliday("Fiestas Patrias", df).astype(int)
	out_df["is19"] = isHoliday("Glorias Ejercito", df).astype(int)
	out_df["is21mayo"] = isHoliday("Glorias navales", df).astype(int)

	processed = out_df.drop(["month", "hour","day", "load_n"], axis=1)
	df = df.drop(['dates'], axis=1)

	return processed[hours_prior:]
