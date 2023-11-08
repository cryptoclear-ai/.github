import pandas as pd
from datetime import datetime
from collections import OrderedDict
import time
import os
from datetime import timedelta


filename = input("Filename (exclude extension)")
transaction = pd.read_csv(filename+".csv")
print("File loaded")

start = time.time()

node_last_transaction = {}
last_transaction_csv = OrderedDict()
first_seen = {}
inception_csv = OrderedDict()

def write_csv(filename, mp, folder=False):
	if folder is False:
		ls = [mp[row] for row in mp]
		df = pd.DataFrame(ls)
		df.to_csv(filename)
	else:
		try:
			os.mkdir(folder)
		except:
			#folder already exists
			pass
		ls = [mp[row] for row in mp]
		df = pd.DataFrame(ls)
		df.to_csv(f"{folder}/{filename}")

def f_write(previous_timestamp):
	print(previous_timestamp)
	for node in node_last_transaction:
		if node not in last_transaction_csv:
			last_transaction_csv[node] = {}
		last_transaction_csv[node]["Address"] = node

		if node not in inception_csv:
			inception_csv[node] = {}
		inception_csv[node]["Address"] = node

		delta = previous_timestamp - node_last_transaction[node]
		last_transaction_csv[node][previous_timestamp.strftime("%Y/%m/%d")] = delta.days

		delta = previous_timestamp - first_seen[node]
		inception_csv[node][previous_timestamp.strftime("%Y/%m/%d")] = delta.days


def read_data_frame(df, year_range):
	previous_timestamp = df.iloc[0]["block_timestamp"].to_pydatetime()
	for index, row in df.iterrows():
		timestamp = row.get("block_timestamp").to_pydatetime()
		from_addr = row.get("from_address")
		to_addr = row.get("to_address")

		if (timestamp.date() > previous_timestamp.date()):
			f_write(previous_timestamp)

		node_last_transaction[from_addr] = timestamp
		node_last_transaction[to_addr] = timestamp

		if from_addr not in first_seen:
			first_seen[from_addr] = timestamp
		if to_addr not in first_seen:
			first_seen[to_addr] = timestamp

		previous_timestamp = timestamp
	f_write(previous_timestamp)

	write_csv(f"{filename}_{year_range}_days_since_inception.csv", inception_csv, "inception_attribute_1")
	write_csv(f"{filename}_{year_range}_days_since_most_recent_transaction.csv", last_transaction_csv, "last_transaction_attribute_2")

#split files into years
def partition_by_year(df):
	start = datetime.strptime(df.iloc[0]["block_timestamp"][:10], "%Y-%m-%d")
	years = set([int(df.iloc[x]["block_timestamp"][:4]) for x, r in df.iterrows()])
	print(filename, "contains:", years)

	df['block_timestamp'] = pd.to_datetime(df['block_timestamp'])
	df = df.sort_values('block_timestamp', ignore_index=True)
	for yr in years:
		print("Starting:", yr)
		data = df.loc[pd.to_datetime(df['block_timestamp']).dt.year == yr]

		global last_transaction_csv 
		global inception_csv 
		last_transaction_csv = OrderedDict()
		inception_csv = OrderedDict()
		read_data_frame(data, f"{int(yr)}_{int(yr)+1}")
		print("Finished:", yr)
	

partition_by_year(transaction)
end = time.time()
elapsed = end - start
print(str(timedelta(seconds=elapsed)))