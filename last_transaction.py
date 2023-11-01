import pandas as pd
from datetime import datetime
from collections import OrderedDict

filename = input()
transaction = pd.read_csv(filename+".csv")

node_last_transaction = {}
dates = []

for index, row in transaction.iterrows():
	timestamp = datetime.strptime(row.get("block_timestamp")[:10], "%Y-%m-%d")
	from_addr = row.get("from_address")
	to_addr = row.get("to_address")

	node_last_transaction[from_addr] = timestamp
	node_last_transaction[to_addr] = timestamp

	if timestamp not in dates:
		dates.append(timestamp)

raw_data = OrderedDict()

for date in dates:
	print(date)
	for node in node_last_transaction:
		if node not in raw_data:
			raw_data[node] = {}
		raw_data[node]["Address"] = node
		delta = date - node_last_transaction[node]
		raw_data[node][date.strftime("%Y/%m/%d")] = delta.days
lis = []
for row in raw_data:
	lis.append(raw_data[row])

#print(lis)

days_since = pd.DataFrame(lis)
days_since.to_csv(filename + "_days_since_most_recent_transaction.csv")