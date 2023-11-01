import pandas as pd
from datetime import datetime
from collections import OrderedDict

filename = input()
transaction = pd.read_csv(filename+".csv")

node_date_made = {}
dates = []

for index, row in transaction.iterrows():
	timestamp = datetime.strptime(row.get("block_timestamp")[:10], "%Y-%m-%d")
	from_addr = row.get("from_address")
	to_addr = row.get("to_address")

	if from_addr not in node_date_made:
		node_date_made[from_addr] = timestamp
	if to_addr not in node_date_made:
		node_date_made[to_addr] = timestamp

	if timestamp not in dates:
		dates.append(timestamp)

raw_data = OrderedDict()

for date in dates:
	print(date)
	for node in node_date_made:
		if node not in raw_data:
			raw_data[node] = {}
		raw_data[node]["Address"] = node
		delta = date - node_date_made[node]
		raw_data[node][date.strftime("%Y/%m/%d")] = delta.days
lis = []
for row in raw_data:
	lis.append(raw_data[row])

#print(lis)

days_since = pd.DataFrame(lis)
days_since.to_csv(filename + "_days_since_appearing.csv")