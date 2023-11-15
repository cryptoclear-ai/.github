import pandas as pd
import numpy as np
from datetime import datetime
from collections import OrderedDict
import time
import os
from datetime import timedelta
import gc
from torch_geometric.data import Data
import torch_geometric.utils
import networkx as nx

filename = input("Filename (exclude extension)")
transaction = pd.read_csv(filename+".csv", usecols=["block_timestamp", "from_address", "to_address", "value"])
print("File loaded")

which_attr = int(input("what attribute"))

start = time.time()

#node_info = {}
#info_csv = OrderedDict()
info_csv = OrderedDict()

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
		df.to_csv(f"{folder}/{filename}", chunksize=50)
		del df
		gc.collect()

def f_write(previous_timestamp, info, nodes):
	print(previous_timestamp)

	info["g"].remove_edges_from(nx.selfloop_edges(info["g"]))

	global which_attr
	adjacency_matrix = nx.to_numpy_array(info['g'])

	if which_attr == 3:
		for node, row in enumerate(adjacency_matrix):
			if info['node_to_addr'][node] not in nodes:
				continue

			if info['node_to_addr'][node] not in info_csv:
				info_csv[info['node_to_addr'][node]] = {}
			info_csv[info['node_to_addr'][node]]["Address"] = info['node_to_addr'][node]
			info_csv[info['node_to_addr'][node]][previous_timestamp.strftime("%Y/%m/%d")] = np.count_nonzero(row)
	if which_attr == 4:
		for node, row in enumerate(adjacency_matrix.T):
			if info['node_to_addr'][node] not in nodes:
				continue

			if info['node_to_addr'][node] not in info_csv:
				info_csv[info['node_to_addr'][node]] = {}
			info_csv[info['node_to_addr'][node]]["Address"] = info['node_to_addr'][node]
			info_csv[info['node_to_addr'][node]][previous_timestamp.strftime("%Y/%m/%d")] = np.count_nonzero(row)
	if which_attr == 5:
		for node, row in enumerate(adjacency_matrix):
			if info['node_to_addr'][node] not in nodes:
				continue

			if info['node_to_addr'][node] not in info_csv:
				info_csv[info['node_to_addr'][node]] = {}
			info_csv[info['node_to_addr'][node]]["Address"] = info['node_to_addr'][node]
			info_csv[info['node_to_addr'][node]][previous_timestamp.strftime("%Y/%m/%d")] = np.sum(row)
	if which_attr == 6:
		for node, row in enumerate(adjacency_matrix.T):
			if info['node_to_addr'][node] not in nodes:
				continue

			if info['node_to_addr'][node] not in info_csv:
				info_csv[info['node_to_addr'][node]] = {}
			info_csv[info['node_to_addr'][node]]["Address"] = info['node_to_addr'][node]
			info_csv[info['node_to_addr'][node]][previous_timestamp.strftime("%Y/%m/%d")] = np.sum(row)

def read_data_frame(df, year_range, nodes, part, info):
	previous_timestamp = df.iloc[0]["block_timestamp"].to_pydatetime()
	for index, row in df.iterrows():
		timestamp = row.get("block_timestamp").to_pydatetime()
		from_addr = row.get("from_address")
		to_addr = row.get("to_address")

		if (timestamp.date() > previous_timestamp.date()):
			f_write(previous_timestamp, info, nodes)

		#node_info[from_addr] = timestamp
		#node_info[to_addr] = timestamp

		if (from_addr in nodes) or (to_addr in nodes):
			if from_addr not in info["dictionary"]:
				info["dictionary"][from_addr] = info["count"]
				info["node_to_addr"][info["count"]] = from_addr
				info["count"] += 1
			if to_addr not in info["dictionary"]:
				info["dictionary"][to_addr] = info["count"]
				info["node_to_addr"][info["count"]] = to_addr
				info["count"] += 1

			if not (info["g"].has_edge(info['dictionary'][from_addr], info['dictionary'][to_addr])):
				info["g"].add_edge(info["dictionary"][from_addr], info["dictionary"][to_addr], weight=row.get("value"))
			else:
				info["g"].edges[info["dictionary"][from_addr], info["dictionary"][to_addr]]['weight'] = (row.get("value") + info["g"].edges[info["dictionary"][from_addr], info["dictionary"][to_addr]]['weight'])

		previous_timestamp = timestamp
	f_write(previous_timestamp, info, nodes)

	string = ''
	global which_attr
	if which_attr == 3:
		write_csv(f"{filename}_{year_range}_num_nodes_sent_chunk_{part}.csv", info_csv, "nodes_sent_attribute_3")
	if which_attr == 4:
		write_csv(f"{filename}_{year_range}_num_nodes_recieved_chunk_{part}.csv", info_csv, "nodes_recieved_attribute_4")
	if which_attr == 5:
		write_csv(f"{filename}_{year_range}_money_sent_chunk_{part}.csv", info_csv, "total_money_sent_attribute_5")
	if which_attr == 6:
		write_csv(f"{filename}_{year_range}_money_received_chunk_{part}.csv", info_csv, "total_money_recieved_attribute_6")


	write_csv(f"{filename}_{year_range}_days_since_inception_chunk_{part}.csv", info_csv, "inception_attribute_1")
	#write_csv(f"{filename}_{year_range}_days_since_most_recent_transaction.csv", info_csv, "info_attribute_2")

def write_part(nodes, data, yr, part, info):
	read_data_frame(data, f"{int(yr)}_{int(yr)+1}", nodes, part, info)

#split files into years
def partition_by_year(df):
	start = datetime.strptime(df.iloc[0]["block_timestamp"][:10], "%Y-%m-%d")
	years = set([int(df.iloc[x]["block_timestamp"][:4]) for x, r in df.iterrows()])
	print(filename, "contains:", years)

	PARTS = 100
	nodes = set(pd.concat([df['from_address'], df['to_address']]))
	size = round(len(nodes)/PARTS)
	print("seprated nodes into: ", PARTS, " groups, each size: ", size)
	parts = [list(nodes)[i * size:(i + 1) * size] for i in range((len(nodes) + size - 1) // size )]

	df['block_timestamp'] = pd.to_datetime(df['block_timestamp'])
	df['value'] = df['value'].astype(float)
	df = df.iloc[np.where(df['value']>0)]
	df = df.iloc[np.where(df['from_address']!=df['to_address'])]
	df = df.sort_values('block_timestamp', ignore_index=True)
	data = [df.loc[pd.to_datetime(df['block_timestamp']).dt.year == yr] for yr in years]
	global transaction

	del transaction
	del nodes
	gc.collect()

	part = 1
	for y in parts:
		print("working on chunk #", part)
		info = {"dictionary": {}, "node_to_addr": {}, "count": 0, "g": nx.DiGraph()}
		for x, yr in zip(data, years):
			print("Starting:", yr)
			#global info_csv 
			global info_csv 
			#info_csv = OrderedDict()
			info_csv = OrderedDict()
			#read_data_frame(x, f"{int(yr)}_{int(yr)+1}")
			write_part(set(y), x, yr, part, info)
			print("Finished:", yr)
		part += 1
	

partition_by_year(transaction)
end = time.time()
elapsed = end - start
print(str(timedelta(seconds=elapsed)))