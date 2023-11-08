import pandas as pd
import torch
from torch_geometric.data import Data
import torch_geometric.utils
from datetime import datetime
from datetime import timedelta
from collections import OrderedDict
import networkx as nx
import matplotlib.pyplot as plt
import time
import os

filename = input("transaction file (exclude extension)")
transaction = pd.read_csv(filename+".csv")
print("file loaded")
start = time.time()

raw_data = OrderedDict()
raw_money_sent = OrderedDict()
raw_node_recieved = OrderedDict()
raw_money_recieved = OrderedDict();

date_list = []
dictionary = {}

node_to_addr = {}

count = 0
edge_dict = {}

def write_to_row(previous_timestamp):
	print(previous_timestamp)
	
	x = list(dictionary.values())
	edges = [[dictionary[pair[0]], dictionary[pair[1]]] for pair in edge_dict]

	edge_index = torch.tensor(edges, dtype=torch.long)
	data = Data(num_nodes=len(x), x=x, edge_index=edge_index.t().contiguous())
	g = torch_geometric.utils.to_networkx(data, to_undirected=False)
	g.remove_edges_from(nx.selfloop_edges(g))
	node_edge = {(dictionary[pair[0]], dictionary[pair[1]]): edge_dict[pair] for pair in edge_dict}

	nx.set_edge_attributes(g, node_edge)

	total_map = {}
	sent_map = {}
	#print(node_to_addr)
	for node in g.nodes():
		num_adj_nodes = 0
		total_sent = 0
		for neighbour in g.neighbors(node):
			num_adj_nodes += 1
			#print(node_to_addr[node], node_to_addr[neighbour])
			total_sent += g[node][neighbour]["weight"]

			if node_to_addr[neighbour] not in sent_map:
				sent_map[node_to_addr[neighbour]] = set()
			sent_map[node_to_addr[neighbour]].add(node)

			if node_to_addr[neighbour] not in total_map:
				total_map[node_to_addr[neighbour]] = 0
			total_map[node_to_addr[neighbour]] += g[node][neighbour]["weight"]

		if node_to_addr[node] not in raw_data:
			raw_data[node_to_addr[node]] = {}
		raw_data[node_to_addr[node]]["Address"] = node_to_addr[node]
		raw_data[node_to_addr[node]][previous_timestamp.strftime("%Y/%m/%d")] = num_adj_nodes

		if node_to_addr[node] not in raw_money_sent:
			raw_money_sent[node_to_addr[node]] = {}
		raw_money_sent[node_to_addr[node]]["Address"] = node_to_addr[node]
		raw_money_sent[node_to_addr[node]][previous_timestamp.strftime("%Y/%m/%d")] = total_sent

	for node in g.nodes():
		if node_to_addr[node] not in raw_money_recieved:
			raw_money_recieved[node_to_addr[node]] = {}
		if node_to_addr[node] not in total_map:
			total_map[node_to_addr[node]] = 0
		raw_money_recieved[node_to_addr[node]]["Address"] = node_to_addr[node]
		raw_money_recieved[node_to_addr[node]][previous_timestamp.strftime("%Y/%m/%d")] = total_map[node_to_addr[node]]

		if node_to_addr[node] not in raw_node_recieved:
			raw_node_recieved[node_to_addr[node]] = {}
		if node_to_addr[node] not in sent_map:
			sent_map[node_to_addr[node]] = set()
		raw_node_recieved[node_to_addr[node]]["Address"] = node_to_addr[node]
		raw_node_recieved[node_to_addr[node]][previous_timestamp.strftime("%Y/%m/%d")] = len(sent_map[node_to_addr[node]])

def write_csv(filename, mp, folder):
	try:
		os.mkdir(folder)
	except:
		#folder already exists
		pass
	ls = [mp[row] for row in mp]
	df = pd.DataFrame(ls)
	df.to_csv(f"{folder}/{filename}")

def read_data_frame(df, year_range, c):
	count = c
	previous_timestamp = df.iloc[0]["block_timestamp"].to_pydatetime()
	for index, row in df.iterrows():
		timestamp = row.get("block_timestamp").to_pydatetime()
		from_addr = row.get("from_address")
		to_addr = row.get("to_address")

		#print("now", timestamp)
		#print("before", previous_timestamp)
		if (timestamp.date() > previous_timestamp.date()):
			write_to_row(previous_timestamp)

		if from_addr not in dictionary:
			dictionary[from_addr] = count
			node_to_addr[count] = from_addr
			count += 1
		if to_addr not in dictionary:
			dictionary[to_addr] = count
			node_to_addr[count] = to_addr
			count += 1

		pair = (from_addr, to_addr)
		attr = {}
		if pair not in edge_dict:
			attr["weight"] = int(row.get("value"))
			edge_dict[pair] = attr
		else:
			edge_dict[pair]["weight"] += int(row.get("value"))
		previous_timestamp = timestamp

	write_to_row(previous_timestamp)

	write_csv(f"{filename}_{year_range}_num_nodes_sent.csv", raw_data, "nodes_sent_attribute_3")
	write_csv(f"{filename}_{year_range}_num_nodes_recieved.csv", raw_node_recieved, "nodes_recieved_attribute_4")
	write_csv(f"{filename}_{year_range}_money_sent.csv", raw_money_sent, "total_money_sent_attribute_5")
	write_csv(f"{filename}_{year_range}_money_received.csv", raw_money_recieved, "total_money_recieved_attribute_6")
	return count

def partition_by_year(df, c):
	count = c
	start = datetime.strptime(df.iloc[0]["block_timestamp"][:10], "%Y-%m-%d")
	years = set([int(df.iloc[x]["block_timestamp"][:4]) for x, r in df.iterrows()])
	print(filename, "contains:", years)

	df['block_timestamp'] = pd.to_datetime(df['block_timestamp'])
	df = df.sort_values('block_timestamp', ignore_index=True)

	for yr in years:
		print("Starting:", yr)
		data = df.loc[pd.to_datetime(df['block_timestamp']).dt.year == yr]
		#print(type(data))

		global raw_data
		global raw_money_sent
		global raw_node_recieved
		global raw_money_recieved

		raw_data = OrderedDict()
		raw_money_sent = OrderedDict()
		raw_node_recieved = OrderedDict()
		raw_money_recieved = OrderedDict();
		count = read_data_frame(data, f"{int(yr)}_{int(yr)+1}", count)
		print("Finished:", yr)

partition_by_year(transaction, count)
end = time.time()
elapsed = end - start
print(str(timedelta(seconds=elapsed)))
