import pandas as pd
import torch
from torch_geometric.data import Data
import torch_geometric.utils
from datetime import datetime
from collections import OrderedDict
import networkx as nx
import matplotlib.pyplot as plt

filename = input()
transaction = pd.read_csv(filename+".csv")

start = datetime.strptime(transaction.iloc[0]["block_timestamp"][:10], "%Y-%m-%d")
end = datetime.strptime(transaction.iloc[-1]["block_timestamp"][:10], "%Y-%m-%d")

date_range = pd.date_range(start, end)
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
	x = []
	edges = []
	for key in dictionary:
		x.append([dictionary[key]])
	for pair in edge_dict:
		edges.append([dictionary[pair[0]] , dictionary[pair[1]]])
	edge_index = torch.tensor(edges, dtype=torch.long)
	data = Data(num_nodes=len(x), x=x, edge_index=edge_index.t().contiguous())
	g = torch_geometric.utils.to_networkx(data, to_undirected=False)
	node_edge = {}
	for pair in edge_dict:
		new_pair = (dictionary[pair[0]], dictionary[pair[1]])
		node_edge[new_pair] = edge_dict[pair]

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

previous_timestamp = datetime.strptime(transaction.iloc[0]["block_timestamp"][:10], "%Y-%m-%d")
for index, row in transaction.iterrows():
	timestamp = datetime.strptime(row.get("block_timestamp")[:10], "%Y-%m-%d")
	from_addr = row.get("from_address")
	to_addr = row.get("to_address")

	#print("now", timestamp)
	#print("before", previous_timestamp)
	if (timestamp > previous_timestamp):
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

def write_csv(filename, mp):
	ls = []
	for row in mp:
		ls.append(mp[row])
	df = pd.DataFrame(ls)
	df.to_csv(filename)

write_csv(filename+"_num_nodes_sent.csv", raw_data)
write_csv(filename+"_num_nodes_recieved.csv", raw_node_recieved)
write_csv(filename+"_money_sent.csv", raw_money_sent)
write_csv(filename+"_money_received.csv", raw_money_recieved)


'''
def get_df_uptill(dt, start_index):
	sub_df = transaction[start_index:]
	for index, row in sub_df.iterrows():
		date = datetime.strptime(row.get("block_timestamp")[:10], "%Y-%m-%d")
		delta = dt - date
		if delta.days < 0:
			cutoff = int(index)
			#print(cutoff)
			return sub_df[:cutoff], cutoff+start_index
	return sub_df, len(transaction)
'''
#print(ftt)
#print(get_df_uptill(end))
'''
dictionary = {}

node_to_addr = {}

count = 0
edge_dict ={}

def getGraph(df,c):
	count = c
	for index, row in df.iterrows():
		from_addr = row.get("from_address")
		if from_addr in dictionary:
			pass
		else:
			dictionary[from_addr] = count
			count += 1
		to_addr = row.get("to_address")
		if to_addr in dictionary:
			pass
		else:
			dictionary[to_addr] = count
			count += 1
		pair = (from_addr, to_addr)
		if pair not in edge_dict:
			edge_dict[pair] = int(row.get("value"))
		else:
			edge_dict[pair] += int(row.get("value"))
	
	#print(dictionary)
	x = []
	for key in dictionary:
		x.append([dictionary[key]])
		node_to_addr[dictionary[key]] = key
	#print(node_to_addr)
	
	edges = []
	for pair in edge_dict:
		edges.append([dictionary[pair[0]], dictionary[pair[1]]])

	edge_index = torch.tensor(edges, dtype=torch.long)
	data = Data(num_nodes=len(x), x=x, edge_index=edge_index.t().contiguous())
	return data,count

start_index = 0
for dt in date_range:
	dt_o = dt.to_pydatetime()
	print(dt_o.strftime("%Y/%m/%d"))
	#try:
	sub_df, start_index = get_df_uptill(dt_o, start_index)
	data,count = getGraph(sub_df, count)
	#print(node_to_addr)
	#print(dictionary)
	g = torch_geometric.utils.to_networkx(data, to_undirected=False)
	#print(g.nodes)
	for node in g.nodes():
		num_adj_nodes = len(list(g.neighbors(node)))
		if node_to_addr[node] not in raw_data:
			raw_data[node_to_addr[node]] = {}
		raw_data[node_to_addr[node]]["Address"] = node_to_addr[node]
		raw_data[node_to_addr[node]][dt_o.strftime("%Y/%m/%d")] = num_adj_nodes
	#except Exception as e:
		#print(str(e))

lis = []
for row in raw_data:
	lis.append(raw_data[row])

#print(lis)
'''
