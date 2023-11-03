import pandas as pd
import torch
from torch_geometric.data import Data
import torch_geometric.utils
from datetime import datetime
from collections import OrderedDict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#assumed dates from only first year of FTT transaction file
start = datetime(2019, 7, 27)
end = datetime(2020, 7, 27)

ftt = pd.read_csv("2019_2020_FTT.csv")

date_range = pd.date_range(start, end)
raw_data = OrderedDict()

#produces a subset dataframe based on the date we are interested in observing
def get_df_uptill(dt):
	for index, row in ftt.iterrows():
		date = datetime.strptime(row.get("block_timestamp")[:10], "%Y-%m-%d")
		delta = dt - date
		if delta.days < 0:
			cutoff = int(index)
			#print(cutoff)
			return ftt[:cutoff]
	return ftt

#generates a graph from a given dataframe
def getGraph(df):
	dictionary = {}
	
	node_to_addr = {}
	edge_value = []
	
	count = 0
	edges = []
	x = []
	
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
	
	#print(dictionary)
	
	for key in dictionary:
		x.append([dictionary[key]])
		node_to_addr[dictionary[key]] = key
	#print(node_to_addr)
	
	for index, row, in df.iterrows():
		from_addr = row.get("from_address")
		to_addr = row.get("to_address")
		edges.append([dictionary[from_addr], dictionary[to_addr]])
	
	edge_index = torch.tensor(edges, dtype=torch.long)
	data = Data(num_nodes=len(x), x=x, edge_index=edge_index.t().contiguous())
	return data, node_to_addr, dictionary

#Used for normalizing values from 0 - 1
value_list = ftt["value"]
[int(value) for value in value_list]

max_value = int(max(value_list))


#Adds value from transaction dataframe to the edges of their nodes given by from and to address
def add_edge_value(df, addr_to_node):
	edge_attr = {}
	value_accumalator = {}
	for index, row in df.iterrows():
		from_node = addr_to_node[row.get("from_address")]
		to_node = addr_to_node[row.get("to_address")]
		
		pair_tup = (from_node, to_node)
		attr = {}
		if pair_tup not in value_accumalator:
			value_accumalator[pair_tup] = int(row.get("value")) / max_value
		else:
			value_accumalator[pair_tup] += int(row.get("value")) / max_value
		
		attr["weight"] = value_accumalator[pair_tup]
		edge_attr[pair_tup] = attr
	return edge_attr


for dt in date_range:
	dt_o = dt.to_pydatetime()
	print(dt_o.strftime("%Y/%m/%d"))
	try:
		#take a subset of the dataframe uptill the iterating date
		df = get_df_uptill(dt_o)
		data, node_to_addr, addr_to_node = getGraph(df)
		#create networkX graph & add edge values
		g = torch_geometric.utils.to_networkx(data, to_undirected=False)
		nx.set_edge_attributes(g, add_edge_value(df, addr_to_node))
		
		# from the docs, find a stable alpha const such that the calculation is stable
		# alpha = 1/max elgenvalue of adjacency matrix
		A = nx.to_numpy_array(g, weight="weight")
		elg, V = np.linalg.eig(A)
		alpha = 1/((max(elg)) +1)
		
		#compute centrality for all nodes, with the weights applied to each edge
		centrality = nx.katz_centrality(g, alpha=alpha, max_iter=1000000, weight="weight")
		#print(g.nodes())
		#store score to their respective nodes for csv writing
		for node, center_score in centrality.items():
			#print(node_to_addr[node])
			if node_to_addr[node] not in raw_data:
				raw_data[node_to_addr[node]] = {}
			raw_data[node_to_addr[node]]["Address"] = node_to_addr[node]
			raw_data[node_to_addr[node]][dt_o.strftime("%Y/%m/%d")] = center_score
	except Exception as e:
		print(str(e))

lis = []
for row in raw_data:
	lis.append(raw_data[row])

#print(lis)

adj = pd.DataFrame(lis)
adj.to_csv("node_centrality.csv")

