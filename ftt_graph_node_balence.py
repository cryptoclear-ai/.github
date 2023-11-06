import pandas as pd
from datetime import datetime

filename = input("total amount sent csv: ")
money_out = pd.read_csv(filename)
filename = input("total amount recieved csv: ")
money_in = pd.read_csv("total_money_recieved.csv")

balence = money_in

for row in range(len(money_in)):
	for col in range(len(money_in.columns)):
		if col <= 1:
			continue
		print(row, col)
		print(type(money_in.iat[row,col]))
		if type(money_in.iat[row,col]) == float or type(money_out.iat[row,col]) == float:
			continue
		balence.iat[row, col] = int(money_in.iat[row,col]) - int(money_out.iat[row,col])

balence.to_csv("node_balence.csv")
