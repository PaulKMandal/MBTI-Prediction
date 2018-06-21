#!/usr/local/bin/python3
import csv
import json

# opening the file csv to clean
csvDataFile = open('mbti_1.csv', 'r')
jsonDataFile = open('formatted_data.json', 'w')
fieldNames = ("Type", "Post")

reader = csv.DictReader(csvDataFile, fieldNames) 

for row in reader:
        json.dump(row, jsonDataFile)
        jsonDataFile.write('\n')

# saving each line of the data(type, post) into a list for formatting
# dataList = list(dataFile)

# print(dataList[1][7:len(dataList[1]) - 3])
# data = {}
# data['type'] = dataList[1][:4]
# data['post'] = dataList[1][7:len(dataList[1]) - 3]
# data['type'] = dataList[2][:4]
# data['post'] = dataList[2][7:len(dataList[1]) - 3]
# json_data = json.dumps(data)

# with open('formatted_data.json', 'w') as outfile:
#    json.dump(json_data, outfile)

