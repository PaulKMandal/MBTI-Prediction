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
