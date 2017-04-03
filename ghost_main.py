#GHOST: Genre, Happening, or Sentiment Tagger
#Michael Berezny, Pat Putnam, Sushant Kafle

"""Import"""
import csv

"""Read Training Data"""
data=[]
with open("PS3_training_data.txt", "r") as f:
    reader=csv.reader(f,delimiter='\t')
    for case in reader:
        d={"ID":case[0], "Sentence":case[1], "Sentiment":case[2], "Event":case[3], "Genre":case[4]}
        data.append(d)

print(data[-3:])

