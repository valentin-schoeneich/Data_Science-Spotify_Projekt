import csv
import pandas as pd


def makeCSVUnique(filename):
    print('reading ' + filename)
    df = pd.read_csv(filename + '.csv')
    print('dropping duplicates in ' + filename)
    df = df.drop_duplicates()
    print('writing to csv')
    df.to_csv(filename + '_unique.csv')


makeCSVUnique('albums')
makeCSVUnique('tracks')
makeCSVUnique('artists')
