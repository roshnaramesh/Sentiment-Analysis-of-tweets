'''
Created on April 30th, 2015.

@author: Roshna and Ashwin
'''

import xlrd

ab_dictionary = {}
stopwords_dictionary = []

# Read tweet data from given Excel/CSV file
def readExcelFile(filename, pres_candidate, category):
    # Open excel workbook
    try:
        wb = xlrd.open_workbook(filename)
    except:
        print 'FILE NOT FOUND!!'
    
    original_tweets = []

    # Select worksheet/sheet based on name such as Obama or Romney    
    sheetSelected = wb.sheet_by_name(pres_candidate)
    number_of_tweets = sheetSelected.nrows

    # Categorize if the retrieved data is either Test data or Training data, so as to ignore Training tweets in case if its test data
    if category == 'train':
        for row_number in range(2, number_of_tweets):
            try:
                #Ignore the encoding whether ascii or utf-8
                #Select tweet from the third column and train the data set using the sentiment value retrieved from column 4.
                tweet = ''.join(sheetSelected.cell(row_number, 3).value).encode('ascii','ignore').strip()
                sentiment = sheetSelected.cell(row_number, 4).value
            except:
                print "Fatal Error Please Check. row_number: ", ''.join(sheetSelected.cell(row_number, 3).value)

            #Check if sentiment is present between positive, negative or neutral
            if sentiment not in (1.0, -1.0, 0.0):
                sentiment = 0.0
            
            # Store the tweet as a record. Append it to the original set of tweets stored as a list.
            Record_tweet = tweet, sentiment
            original_tweets.append(Record_tweet)
    else:
        if pres_candidate == 'Obama':
            for row_number in range(number_of_tweets):
                tweet = ''.join(sheetSelected.cell(row_number, 0).value).encode('ascii','ignore').strip()
                sentiment = sheetSelected.cell(row_number, 4).value
                
                Record_tweet = tweet, sentiment
                original_tweets.append(Record_tweet)
        else:
            for row_number in range(2, number_of_tweets):
                tweet = ''.join(sheetSelected.cell(row_number, 3).value).encode('ascii','ignore').strip()
                sentiment = sheetSelected.cell(row_number, 6).value
                
                Record_tweet = tweet, sentiment
                original_tweets.append(Record_tweet)
    
    return original_tweets

# Enhance Tweets by removing Stop words
def readStopwordsFile(stopwordsFile):
    global stopword_dictionary
    
    with open(stopwordsFile) as f:
        lines_number = f.readlines()
    
    for stopword in lines_number:
        stopwords_dictionary.append(stopword.strip())
    
    return list(set(stopwords_dictionary))

# Read a simple text file containing some abbreviations and their expansions. Use this text file to remove such words/expand them to their abbreviations for text - preprocessing

def Abbreviations_From_File(FlatFile):
    global ab_dictionary
    
    f = open(FlatFile)
    number_of_lines = f.readlines()
    f.close()
    for i in number_of_lines:
        tmp = i.split('|')
        ab_dictionary[tmp[0]] = tmp[1]

    return ab_dictionary
