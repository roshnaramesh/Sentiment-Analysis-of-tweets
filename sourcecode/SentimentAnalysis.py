'''
Created on April 30th, 2015

@author: Roshna and Ashwin
'''

import logging
from ReadTweets import readExcelFile
import NBayesClassifier

TRAINING_DATA_FILE = '../data/training-Obama-Romney-tweets.xlsx'
TEST_DATA_FILE = '../data/Project2_Testing.xlsx'

def word_feature_set(words):
    return dict([(word, True) for word in words])

if __name__ == '__main__':
    
    # Create a Naive Bayes NaiveBayesClassifier from the created Classifier Class
    Classifier_obama = NBayesClassifier.NaiveBayesAlgo()
    Classifier_romney = NBayesClassifier.NaiveBayesAlgo()

    # Read the given Excel file to retrieve Training data
    training_tweets_obama = readExcelFile(TRAINING_DATA_FILE, 'Obama', 'train')
    training_data_features_obama = Classifier_obama.retrieve_features(word_feature_set, training_tweets_obama)
    training_tweets_romney = readExcelFile(TRAINING_DATA_FILE, 'Romney', 'train') 
    training_data_features_romney = Classifier_romney.retrieve_features(word_feature_set, training_tweets_romney)
    
    # Same Excel File Call to Retrieve Test Data
    test_tweets_obama = readExcelFile(TEST_DATA_FILE, 'obama-test', 'test')
    test_tweets_romney = readExcelFile(TEST_DATA_FILE, 'romney-test', 'test')
    test_feature_data_obama = Classifier_obama.retrieve_features(word_feature_set, test_tweets_obama)
    test_features_data_romney = Classifier_romney.retrieve_features(word_feature_set, test_tweets_romney)

    # Train the data set using the given training set using Naive Bayes Classifier
    Classifier_obama.TrainAlgo(training_data_features_obama, test_feature_data_obama)
    Classifier_romney.TrainAlgo(training_data_features_romney, test_features_data_romney)
    
    # Calculate the accuracy, F-score, Precision, Recall for Obama tweets
    print "Classification Results for Obama"
    Classifier_obama.accuracy()
    Classifier_obama.stats()
    

    # Calculate the accuracy, F-score, Precision, Recall for Romney tweets
    print "\n \n Classification results for Romney"
    Classifier_romney.accuracy()
    Classifier_romney.stats()

    # Print the confusion matrix for Obama
    print "\n \n Confusion matrix for Obama"
    Classifier_obama.confusion_matrix()


    #Print the confusion matrix for Romney
    print "\n \n Confusion matrix for Romney"
    Classifier_romney.confusion_matrix()
#end
