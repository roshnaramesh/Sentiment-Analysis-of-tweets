'''
Created on May 30th, 2015

@authors: Roshna and Ashwin
'''

from DataPreProcessing import Segregate_Tweets
from nltk.classify import NaiveBayesClassifier
from nltk.metrics.confusionmatrix import ConfusionMatrix
import nltk.classify.util
import nltk.metrics
import collections

class NaiveBayesAlgo:
    classifier = None
    TrainAlgo_features = None
    test_list = []
    testing_features = None
    originalTweets = None
    classifiedTweets = None
    reference_collection = []
    
    
    # Create the feature sets as learnt from TrainAlgoing data set
    def retrieve_features(self, featx, orig_tweets):
        
        posTweets, negTweets, neuTweets = Segregate_Tweets(orig_tweets,)
        
        positiveFeatures = [(featx(tweets),'pos') for tweets, sent in posTweets]
        negativeFeatureSet = [(featx(tweets),'neg') for tweets, sent in negTweets]
        neutralFeatureSet = [(featx(tweets),'neu') for tweets, sent in neuTweets]
        
        
        Extracted_Features = positiveFeatures + negativeFeatureSet + neutralFeatureSet

        return Extracted_Features

    # TrainAlgo the feature sets to produce reference and originalTweets lists
    def TrainAlgo(self, TrainAlgo_features, testing_features):
        self.classifier = NaiveBayesClassifier.train(TrainAlgo_features)
        self.TrainAlgo_features = TrainAlgo_features
        self.testing_features = testing_features
        
        self.originalTweets = collections.defaultdict(set)
        self.classifiedTweets = collections.defaultdict(set)
        
        self.reference_collection = list()
        self.test_list = list()
        for i, (feats, label) in enumerate(testing_features):
            self.originalTweets[label].add(i)
            self.reference_collection.append(label)
            observed = self.classifier.classify(feats)
            self.classifiedTweets[observed].add(i)
            self.test_list.append(observed)

    
    # Calculate Precision, Recall and F-score of positive, negative, neutral and mixed sentiment tweets
    def stats(self):
        
        print 'POSITIVE CLASS'
        print 'precision:', nltk.metrics.precision(self.originalTweets['pos'], self.classifiedTweets['pos'])
        print 'recall:', nltk.metrics.recall(self.originalTweets['pos'], self.classifiedTweets['pos'])
        print 'F-score:', nltk.metrics.f_measure(self.originalTweets['pos'], self.classifiedTweets['pos'])
        print
        print 'NEGATIVE CLASS'
        print 'precision:', nltk.metrics.precision(self.originalTweets['neg'], self.classifiedTweets['neg'])
        print 'recall:', nltk.metrics.recall(self.originalTweets['neg'], self.classifiedTweets['neg'])
        print 'F-score:', nltk.metrics.f_measure(self.originalTweets['neg'], self.classifiedTweets['neg'])
        print
        print 'NEUTRAL CLASS'
        print 'precision:', nltk.metrics.precision(self.originalTweets['neu'], self.classifiedTweets['neu'])
        print 'recall:', nltk.metrics.recall(self.originalTweets['neu'], self.classifiedTweets['neu'])
        print 'F-score:', nltk.metrics.f_measure(self.originalTweets['neu'], self.classifiedTweets['neu'])
        
    # Compute Accuracy
    def accuracy(self):
        print 'ACCURACY For Data Set:', nltk.classify.util.accuracy(self.classifier, self.testing_features)

    # Build and print the Confusion Matrix
    def confusion_matrix(self):
        cm = ConfusionMatrix(self.reference_collection, self.test_list)
        print cm;
