'''
Updated on 30th April, 2015

@authors : Roshna and Ashwin
'''
import re
from ReadTweets import Abbreviations_From_File, readStopwordsFile
from nltk.stem import WordNetLemmatizer as wnl
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import naive_bayes
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import string

exclude = set(string.punctuation)
stopwords2File = '../data/stopwords2.txt'
abbrFile = '../data/abbr.txt'
contractions_dict = { 
"ain't": "am not; are not; is not; has not; have not",
"aren't": "are not; am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

stopwords2 = readStopwordsFile(stopwords2File)
abbreivations_dictionary = Abbreviations_From_File(abbrFile)

# This function segregates tweets into positive, negative, neutral and mixed tweets
def Segregate_Tweets(actual_tweet_fromFile):
    positive_tweets = []
    negative_tweets = []
    neutral_tweets = []

    for tweet,sentiment in actual_tweet_fromFile:
        if sentiment == 0.0:
            neutral_tweets.append((process_tweet(tweet),sentiment))
        elif sentiment == 1.0:
            positive_tweets.append((process_tweet(tweet),sentiment))
        elif sentiment == -1.0:
            negative_tweets.append((process_tweet(tweet),sentiment))
    return positive_tweets, negative_tweets, neutral_tweets
#end

# Series of Data-Pre-Processing Steps performed to process tweets retrieved from Excel File
def process_tweet(tweet):
    stemmer = SnowballStemmer("english")
    #Removing URls
    tweet = re.sub('((www\.[\s]+)|(https?://[^\s]+))','',tweet)
    
    #Replace 2 or more repetitions of a character
    tweet = replaceTwoOrMore(tweet)

    #Removing usernames
    tweet = re.sub('@[^\s]+','',tweet)

    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    #Replace the hex code "\xe2\x80\x99" with single quote
    tweet = re.sub(r'\\xe2\\x80\\x99', "'", tweet)
    
    #Removing <e> and <a> tags
    tweet = re.sub(r'(<e>|</e>|<a>|</a>|\n)', '', tweet)
    
    #Removing apostrophe
    tweet = tweet.replace("\'s",'')
    
    #Expanding contractions. For example, "can't" will be replaced with "cannot"
    tweet = expand_contractions(tweet)

    #Removing punctuation
    tweet = ''.join(ch for ch in tweet if ch not in exclude)
    
    #Removing words that end with digits
    tweet = re.sub(r'\d+','',tweet)
    
    #Removing words that start with a number or a special character
    tweet = re.sub(r"^[^a-zA-Z]+", ' ', tweet)

    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[\s]+)|(https?://[^\s]+))',' ', tweet)

    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)

    # remove underscores
    tweet = re.sub('_',' ', tweet)

    #deal with horizontal ellipsis
    tweet = re.sub(u'\u2026', ' ', tweet)

    #deal with double quotation mark
    tweet = re.sub(u'[\u201c\u201d]', '"', tweet)

    #deal with single quotation mark
    tweet = re.sub(u'[\u2018\u2019]', '\'', tweet)
    
    #deal with truncated url
    tweet = re.sub('(^|)?http?s?:?/?/?.*?( |$)', ' ', tweet)          

    #deal with retweet
    tweet = re.sub(u'(RT |\\\\|\u201c)"?@.*?[: ]', ' ', tweet)

    #deal with username
    tweet = re.sub('\.?@.*?( |:|$)', ' ', tweet)
    
    tweet = re.sub(r"\.\.+",' ', tweet) 

    #Remove hash character
    tweet = re.sub('[#]', ' ', tweet)
    # remove special symbols
    tweet = re.sub('[][!"$*,/;<=>?@\\\\^_`{|}~]', ' ', tweet)        

    #remove repition of letters
    tweet = re.sub('( - )', ' ', tweet)
    tweet = re.sub('---', ' ', tweet)
    tweet = re.sub('\.\.\.', ' ', tweet)
    tweet = re.sub('(, |\.( |$))', ' ', tweet)
    
    #tweet = re.sub("\S*\d\S*", " ", tweet).strip()
    tweet = re.sub(r'[^\x00-\x7F]',' ', tweet)
    tweet = re.sub('@',' ', tweet)
    tweet = re.sub(r'\\([^\s]+)',' ', tweet)
    punctuation = re.compile(r'[-.?!,":;()|$%&*+/<=>[\]^`{}~]')

    #Replace with space the punctuation 
    tweet = punctuation.sub(' ', tweet)
    tweet = re.sub('&amp', ' ', tweet)

    tweet = re.sub(r'pic.twitter.*?$', '', tweet)
    tweet = re.sub(r'pic.twitter.*? ', '', tweet)
    tweet = re.sub('((www\.[^\s]+)|(http://[^\s]+))','',tweet)  #remove url
    tweet = re.sub(r'#([^\s])*$', '', tweet)
    tweet = re.sub(r'#([^\s])* ', '', tweet)
    tweet = re.sub(r'\@([^\s])*$', '', tweet)
    tweet = re.sub(r'\@([^\s])* ', '', tweet)
    tweet = re.sub("\d",'',tweet)                               #remove digits
    tweet = re.sub("[!\'.\"%/*$;:\(\):,?]",'',tweet)                        #remove special characters
    tweet = re.sub(r'\-',' ',tweet)                             #replace - with white space
    tweet = re.sub(r'\'m', ' am', tweet)
    tweet = re.sub(r'\'d', ' would', tweet)
    tweet = re.sub(r'\'ll', ' will', tweet)
    tweet = re.sub(r'\&', 'and', tweet)
    tweet = re.sub(r'\b\w{1,3}\b','', tweet).strip()            #remove 3 or lesser length words
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub('<>', '', tweet)

    # Process emoticons
    tweet = replace_emoticons(tweet)
    
    #remove trailing duplicate characters
    words=[]
    for word in tweet.split():
        l=word[-1]
        word = word.rstrip(word[-1])
        word += l
        words.append(word)
    tweet = ' '.join(words)

    #stemming
    words=[]
    for word in tweet.split():
        data = stemmer.stem(word)
        words.append(str(data))
    tweet = ' '.join(words)

    #stop word removal
    stop = stopwords.words('english')
    final = [i for i in tweet.split() if i not in stop]
    tweet = ' '.join(final)
    
            
    tweet = tweet.strip('\'"')


    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)

    #Replace all words that don't start with a letter, number or an underscore with an empty string
    tweet = re.sub(r'\\[xa-z0-9.*]+', '', tweet)
    
    #Remove trailing spaces and full stops
    tweet = tweet.strip(' .')
    
    #Convert CamelCaseWords to space delimited words
    tweet = convertCamelCase(tweet)
    
    #Convert everything to lower characters
    tweet = tweet.lower()

    #Tokenize the tweet
    tweet = tokenize_tweet(tweet)

    #Replace abbreviations with their corresponding meanings
    tweet = ExpandAbbreviation(tweet)
    
    #Lemmatize the words in tweets
    tweet = wordLemmatizer(tweet)
    
    #Remove stopwords2 from the tweet
    tweet = removeStopWords(tweet, stopwords2)
    
    #Removing duplicates
    tweet = list(set(tweet))
        
    return tweet
#end

# Removing stopwords2
def removeStopWords(tweet, stopwords2):
    tmp = []
    for i in tweet:
        if i not in stopwords2:
            tmp.append(i)

    return tmp
#end


# This method replaces two or more consecutive letters with the same character to
# something shorter. For example, gooooooood becomes good.
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL) 
    return pattern.sub(r"\1\1", s)
#end

# This method converts camel cased words into space delimited words.
# For example: ThisIsASentence will be changed to This Is A Sentence
def convertCamelCase(word):
    return re.sub("([a-z])([A-Z])","\g<1> \g<2>",word)
#end

def is_ascii(self, word):
    return all(ord(c) < 128 for c in word)
#end

# This function checks the dictionary containing abbreviations and their meanings as (key,value) pairs
# and replaces the key with the corresponding value
def ExpandAbbreviation(s):
    for word in s:
        if word.lower() in abbreivations_dictionary.keys():
            s = [abbreivations_dictionary[word.lower()] if word.lower() in abbreivations_dictionary.keys() else word for word in s]
    return s
#end

# Tokenize the tweet and split the words
def tokenize_tweet(tweet):
    return word_tokenize(tweet)
#end

# This method lemmatizes each word in a tweet. The method accepts a list, lemmatizes each word
# and returns back the list
def wordLemmatizer(tweet_words):
    return [wnl().lemmatize(word) for word in tweet_words]



def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))


def replace_emoticons(emoticons_tweet):
        #"""Replace the emoticons of the input string with the corresponding string"""
        # function to preprocess    
    conv_dict_multi = { ('>:]', ':-)', ':)', ':o)', ':]', ':3', ':c', ':>', '=]', '8)', '=)', ':}', ':^)', ':)','|;-)', '|-o):', '>^_^<', '<^!^>', '^/^', '(*^_^*)', '(^<^)', '(^.^)', '(^?^)', '(^?^)', '(^_^.)', '(^_^)', '(^^)', '(^J^)', '(*^?^*)', '^_^', '(^-^)', '(?^o^?)', '(^v^)', '(^u^)', '(^?^)', '( ^)o(^ )', '(^O^)', '(^o^)', '(^?^)', ')^o^('):'_HAPPY_',
            ('>:[', ':-(', ':(', ':-c', ':c', ':-<', ':<', ':-[', ':[', ':{', '>.>', '<.<', '>.<', '(\'_\')', '(/_;)', '(T_T)', '(;_;)', '(;_:)', '(;O;)', '(:_;)', '(ToT)', '(T?T)', '(>_<)', '>:\\', '>:/', ':-/', ':-.', ':/', ':\\', '=/', '=\\', ':S'):'_SAD_',
            ('>:D', ':-', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D', '=-3', '=3', '8-)', ':-))'):'good',
            ('D:<', 'D:', 'D8', 'D;', 'D=', 'DX', 'v.v', '>:)', '>;)', '>:-)', ':\'-(', ' :\'-)', ':\')', ':-||'):'bad'}
            #Convert to the one-to-one dict
    conv_dict = {}
    for k, v in conv_dict_multi.items():
        for key in k:
            conv_dict[key] = v
            #Replace the emoticons        
        for smiley, conv_str in conv_dict.iteritems():
            emoticons_tweet = emoticons_tweet.replace(smiley, conv_str)
        return emoticons_tweet
#end
