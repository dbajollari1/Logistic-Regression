import pandas as pd
from collections import Counter
import numpy as np
#from sklearn.linear_model import LogisticRegression

#public varible 
docFreq = {} #stores counts of each unique word in tweets

class MyLogisticRegression:
    def __init__(self,x,y):      
        self.intercept = np.ones((x.shape[0], 1))  
        self.x = np.concatenate((self.intercept, x), axis=1)
        self.weight = np.zeros(self.x.shape[1])
        self.y = y
        
    #Sigmoid method
    def sigmoid(self, x, weight):
        z = np.dot(x, weight)
        return 1 / (1 + np.exp(-z))
    
    #method to calculate the Loss
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    #Method for calculating the gradients
    def gradient_descent(self, X, h, y):
        return np.dot(X.T, (h - y)) / y.shape[0]

    
    def fit(self, lr , iterations):
        for i in range(iterations):
            sigma = self.sigmoid(self.x, self.weight)
            
            loss = self.loss(sigma,self.y)

            dW = self.gradient_descent(self.x , sigma, self.y)
            
            #Updating the weights
            self.weight -= lr * dW

        return print('fitted successfully to data')
    
    #Method to predict the class label.
    def predict(self, x_new , treshold):
        x_new = np.concatenate((self.intercept, x_new), axis=1)
        result = self.sigmoid(x_new, self.weight)
        result = result >= treshold
        y_pred = np.zeros(result.shape[0])
        for i in range(len(y_pred)):
            if result[i] == True: 
                y_pred[i] = 1
            else:
                continue
                
        return y_pred

#preprocess texual data before feature extraction
def preProcessData(): 
    dataTable = pd.read_csv('swad_train.csv')

    # convert all tweets to lowercase
    dataTable['Tweet'] = dataTable['Tweet'].str.lower() 

    #pad all punctuation from punctuation.txt
    punctuations = open("punctuations.txt").read().split()
    for punctuation in punctuations:
        dataTable['Tweet'] = dataTable['Tweet'].str.replace(punctuation,' ' + punctuation + ' ')

    #remove stopwords from Tweet
    stopWords= open("stopwords.txt").read().split()
    dataTable['Tweet'] = dataTable['Tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopWords)]))

    return dataTable

#returns frequency of word in documet
def get_word_freq(word):
    c = 0
    try:
        c = docFreq[word]
    except:
        pass
    return c

#calculates the counts of each unique words in the dataset
def set_doc_freqs(dataTable):
    for i in range(len(dataTable['Tweet'])):
        tokens = dataTable['Tweet'][i]
        splits = tokens.split()
        for w in splits:
            try:
                docFreq[w].add(i)
            except:
                docFreq[w] = {i}
    for i in docFreq:
        docFreq[i] = len(docFreq[i])


#extract tf_idf features from dataset
def get_tf_idf(dataTable):
    doc = 0
    tf_idf = {}
    N = len(dataTable['Tweet'])

    for i in range(N):
        tokens = dataTable['Tweet'][i].split()
    
        counter = Counter(tokens + dataTable['Tweet'][i].split())
        words_count = len(tokens + dataTable['Tweet'][i].split())
    
        for token in np.unique(tokens):
        
            tf = counter[token]/words_count
            df = get_word_freq(token)
            idf = np.log((N+1)/(df+1))
        
            tf_idf[doc, token] = tf*idf
    
        doc += 1

    return tf_idf

# Document Vectorization
def vectorization(tf_idf, N):
    total_vocab_size = len(docFreq)
    total_vocab = [x for x in docFreq]
    vectors = np.zeros((N, total_vocab_size))
    for i in tf_idf:
        ind = total_vocab.index(i[1])
        vectors[i[0]][ind] = tf_idf[i]
    
    return vectors

def main():
    dataTable = preProcessData()
    set_doc_freqs(dataTable)
    tf_idf = get_tf_idf(dataTable)
    
    vectors = vectorization(tf_idf, len(dataTable['Tweet']))


    #Preparing the data
    x = vectors
    y = dataTable['Label'].map({'Yes':1 ,'No':0})

    #creating the class Object
    regressor = MyLogisticRegression(x,y)

    #
    regressor.fit(0.1 , 1000)

    y_pred = regressor.predict(x,0.37)

    accruacy = sum(y_pred == y) / y.shape[0]
    print('accuracy = ' + format(accruacy))

    """ logisticRegr = LogisticRegression()
    logisticRegr.fit(x,y)
    predictions = logisticRegr.predict(x)
    score = logisticRegr.score(x,y)
    print(score) """

    
if __name__ == "__main__":
    main()