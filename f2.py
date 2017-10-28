import csv


#IGDTUW - Berkeley - 0.595453099969

import numpy as np
import sys
import numpy as np
from sklearn import metrics

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))
    #return metrics.mutual_info_score(a,b)	
    
#ETH Zurichmodel_curriculum=[0.06, 0.0238095238095, 0.0238095238095, 0.0681818181818, 0.0681818181818, 0.0681818181818, 0.0652173913043, 0.0238095238095, 0.0652173913043, 0.0625, 0.0238095238095, 0.0238095238095, 0.0625, 0.0681818181818, 0.0652173913043, 0.0681818181818, 0.0238095238095, 0.0238095238095, 0.0238095238095, 0.0238095238095, 0.0238095238095, 0.06, 0.0681818181818, 0.06, 0.0238095238095, 0.0681818181818, 0.0238095238095, 0.0681818181818, 0.0238095238095, 0.0238095238095, 0.0238095238095, 0.0238095238095, 0.0681818181818, 0.0555555555556, 0.0238095238095, 0.0238095238095, 0.0238095238095, 0.0238095238095, 0.0652173913043, 0.0238095238095, 0.0238095238095, 0.0555555555556, 0.0681818181818, 0.104166666667, 0.0625, 0.0681818181818, 0.0238095238095, 0.0681818181818, 0.06, 0.0238095238095]#ETH_Zurich

#Berkeley
model_curriculum = [0.0792156862745, 0.0104529616725, 0.0108303249097, 0.0104529616725, 0.0104529616725, 0.00990099009901, 0.0100334448161, 0.00363636363636, 0.0108303249097, 0.0108303249097, 0.0223642172524, 0.0104529616725, 0.00363636363636, 0.010752688172, 0.0105263157895, 0.00363636363636, 0.00363636363636, 0.0108303249097, 0.0170648464164, 0.0108303249097, 0.0108303249097, 0.00363636363636, 0.0161812297735, 0.0106761565836, 0.0148367952522, 0.0104529616725, 0.00363636363636, 0.0108303249097, 0.00363636363636, 0.0106761565836, 0.00977198697068, 0.00990099009901, 0.00363636363636, 0.0108303249097, 0.010752688172, 0.010101010101, 0.0102389078498, 0.016835016835, 0.0106007067138, 0.0106761565836, 0.010752688172, 0.0104529616725, 0.00363636363636, 0.0344827586207, 0.00363636363636, 0.00363636363636, 0.00996677740864,  0.00363636363636, 0.0105263157895, 0.00363636363636]
import numpy as np

corpus = open('IGDTUW.txt').read()
#Creatinig dictionary of unique terms, indexed and counted
dic = {}    
arr=[]
arrv=[]
for item in corpus.split():  
     if item in dic:
         dic[item] += 1
     else:  
         dic[item] = 1  

arr = dic.keys()
arrv= dic.values()
arrid=range(0,len(arr))

#Replacing actual words in doc with the word id's
Imgvv=[]
for w in corpus.split():
    for i in arrid:
        if w == arr[i]:
            Imgvv.append(i)

Imgv = [Imgvv] # Array of (array of) words in documents (replaced with id's)
Vocab = arr #Vocab of unique terms

I =  len(Imgv) #Image number
M = 50# Part number - hardwired (supervised learning)
V = len(Vocab) #vocabulary

#Dirichlet constants
alpha=0.5
beta=0.5

#Initialise the 4 counters used in Gibbs sampling
Na = np.zeros((I, M)) + alpha     # umber of words for each document, topic combo i.e 11, 12,13 -> 21,22,23 array.
Nb = np.zeros(I) + M*alpha        # number of words in each image
Nc = np.zeros((M, V)) + beta      # word count of each topic and vocabulary, times the word is in topic M and is of vocab number 1,2,3, etc..
Nd = np.zeros(M) + V*beta         # number of words in each topic

m_w = [] #topic of the current word
m_i_w=[] # topic of the image of the word 
#Filling up counters
for i,img in enumerate(Imgv):
    for w in img:
        m = np.random.randint(0,M)
        m_w.append(m)
        Na[i,m] += 1
        Nb[i] += 1
        Nc[m,w] += 1
        Nd[m] += 1  
    m_i_w.append(np.array(m_w)) #creating a relationship between topic to word per doc

#Gibbs Sampling
m_i=[]
q = np.zeros(M) 
for t in xrange(500): #Iterations   
    for i,img in enumerate(Imgv): #in the Imgv matrix there are i documents which are arrays (img) filled with words
        m_w = m_i_w[i] #Finding topic of word
        Nab = Na[i] #Taking ith row of the Na counter (array)
        for n, w in enumerate(img): #in img there are n words of value w
            m = m_w[n]  # From the intialised/appended topic-word value we draw the "guessed" topic
            Nab[m] -= 1 
            Nb[i] -= 1  #In Gibbs Samp. we compute for all values except the current (x,y) position
            Nc[m,w] -= 1 #So we move the counter of this positon down one, compute
            Nd[m] -= 1 #And then add one back after reloading the topic for the word

            q = (Nab*(Nc[:,w]))/((Nb[i])*(Nd)) # computing topic probability
            q_new = np.random.multinomial(1, q/q.sum()).argmax() # choosing new topic based on this
            m_w[n] = q_new      # assigning word to topic, replacing the guessed topic from init.

            Nab[q_new] += 1 #Putting the counters back to original value before redoing process.
            Nb[i] += 1
            Nc[q_new,w] += 1
            Nd[q_new] += 1

WordDist = Nc/Nd[:, np.newaxis]  # This gives us the words per topic

present = []

for m in xrange(M): #Displaying results
    for w in np.argsort(-WordDist[m])[:1]:
	#print(WordDist[m,w])
        present.append(WordDist[m,w])

from scipy.stats import entropy      

print(entropy(model_curriculum, present))








