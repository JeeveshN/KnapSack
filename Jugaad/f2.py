import csv
import numpy as np
import sys
import numpy as np
from sklearn import metrics

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))
    #return metrics.mutual_info_score(a,b)	

#MIT    
model_curriculum_MIT = [0.0294117647059, 0.0294117647059, 0.0294117647059, 0.0294117647059, 0.0294117647059, 0.0789473684211, 0.0294117647059, 0.0789473684211, 0.0294117647059, 0.0833333333333, 0.0294117647059, 0.0294117647059, 0.0294117647059, 0.0294117647059, 0.0789473684211, 0.0294117647059, 0.0294117647059, 0.0294117647059, 0.0833333333333, 0.0294117647059, 0.0294117647059, 0.0294117647059, 0.0789473684211, 0.125, 0.0833333333333, 0.0789473684211, 0.0833333333333, 0.108695652174, 0.0833333333333, 0.0294117647059, 0.0833333333333, 0.0833333333333, 0.0294117647059, 0.0294117647059, 0.0833333333333, 0.0294117647059, 0.0294117647059, 0.075, 0.0833333333333, 0.0294117647059, 0.0833333333333, 0.0789473684211, 0.0833333333333, 0.0294117647059, 0.0294117647059, 0.0294117647059, 0.0714285714286, 0.0294117647059, 0.0294117647059, 0.0833333333333
]

#UCBerkeley

model_curriculum_UCB = [0.03125, 0.0283018867925, 0.030612244898, 0.03, 0.03125, 0.0106382978723, 0.0288461538462, 0.0106382978723, 0.0106382978723, 0.0106382978723, 0.0106382978723, 0.0106382978723, 0.0106382978723, 0.03125, 0.0267857142857, 0.0833333333333, 0.0471698113208, 0.0106382978723, 0.0106382978723, 0.0106382978723, 0.0283018867925, 0.030612244898, 0.0294117647059, 0.03125, 0.0288461538462, 0.03125, 0.03125, 0.0288461538462, 0.0106382978723, 0.0288461538462, 0.0294117647059, 0.03125, 0.03125, 0.0106382978723, 0.03125, 0.0106382978723, 0.0288461538462, 0.0106382978723, 0.0283018867925, 0.0245901639344, 0.030612244898, 0.03125, 0.0438596491228, 0.03, 0.03, 0.0106382978723, 0.0106382978723, 0.0106382978723, 0.03125, 0.03125]

#ETHZurich

model_curriculum_ETH = [0.0344827586207, 0.0344827586207, 0.0344827586207, 0.0967741935484, 0.0344827586207, 0.0967741935484, 0.0344827586207, 0.0967741935484, 0.0344827586207, 0.0344827586207, 0.0967741935484, 0.0344827586207, 0.0344827586207, 0.0344827586207, 0.0344827586207, 0.0909090909091, 0.0344827586207, 0.179487179487, 0.0909090909091, 0.0344827586207, 0.0344827586207, 0.0344827586207, 0.0344827586207, 0.0909090909091, 0.0344827586207, 0.0967741935484, 0.0344827586207, 0.151515151515, 0.0344827586207, 0.0967741935484, 0.0344827586207, 0.0344827586207, 0.0344827586207, 0.0909090909091, 0.0909090909091, 0.0857142857143, 0.0344827586207, 0.0909090909091, 0.0344827586207, 0.0344827586207, 0.0967741935484, 0.0967741935484, 0.0344827586207, 0.0967741935484, 0.0344827586207, 0.0967741935484, 0.0344827586207, 0.0909090909091, 0.0344827586207, 0.0344827586207]
import numpy as np

def get_result(corpus):
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

	MIT_Result = (1 - (entropy(model_curriculum_MIT, present)))*100
	ETH_Result = (1 - (entropy(model_curriculum_ETH, present)))*100
	UCB_Result = (1 - (entropy(model_curriculum_UCB, present)))*100
	return MIT_Result,ETH_Result,UCB_Result








