import array
import scipy as sp
import numpy as np
from matplotlib import pyplot as plt

def gibbs(sequences, steps):
    samples = []
    startPositionGuesses = np.random.randint(0, sequenceLength-wordLength, numberOfSequences)
    for step in range(steps):
        for i in range(numberOfSequences):
            #Note to self, the profile matrix is just a sum matrix
            profileMatrix, backgroundVector = getProfiles(startPositionGuesses, sequences, i)
            startChances = getStartChances(sequences[i], profileMatrix, backgroundVector)
            #print("startChances", startChances)
            choice = sample_categorical(startChances)
            startPositionGuesses[i]=choice
            #print(choice)
        samples.append(list(startPositionGuesses))
    return samples

    #while(startPositionGuesses!=lastStartPositionGuesses):
    #	lastStartPositionGuesses = startPositionGuesses
    	#No need to profile we have the word given basically
    	#for i in range(len(sequences)):
    	#	print("here")
    	#scores = []
    	#for j in range

#profile, number of letter in position [letter][position]
def getProfiles(startPositions, sequences, ignored=-1):
    profileMatrix = [[0 for x in range(wordLength)] for x in range(alphabetSize)]
    #print(profileMatrix)
    #profileMatrix = [alphabetSize][sequenceLength]
    backgroundVector = [0]*alphabetSize
    for seqNum in range(numberOfSequences):
        if(seqNum==ignored):
            continue
        for i in range(sequenceLength):
            if(i>=startPositions[seqNum] and i<startPositions[seqNum]+wordLength):
                profileMatrix[ sequences[seqNum][i] ][ i-startPositions[seqNum] ]+=1
            else: 
                backgroundVector[sequences[seqNum][i]]+=1    
    return profileMatrix, backgroundVector


def getStartChances(sequence, profileMatrix, backgroundVector):
    chanceVector=[]
    for i in range(sequenceLength-wordLength):
        chanceVector.append(chanceOfPos(i, sequence, profileMatrix, backgroundVector))
    normalizer = sum(chanceVector)
    #print("normalizer", normalizer)
    #print(chanceVector)
    for i in range(len(chanceVector)):
        chanceVector[i]/=normalizer
    return chanceVector

def chanceOfPos(start, sequence, profileMatrix, backgroundVector):
    if(start+wordLength>sequenceLength):
        return 0
    numerator = 1
    denominator = 1
    for i in range(wordLength):
        numerator*=chanceOfLetterInPosition(sequence[start+i], i, profileMatrix)
        denominator*=chanceOfLetterInBackground(sequence[start+i], backgroundVector)
    #print(numerator)
    #print(denominator)
    return numerator/denominator

#chance of (position, letter) 
def chanceOfLetterInPosition(letter, position, profileMatrix):
    return (profileMatrix[letter][position] + alpha[letter])/(numberOfSequences-1 + sum(alpha))


def chanceOfLetterInBackground(letter, backgroundVector): 
    return (backgroundVector[letter] + alphaBG[letter])/(sequenceLength-wordLength + sum(alphaBG))

class MagicWordModel:
    def __init__(self, w, m, k, alpha, alphaBG):
        self.w = w
        self.M = m
        self.K = k
        
        self.theta = np.array([np.random.dirichlet(alpha) for _ in range(self.w)])

        #print(self.theta)

        self.theta_bg = np.random.dirichlet(alphaBG)
        self.facit = []

    def generate_sequence(self):
        sequence = np.zeros(self.M, dtype=int)
        # Sample magic word start position
        r = np.random.randint(0, self.M-self.w)
        facit.append(r)

        for i in range(self.M):
            if r <= i < r+self.w:
                sequence[i] = self.sample_q(i-r)
            else:
                sequence[i] = self.sample_q()

        return sequence

    def sample_q(self, j=None):
        theta = self.theta_bg if j is None else self.theta[j]
        cdf = np.cumsum(theta)
        x = np.random.random()

        for i, _ in enumerate(cdf):
            if x < cdf[i]:
                return i
        else:
            return len(cdf) - 1

def sample_categorical(pmf):
    cdf = np.cumsum(pmf)
    x = np.random.random()

    for i, _ in enumerate(cdf):
        if x < cdf[i]:
            return i
    else:
        print("sample_categorical failed.")
        return len(cdf) - 1

def sample(rawSamples):
    samples = []
    for i in range(burnIn, len(rawSamples), period):
        samples.append(rawSamples[i])
    return samples


def make_plots(samples, facit):
    sequence_samples = np.transpose(samples)
    for n in range(numberOfSequences):
        ax = plt.subplot(510+n+1)

        ind = np.arange(sequenceLength-wordLength)
        colors = ['r' if i == facit[n] else '#348ABD' for i in range(sequenceLength-wordLength)]
        ax.bar(ind, np.bincount(sequence_samples[n], minlength=sequenceLength-wordLength), color=colors)

        plt.legend(loc="upper right")
        #plt.title("Posterior distributions of $p_A$, $p_B$, and delta unknowns")
    plt.show()

def plot_cma(samples):
    """Plots the cumulative moving average for all N sequences."""
    iterations = np.arange(len(samples))

    sequence_samples = np.transpose(samples)

    for _, seq in enumerate(sequence_samples):
        count = len(seq)
        cma = np.zeros(count, dtype=float)
        cma[0] = seq[0]
        for i in range(1, count):
            cma[i] = (seq[i] + (i-1) * cma[i-1]) / i

        plt.plot(iterations, cma, linestyle='-')

    plt.xlabel("Iterations")
    plt.show()


#sequence parameters
alphabetSize = 4
numberOfSequences = 5
sequenceLength = 20
wordLength = 10
alpha = [2, 3, 1, 1]
alphaBG = [1, 1, 1, 1]

#Sampling parameters
steps = 100000
burnIn = 1000
period = 100

if __name__ == '__main__':
    #Randomize word start indices for each sequence
    wordStartIndices = np.random.randint(sequenceLength-wordLength, size=numberOfSequences)
    #print(wordStartIndices)

    #instanciate sequence modeler
    mwm =  MagicWordModel(wordLength, sequenceLength, alphabetSize, alpha, alphaBG)
    facit = mwm.facit

    #generate the sequences
    sequences = []
    for _ in range(numberOfSequences):
    	sequences.append(mwm.generate_sequence())

    #Gibbs
    rawSamples = gibbs(sequences, steps)
    
    samples = sample(rawSamples)

    plot_cma(samples)

    """percentiles = [0 for i in range(4)]
    for i in range(0, 10):
        percentiles[rawSamples[i][0]]+=1/10
    print(percentiles)
    percentiles = [0 for i in range(4)]
    for i in range(10, 100):
        percentiles[rawSamples[i][0]]+=1/90
    print(percentiles)
    percentiles = [0 for i in range(4)]
    for i in range(100, 1000):
        percentiles[rawSamples[i][0]]+=1/900
    print(percentiles)
    percentiles = [0 for i in range(4)]
    for i in range(1000, 10000):
        percentiles[rawSamples[i][0]]+=1/9000
    print(percentiles)
    percentiles = [0 for i in range(4)]
    for i in range(10000, 20000):
        percentiles[rawSamples[i][0]]+=1/10000
    print(percentiles)"""


    make_plots(samples, facit)
    