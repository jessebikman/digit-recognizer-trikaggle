#!/usr/bin/python2.6

from sklearn.cluster import KMeans
from numpy import genfromtxt, savetxt, transpose
import time

def main():

    # For timing
    start = time.clock()

    # Create the training set; skip the header row with [1:] (64 bit ints)
    dataset = genfromtxt(open('data/train.csv','r'), delimiter = ',', dtype = 'int64')[1:]
    # Pulls out classifer labels
    target = [x[0] for x in dataset]
    # Pull out training data
    train = [x[1:] for x in dataset]
    # Create the test set
    test = genfromtxt(open('data/test.csv','r'), delimiter = ',', dtype = 'int64')[1:]

    # Using K-means as classifier. 

    km = KMeans(n_clusters = 10, init = 'k-means++', n_jobs = 2, precompute_distances = True)
    km.fit(train, target[0:])
    predicted = [km.predict(test)]

    # Generate the submission csv by adding commas to the probability samples
    savetxt('data/submission.csv', transpose(predicted), delimiter = ',', fmt = '%i')
    # Print time to run
    end = time.clock()
    print "Time to run was %s seconds" % (end - start)


if __name__ == "__main__":
    main()
