from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
import time

def main():

    # For timing
    start = time.clock()

    # Create the training set; skip the header row with [1:] (64 bit floats)
    dataset = genfromtxt(open('data/train.csv','r'), delimiter = ',', dtype = 'f8')[1:]    
    # Pulls out classifer labels
    target = [x[0] for x in dataset]
    # Pull out training data
    train = [x[1:] for x in dataset]
    # Create the test set
    test = genfromtxt(open('data/test.csv','r'), delimiter = ',', dtype = 'f8')[1:]

    # Create and train the random forest classifier
    # - Use n_estimators to specify the number of trees in the random forest
    # - Use n_jobs to assign workers to different CPU cores
    rf = RandomForestClassifier(n_jobs = 4, n_estimators = 500)
    # Build a forest of trees from the training set with target 
    # values that correspond to classification classes
    rf.fit(train, target[0:])

    # Extract probability samples from built random forest classifer applied to test set

    # In this case, the probability samples represents output from a likelihood function
    # for a Bernoulli random distribution
    predicted_probs = [x[1] for x in rf.predict_proba(test)]

    # Generate the submission csv by adding commas to the probability samples
    savetxt('data/submission.csv', predicted_probs, delimiter = ',', fmt = '%f')

    # Print time to run
    end = time.clock()
    print "Time to run was %s seconds" % (end - start)


if __name__ == "__main__":
    main()
