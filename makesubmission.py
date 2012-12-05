from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt

def main():
    # Create the training set; skip the header row with [1:] (64 bit floats)
    dataset = genfromtxt(open('data/train_first_785.csv','r'), delimiter = ',', dtype = 'f8')[1:]    
    # Pulls out classifer labels
    target = [x[0] for x in dataset]
    # Pull out training data
    train = [x[1:785] for x in dataset]
    # Create the test set
    test = genfromtxt(open('data/test_first_785.csv','r'), delimiter = ',', dtype = 'f8')[1:]

    # Create and train the random forest classifier
    # - Use n_estimators to specify the number of trees in the random forest
    # - Use n_jobs to assign workers to different CPU cores
    rf = RandomForestClassifier(n_jobs = 2)
    # Build a forest of trees from the training set with target 
    # values that correspond to classification classes
    rf.fit(train, target[0:784])

    # Extract probability samples from built random forest classifer applied to test set

    # In this case, the probability samples represents output from a likelihood function
    # for a Bernoulli random distribution
    predicted_probs = [x[1] for x in rf.predict_proba(test)]

    # Generate the submission csv by adding commas to the probability samples
    savetxt('Data/submission.csv', predicted_probs, delimiter = ',', fmt = '%f')



if __name__ == "__main__":
    main()
