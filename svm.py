
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Convert numeric label into string

label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Display training data

def train(n):
    plt.imshow(train_data[n])
    plt.show()

    print(label[(train_labels[n])])

# Display test data

def test(n):
    plt.imshow(test_data[n])
    plt.show()

    print(label[(test_labels[n])])

# Load byte data from file

def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
    return data

# Return train_data, train_labels, test_data, test_labels

def load_cifar10_data(data_dir):
    train_data = None
    train_labels = []

    for i in range(1, 6):
        data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            train_data = data_dic['data']
        else:
            train_data = np.vstack((train_data, data_dic['data']))
        train_labels += data_dic['labels']

    test_data_dic = unpickle(data_dir + "/test_batch")
    test_data = test_data_dic['data']
    test_labels = test_data_dic['labels']

    train_data = train_data.reshape((len(train_data), 3, 32, 32))
    train_data = np.rollaxis(train_data, 1, 4)
    train_labels = np.array(train_labels)

    test_data = test_data.reshape((len(test_data), 3, 32, 32))
    test_data = np.rollaxis(test_data, 1, 4)
    test_labels = np.array(test_labels)

    return train_data, train_labels, test_data, test_labels

data_dir = 'cifar-10-batches-py'
train_data, train_labels, test_data, test_labels = load_cifar10_data(data_dir)

--------------------------------------------------------------------------------

c = train_data.reshape(50000, 32*32*3)
X_train = (c - np.mean(c, axis=0))/(np.std(c, axis=0))
y_train = train_labels

c = test_data.reshape(10000, 32*32*3)
X_test = (c - np.mean(c, axis=0))/(np.std(c, axis=0))
y_test = test_labels

weights = np.random.normal(0, 1/np.sqrt(32*32*3), (32*32*3,10))
biases = np.random.normal(0, 1, (10))
lam = 0.05

for i in range(100):
    a = 0.001/(1+i)
    for j in range(50000):
        loc = np.random.randint(50000)
        output = np.dot(X_train[loc], weights) + biases

        weights -= a*lam*weights

        y = y_train[loc]
        for c in range(10):
            if((output[c] > output[y] - 1) & (c!=y)):
                weights[:,y] += a*X_train[loc]
                weights[:,c] -= a*X_train[loc]
 
                biases[y] += a
                biases[c] -= a

    output = np.dot(X_test, weights) + biases
    activation = np.argmax(output, axis=1)
    print(np.sum(y_test == activation))







