
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

import numpy as np
from scipy import spatial

X_train = train_data.reshape(50000, 32*32*3)/256
y_train = train_labels

X_test = test_data.reshape(10000, 32*32*3)/256
y_test = test_labels

# distances between X_train values and X_test values
dist = spatial.distance.cdist(X_train, X_test)

# sorts distances by argument
dist_sort = dist.argsort(axis=0)


k = np.arange(1,100)

for i in range(k.shape[0]):

    # replaces arguments with corresponding label
    labs = y_train[dist_sort[0:k[i]]]

    # selects most frequent label from each column
    pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=labs)

    # print accuracy
    print(k[i], np.sum(pred == test_labels))









