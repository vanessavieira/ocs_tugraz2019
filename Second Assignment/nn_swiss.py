import numpy as np
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def softplus(z):
    a = np.zeros(len(z))
    for i in range(0,len(z)):
        a[i] = np.log(1+np.exp(z[i]))
    return a

def d_softplus(z):
    a = np.zeros((len(z), len(z)))
    for i in range(0, len(z)):
        a[i][i] = np.exp(z[i])/(1+np.exp(z[i]))
    return a

def softmax(z):
    sum = 0
    for j in range(0,len(z)):
        sum += np.exp(z[j])
    a = np.zeros(len(z))
    for i in range (0,len(z)):
        a[i] = np.exp(z[i])/sum
    return a

def d_softmax(z):
    s = softmax(z)
    m = np.diag(s)
    for i in range(len(m)):
        for j in range(len(m)):
            if i == j:
                m[i][j] = s[i] * (1 - s[i])
            else:
                m[i][j] = -s[i] * s[j]
    return m

def feed_forward(x, W, b):
    # convert to column vector
    a = [x]
    z = [x]
    #first layer
    z.append(W[0]@a[0] + b[0])
    a.append(softplus(z[1]))
    #second layer
    z.append(W[1]@a[1] + b[1])
    a.append(softmax(z[2]))

    return a, z

def individual_loss(y, y_tilde):
   # assert y_tilde.shape == y.shape
    loss = 0
    for i in range(0, 4):
        loss += y[i] * np.log(y_tilde[i])
    return -loss

def init_params(N_H=4):
    # initialize parameters
    mu, sigma = 0, 0.05  # mean and standard deviation
    W = [np.random.normal(mu, sigma, size = (N_H,3)),  # W0
         np.random.normal(mu, sigma, size = (4,N_H))]  # W1
    b = [np.random.normal(mu, sigma, size = N_H),     # b0
         np.random.normal(mu, sigma, size = 4)]     # b1
    return W, b

def d_individual_loss(y,y_tilde):
    loss = np.zeros(4)
    for i in range(0, 4):
        loss[i] = y[i]/y_tilde[i]
    return -loss

def back_prop(y, W, a, z):
    # compute errors e in reversed order
    assert (len(a) == len(z))
    e2 = d_softmax(z[2]) @ d_individual_loss(y, a[-1])
    e1 = d_softplus(z[1]) @ (W[1].T @ e2)
    # compute gradient for W an b
    dW = [None] * len(a)
    db = [None] * len(a)
    dW[0] = np.outer(e1, a[0])
    dW[1] = np.outer(e2, a[1])
    db[0] = e1
    db[1] = e2
    return dW, db


def plotting3d(x, y, label, S):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for s in range(S):
        x_1 = x[s, 0]
        x_2 = x[s, 1]
        x_3 = x[s, 2]

        # true labels
        if y[s] == 0:
            ax.scatter(x_1, x_2, x_3, c='blue', marker='o', s=30)

        if y[s] == 1:
            ax.scatter(x_1, x_2, x_3, c='black', marker='v', s=25)

        if y[s] == 2:
            ax.scatter(x_1, x_2, x_3, c='red', marker='D', s=25)

        if y[s] == 3:
            ax.scatter(x_1, x_2, x_3, c='green', marker='+', s=30)

        # predicted labels
        if label[s] == 0:
            ax.scatter(x_1, x_2, x_3, c='magenta', marker='o', s=5)

        if label[s] == 1:
            ax.scatter(x_1, x_2, x_3, c='yellow', marker='v', s=5)

        if label[s] == 2:
            ax.scatter(x_1, x_2, x_3, c='orange', marker='D', s=5)

        if label[s] == 3:
            ax.scatter(x_1, x_2, x_3, c='cyan', marker='+', s=5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.legend()
    #plt.legend()
    plt.show()


if __name__ == '__main__':
    # load the data set
    data_set = np.load('./data.npz')

    print(data_set.files)
    # get the training data
    x_train = data_set['x_train']
    y_train = data_set['y_train']
    # get the test data
    x_test = data_set['x_test']
    y_test = data_set['y_test']

    print('Training data:', x_train.shape, y_train.shape)
    print('Test data:', x_test.shape, y_test.shape)

    # choose algorithm parameters
    #step size
    alpha = 0.05
    max_it = 1000
    with_armijo = True
    N_H = [512] #4, 32 or 512

    for j in N_H:
        W, b = init_params(j)

        L = np.zeros(max_it + 1)
        new_L = np.zeros(max_it + 1)
        A = np.zeros(max_it + 1)
        S = x_train.shape[0]
        label_train = np.zeros(S)

        L_ = 0
        A_ = 0
        S_ = x_test.shape[0]
        label_test = np.zeros(S_)

    ## Training nn
        print ("\nTraining data:")
        for it in range(max_it + 1):
            # iterate over all samples and sum up the gradients
            dW = [0, 0]
            db = [0, 0]
            count = 0
            for xi, yi in zip(x_train, y_train):
                y = np.zeros(4)
                y[int(yi)] = 1
                ai, zi = feed_forward(xi, W, b)
                dWi, dbi = back_prop(y, W, ai, zi)
                L[it] += individual_loss(y, ai[-1]) / S

                correct = np.argmax(ai[-1])
                if int(yi) == correct:
                    A[it] += 1 / S

                label_train[count] = correct

                # sum up gradients
                for idx in range(2):
                    dW[idx] += (dWi[idx] / S)
                    db[idx] += (dbi[idx] / S)

                count = count + 1

            if not with_armijo:
                for idx in range(2):
                    W[idx] = W[idx] - alpha * dW[idx]
                    b[idx] = b[idx] - alpha * db[idx]
            if with_armijo:
                alpha = 1
                g = np.sum(dW[0] ** 2) + np.sum(dW[1]** 2) + np.sum(db[0] ** 2) + np.sum(db[1] ** 2)
                c = 0.01
                for bt in range(10):
                    W_ = [0,0]
                    b_ = [0,0]
                    for idy in range(2):
                        W_[idy] = W[idy] - alpha * dW[idy]
                        b_[idy] = b[idy] - alpha * db[idy]
                    loss = 0
                    loss_ = 0
                    for xj, yj in zip(x_train, y_train):
                        y = np.zeros(4)
                        y[int(yj)] = 1
                        aj, zj = feed_forward(xj, W, b)
                        a_, z_ = feed_forward(xj, W_, b_)
                        loss += individual_loss(y, aj[-1]) / S
                        loss_ += individual_loss(y, a_[-1]) / S
                    if (loss - loss_) >= (c * alpha * g):
                        W = W_
                        b = b_
                        break

                    else:
                        alpha = alpha * 0.5

            if it % 50 == 0:
                print("Iteration: " + str(it))
                print("Accuracy: " + str(A[it]))

        if (with_armijo):
            plt.title('Total Loss for N_H = ' + str(j) + ', using armijo')
            plt.xlabel('Iterations')
            plt.ylabel('Total Loss')
            plt.plot(L)
            plt.show()
            plt.title('Accuracy for N_H = ' + str(j) + ', using armijo')
            plt.xlabel('Iterations')
            plt.ylabel('Accuracy')
            plt.plot(A)
            plt.show()
            plotting3d(x_train, y_train, label_train, S)


        else:
            plt.title('Total Loss for N_H = ' + str(j) + ', not using armijo')
            plt.xlabel('Iterations')
            plt.ylabel('Total Loss')
            plt.plot(L)
            plt.show()
            plt.title('Accuracy for N_H = ' + str(j) + ', not using armijo')
            plt.xlabel('Iterations')
            plt.ylabel('Accuracy')
            plt.plot(A)
            plt.show()

            plotting3d(x_train, y_train, label_train, S)

    ## Testing trained nn
        print ("\nTesting data:")
        count_ = 0
        for xi, yi in zip(x_test, y_test):
            y = np.zeros(4)
            y[int(yi)] = 1
            ai,zi = feed_forward(xi,W,b)
            L_ += individual_loss(y, ai[-1]) / S_

            correct = np.argmax(ai[-1])
            if int(yi) == correct:
                A_ += 1 / S_

            label_test[count_] = correct
            count_ = count_ + 1

        print("Accuracy: " + str(A_))
        print("Total Loss: " + str(L_))

        plotting3d(x_test,y_test,label_test, S_)

