import numpy as np
import matplotlib.pyplot as plt


def h(l, x):
    arr = []
    for i in range(0, 3):
        arr.append(np.arctan2((l[1, i] - x[1]), (l[0, i] - x[0])))
    return arr


def obj_f(alpha, l, x):
    return np.linalg.norm(np.subtract(alpha, h(l, x))) ** 2


def dh(l, x):
    arr = []
    for i in range(0, 3):
        dhx1 = (l[1, i] - x[1]) / ((l[0, i] - x[0]) ** 2 + (l[1, i] - x[1]) ** 2)
        dhx2 = (x[0] - l[0, i]) / ((l[0, i] - x[0]) ** 2 + (l[1, i] - x[1]) ** 2)
        arr.append([dhx1, dhx2])
    return arr


def estimate_position(towers, z):
    # initial value (2,2)
    # lamda value {0.6, 0.9}
    lamda = 0.9
    xsi = [np.array([2, 2])]

    # H matrix initialization
    H = np.eye(2) * 0.01
    # objective func
    obj_func = []

    for alpha in z:
        # take value from last iteration
        x = xsi[-1]
        h_func = h(towers, x)
        gradh_func = dh(towers, x)
        zi = np.add(np.subtract(alpha, h_func), np.dot(gradh_func, x))
        Ci = np.asarray(gradh_func)
        H = np.add(lamda * H, np.dot(Ci.transpose(), Ci))
        newX = x + np.linalg.pinv(H).dot(Ci.transpose()).dot(np.subtract(zi, np.dot(gradh_func, x)))
        xsi.append(newX)
        obj_func.append(obj_f(alpha, towers, x))

    def plot_estimate_position():
        plt.title('Estimating position for lambda = ' + str(lamda))
        plt.plot(xsi[0][0], xsi[0][1], "*", color="green", markersize=15, label=r'initial position')

        x = np.asarray(xsi)[:, 0]
        l1 = np.asarray(towers)[0, :]
        l2 = np.asarray(towers)[1, :]
        y = np.asarray(xsi)[:, 1]

        # generating 2D grid
        x_1 = np.linspace(-25, 20, 20)
        x_2 = np.linspace(-25, 20, 20)
        X_1, X_2 = np.meshgrid(x_1, x_2)
        XY = np.stack([X_1.flatten(), X_2.flatten()], axis=0)

        def func(xy):
            f_all = []
            for i in range(xy.shape[1]):
                f = 0
                for zii in z:
                    gi = (np.linalg.norm(np.subtract(zii, h(towers, xy[:, i]))) ** 2) / 2
                    f = gi + lamda * f
                f_all.append(f)
            return f_all

        objective = np.asarray(func(XY)).reshape(X_1.shape)
        CS = plt.contour(X_1, X_2, objective)

        plt.scatter(x, y, label=r'sequential position')
        plt.scatter(l1, l2, label=r'light tower')
        plt.plot(xsi[-1][0], xsi[-1][1], "*", color="red", markersize=15, label=r'optimal position')
        plt.legend(loc="lower right")
        plt.show()

    plot_estimate_position()

    print("Estimated position:")
    print(xsi[-1])
    return xsi[-1]


def g(z, l, x):
    temp = []
    for i in range(0, 3):
        temp.append(np.linalg.norm(np.subtract(l[:, i], x)))
    return np.subtract(z, temp)


def dg(l, t, x0, v):
    res = []
    for i in range(0, 3):
        li = l[:, i]
        tv = t * np.asarray(v)
        norm = (np.linalg.norm(li - x0 - tv))
        dgi = []
        # dgdx
        for j in range(0, 2):
            lx = l[j, i] - x0[j] - t * v[j]
            dgi.append(lx / norm)
        # dgdv
        for j in range(0, 2):
            lx = l[j, i] - x0[j] - t * v[j]
            dgi.append(t * lx / norm)
        res.append(dgi)
    return res


def estimate_motion(towers, z, x0):
    # state vector
    xs = [[x0[0], x0[1], 0, 0]]
    position = []
    coursesx = []
    coursesy = []
    t_80 = []
    t_150 = []
    lamda = 0.9
    t_final = 0
    # H matrix initialization
    H = np.eye(4) * 0.01
    for t, alpha in enumerate(z):
        xsi = xs[-1]
        x = [xsi[0] + t * xsi[2], xsi[1] + t * xsi[3]]
        func = g(alpha, towers, x)
        grad = dg(towers, t, [xsi[0], xsi[1]], [xsi[2], xsi[3]])
        # zi = g + gd*x
        zi = np.subtract(func, np.dot(grad, xsi))
        Ci = (-1) * np.asarray(grad)
        H = np.add(lamda * H, np.dot(Ci.transpose(), Ci))
        newXsi = xsi + np.linalg.pinv(H).dot(Ci.transpose()).dot(np.subtract(zi, np.dot(Ci, xsi)))
        xs.append(newXsi)
        x = [newXsi[0] + t * newXsi[2], newXsi[1] + t * newXsi[3]]
        position.append(x)
        coursesx.append(newXsi[2])
        coursesy.append(newXsi[3])

        if (t < 80):
            t_80.append(x)
            t_150.append(x)
        if (80 <= t < 150):
            t_150.append(x)

    def plot_estimate_motion(positions, t):
        plt.title('Estimating motion for lambda = ' + str(lamda) + ' with t = ' + str(t))
        plt.plot(xs[0][0], xs[0][1], "*", color="green", markersize=15, label=r'initial position')

        x = np.asarray(positions)[:, 0]
        l1 = np.asarray(towers)[0, :]
        l2 = np.asarray(towers)[1, :]
        y = np.asarray(positions)[:, 1]

        x_1 = np.linspace(-10, 25, 50)
        x_2 = np.linspace(-25, 10, 50)
        X_1, X_2 = np.meshgrid(x_1, x_2)
        XY = np.stack([X_1.flatten(), X_2.flatten()], axis=0)

        def func(xy):
            f_all = []
            for i in range(xy.shape[1]):
                f = 0
                for k in range(0,t):
                    gi = (np.linalg.norm(g(z[k], towers, xy[:, i])) ** 2) / 2
                    f = (gi + lamda * f)
                f_all.append(f)
            return f_all

        objective = np.asarray(func(XY)).reshape(X_1.shape)
        plt.contour(X_1, X_2, objective, 13)

        plt.scatter(x, y, label=r'sequential position')
        plt.scatter(l1, l2, label=r'light tower')
        plt.plot(positions[-1][0], positions[-1][1], "*", color="red", markersize=15, label=r'optimal position')
        plt.legend(loc="lower right")
        plt.show()

    plot_estimate_motion(t_80, 80)
    plot_estimate_motion(t_150, 150)
    plot_estimate_motion(position, 200)

    # plot courses
    plt.title('Courses with lambda = ' + str(lamda))
    plt.scatter(coursesx, coursesy)
    plt.show()

    print("Final position:")
    print(position[-1])
    print("Final course:")
    print("[" + str(coursesx[-1]) + "," + str(coursesy[-1]) + "]")
    return position[-1]


if __name__ == '__main__':
    # load the data
    data = np.load('./data_position.npz')
    # towers
    towers = data['towers']
    # measurements
    z = data['z'] * np.pi / 180

    print('Towers:', towers.shape)
    print('Measurements:', z.shape)

    x0 = estimate_position(towers, z)

    # load the data
    data = np.load('./data_motion.npz')
    # towers
    towers = data['towers']
    # measurements
    z = data['z']

    print('Towers:', towers.shape)
    print('Measurements:', z.shape)

    x = estimate_motion(towers, z, x0)