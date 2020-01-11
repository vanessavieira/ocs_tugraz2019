import numpy as np
import matplotlib.pyplot as plt

def h(l,x):
    arr = []
    for i in range(0,3):
        arr.append(np.arctan2((l[1,i]-x[1]),(l[0,i]-x[0])))
    return arr

def dh(l,x):
    arr = []
    for i in range(0,3):
        dhx1 = (l[1,i]-x[1])/((l[0,i]-x[0])**2 + (l[1,i]-x[1])**2)
        dhx2 = (x[0]-l[0,i])/((l[0,i]-x[0])**2 + (l[1,i]-x[1])**2)
        arr.append([dhx1,dhx2])
    return arr

def obj_f(alpha, l, x):
    return np.linalg.norm(np.subtract(alpha, h(l, x)))**2

def estimate_position(towers, z):
    #initial value (2,2)
    xsi = [np.array([2,2])]
    #lamda value {0.6, 0.9}
    lamda = 0.9
    #H matrix initialization
    H = np.eye(2) * 0.01
    #objective func
    obj_func = []
    for alpha in z:
        #take value from last iteration
        x = xsi[-1]
        h_func = h(towers, x)
        gradh_func = dh(towers, x)
        zi = np.add(np.subtract(alpha, h_func),np.dot(gradh_func,x))
        Ci = np.asarray(gradh_func)
        H = np.add(lamda*H, np.dot(Ci.transpose(),Ci))
        newX = x + np.linalg.pinv(H).dot(Ci.transpose()).dot(np.subtract(zi,np.dot(gradh_func,x)))
        xsi.append(newX)
        obj_func.append(obj_f(alpha, towers, x))
    x = np.asarray(xsi)[:, 0]
    l1 = np.asarray(towers)[0, :]
    l2 = np.asarray(towers)[1, :]
    y = np.asarray(xsi)[:, 1]
    plt.scatter(x, y)
    plt.scatter(l1, l2)
    plt.show()
    print("Estimated position:")
    print(xsi[-1])
    return xsi[-1]

def g(z,l, x):
    temp = []
    for i in range(0,3):
        temp.append(np.linalg.norm(np.subtract(l[:,i], x)))
    return np.subtract(z, temp)

def dg(l,t,x0, v):
    res = []
    for i in range(0, 3):
        li = l[:,i]
        tv = t*np.asarray(v)
        norm = (np.linalg.norm(li - x0 - tv))
        dgi = []
        #dgdx
        for j in range(0,2):
            lx = l[j,i] - x0[j] - t*v[j]
            dgi.append(lx/norm)
        #dgdv
        for j in range(0, 2):
            lx = l[j, i] - x0[j] - t * v[j]
            dgi.append(t*lx / norm)
        res.append(dgi)
    return res

def estimate_motion(towers, z, x0):
    #position
    xs = [[x0[0], x0[1], 1,1]]
    position = []
    lamda = 0.9
    # H matrix initialization
    H = np.eye(4)*0.01
    obj_func = []
    for t, alpha in enumerate(z):
        xsi = xs[-1]
        x = [xsi[0] + t*xsi[2], xsi[1] + t*xsi[3]]
        position.append(x)
        func = g(alpha, towers, x)
        grad = dg(towers, t, [xsi[0], xsi[1]], [xsi[2], xsi[3]])
        #zi = g + gd*v
        zi = np.subtract(func, np.dot(grad, xsi))
        Ci = (-1)*np.asarray(grad)
        H = np.add(lamda * H, np.dot(Ci.transpose(), Ci))
        newXsi = xsi + np.linalg.pinv(H).dot(Ci.transpose()).dot(np.subtract(zi, np.dot(Ci, xsi)))
        xs.append(newXsi)
        obj_func.append(func)
    x = np.asarray(position)[:,0]
    l1 = np.asarray(towers)[0,:]
    l2 = np.asarray(towers)[1, :]
    y = np.asarray(position)[:,1]
    plt.scatter(x,y)
    plt.scatter(l1,l2)
    plt.show()
    print("Final position:")
    print(position[-1])
    return position[-1]

if __name__ == '__main__':
    # load the data
    data = np.load('./data_position.npz')
    # towers
    towers = data['towers']
    # measurements
    z = data['z'] * np.pi/180

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