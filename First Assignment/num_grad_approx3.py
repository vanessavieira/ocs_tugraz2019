
def f(x, y):
    return (2 * x ** 2) + (3 * y ** 2) - (2 * x * y) + (2 * x) - (3 * y)


def grad_x(x,y):
    return 4*x - 2*y + 2


def grad_y(x,y):
    return 6*y - 2*x - 3


def grad_approx_x(x,y,eps=0.001):
    return (f(x+eps,y)-f(x-eps,y))/(2*eps)


def grad_approx_y(x,y,eps=0.001):
    return (f(x,y+eps)-f(x,y-eps))/(2*eps)


x = [1, 2, 3]

y = [1, 2, 3]

for i in range(0,3):
    out_grad_x = grad_x(x[i],y[i])
    out_grad_y = grad_y(x[i],y[i])
    out_grad_approx_x = grad_approx_x(x[i],y[i])
    out_grad_approx_y = grad_approx_y(x[i],y[i])

    print("Result for point (" + str(x[i]) + "," + str(y[i]) + ") is")
    print("Gradient analytically computed: \ngrad_x = "+ str(out_grad_x)
          + ", " + "grad_y = " + str(out_grad_y))
    print("Numerical approximation of gradient: \ngrad_x = "
          + str(out_grad_approx_x) + ", " + "grad_y = " + str(out_grad_approx_y) + "\n")