import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 2*x + 1

def f_gradient(x):
    return 2*x + 2

def gradient_descent(learning_rate=0.1, max_iterations=100, tolerance=1e-6):
    x = 10
    loss_history = [f(x)]
    gradient_history = [np.abs(f_gradient(x))]

    for i in range(max_iterations):
        grad = f_gradient(x)
        x_new = x - learning_rate * grad

        loss_history.append(f(x_new))
        gradient_history.append(np.abs(f_gradient(x_new)))

        if abs(x_new - x) < tolerance:
            print(f"Converged after {i+1} iterations!")
            break

        x = x_new

    return loss_history, gradient_history

loss_history, gradient_history = gradient_descent()
iterations = range(len(loss_history))

plt.figure(figsize=(10, 6))

plt.plot(iterations, loss_history, label="Loss (f(x))", color="blue", linewidth=2)

plt.twinx()
plt.plot(iterations, gradient_history, label="Gradient Norm", color="orange", linestyle="--", linewidth=2)

plt.title("Convergence of Loss and Gradient Norm", fontsize=14)
plt.xlabel("Iterations", fontsize=12)
plt.ylabel("Loss (blue) / Gradient Norm (orange)", fontsize=12)
plt.legend(loc="upper right")
plt.grid(True)
plt.show()

