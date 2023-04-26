import matplotlib.pyplot as plt

def plot_mse(models, mse_values):
    plt.bar(models, mse_values)
    plt.xlabel('Models')
    plt.ylabel('Mean Squared Error')
    plt.title('Model Comparison - MSE')
    plt.show()

def plot_r2(models, r2_values):
    plt.bar(models, r2_values)
    plt.xlabel('Models')
    plt.ylabel('R-squared')
    plt.title('Model Comparison - R2')
    plt.show()

