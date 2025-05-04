import matplotlib.pyplot as plt

def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.title('Training Rewards over Time')
    plt.show()
