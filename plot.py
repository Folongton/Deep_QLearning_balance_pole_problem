import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

class Plotting:

    def plot(values, moving_avg_period): 
        plt.figure(2)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(values)
        plt.plot(get_moving_average(moving_avg_period, values))
        plt.show(block=True)

    def get_moving_average(period, values): 
        values = pd.Series(values)
        ma = values.rolling(period).mean()
        ma = ma.fillna(0)
        return ma


Plotting.plot(np.random.rand(300),100)


