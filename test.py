import numpy as np
import matplotlib as mpl
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    data = np.random.multivariate_normal([0, 0, 0], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], size=1000)
    data = pd.DataFrame(data, index=[0,1])
    mpl.rc("figure")
    sns.kdeplot(data, shade=True)
    mpl.pyplot.show()
