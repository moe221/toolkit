import pandas as pd
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt
import numpy as np


def plot_dist_box_qq(df: pd.DataFrame):

    """
    function which takes a dataframes and plots the distribution, boxplot and
    QQ plot for all numerical columns
    """

    # selecting numerical columns
    # make sure all your numerical columns are detected as numerical columns...
    # how ?

    # if necessary, use methods like astype("float") from numpy

    # selecting numerical columns

    X_numerical = df.select_dtypes(exclude=['object'])
    numerical_features = X_numerical.columns

    # Looping over the different numerical features

    for numerical_feature in numerical_features:
        # creating a subplot 1 row x 3 columns
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # the left graph shows the distribution
        ax[0].set_title(f"Distribution of: {numerical_feature}")
        sns.histplot(x=df[numerical_feature], kde=True, ax=ax[0])
        # the graph in the middle shows the boxplot and spot outliers
        ax[1].set_title(f"Boxplot of: {numerical_feature}")
        sns.boxplot(x=df[numerical_feature], ax=ax[1])
        # the graph on the right is the QQ plot which detects the Gaussianity
        ax[2].set_title(f"Gaussianity of: {numerical_feature}")
        qqplot(df[numerical_feature], line='s', ax=ax[2])
        # and it's show time

        plt.show()


def null_values_heatmap(df: pd.DataFrame, size=(15, 10)):

    """
    Function that plots all columns with null values as a heatmap
    params: df = pandas dataframe to plot
            size(OPTIONAL) = Size of heatmap
    """
    plt.figure(figsize=size)
    sns.heatmap(df.isnull(),
            yticklabels=False,
            cbar=False,
            cmap='viridis')
    plt.show()

if __name__ == "__main__":

    X = np.random.normal(loc=0, scale=1, size=1000)
    data = pd.DataFrame(data=X, columns=["values"])
    plot_dist_box_qq(data)
