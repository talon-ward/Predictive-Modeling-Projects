#
# All code in the file was written by Talon Ward to demonstrate basic predictive
# modeling techniques.
#

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import functools

data_path = 'iris.data'
iris_header = {0 : 'Sepal Length', # Dictionaries for string literals
               1 : 'Sepal Width',  # make the code cleaner.
               2 : 'Petal Length',
               3 : 'Petal Width',
               4 : 'Classification'}
iris_labels = {0 : 'Iris-setosa',
               1 : 'Iris-versicolor',
               2 : 'Iris-virginica'}

def load_data():
    """Loads Iris data from csv file and returns three DataFrame objects."""
    column_names = [iris_header[i] for i in iris_header]
    iris_data = pd.read_csv(data_path, header=None, names=column_names)
    setosa = iris_data.loc[iris_data[iris_header[4]] == iris_labels[0]]
    versic = iris_data.loc[iris_data[iris_header[4]] == iris_labels[1]]
    virgin = iris_data.loc[iris_data[iris_header[4]] == iris_labels[2]]
    return [setosa, versic, virgin]

def visualize_data(setosa, versic, virgin):
    """Creates a visualization of the data to help understand the relationship. The
    figure and axes are returned, so the visualization can be updated to include markers
    at the points that are misclassified."""
    fig, ax = plt.subplots(nrows=4, ncols=4)
    fig.suptitle('Plots of Iris Data: Sepal Length, Sepal Width, Petal Length, Petal Width (Left to Right, Top to Bottom)')
    for i in range(4):
        for j in range(4):
            ax[i][j].scatter(setosa[iris_header[i]], setosa[iris_header[j]], c='red')
            ax[i][j].scatter(versic[iris_header[i]], versic[iris_header[j]], c='blue')
            ax[i][j].scatter(virgin[iris_header[i]], virgin[iris_header[j]], c='green')
    fig.legend((iris_labels[0], iris_labels[1], iris_labels[2]))
    return fig, ax

def calc_param_array(data):
    """Calculates bandwidth parameter for kernel density estimation."""
    return [1.06 * data[iris_header[i]].std() * data[iris_header[i]].count() ** (-0.2) for i in range(4)]

def create_model_1(setosa, versic, virgin):
    """This model assumes that each attribute is drawn from a Gaussian distribution
    where each of the different types of flowers has different parameters (mean and
    standard deviation). These parameters are approximated by using the sample mean
    and standard deviations. These distributions are assumed to be independent. The
    model is then the probability of a species given a particular observation.
    Classification is accomplished by choosing the species with the highest probability.
    This function returns a 6x4 array of predictor means and standard deviations."""
    means_setosa = [setosa[iris_header[i]].mean() for i in range(4)]
    means_versic = [versic[iris_header[i]].mean() for i in range(4)]
    means_virgin = [virgin[iris_header[i]].mean() for i in range(4)]
    stds_setosa = [setosa[iris_header[i]].std() for i in range(4)]
    stds_versic = [versic[iris_header[i]].std() for i in range(4)]
    stds_virgin = [virgin[iris_header[i]].std() for i in range(4)]
    return [means_setosa, means_versic, means_virgin, stds_setosa, stds_versic, stds_virgin]

def create_model_2(setosa, versic, virgin):
    """This model uses kernel density estimation to approximate the predictor distributions.
    The kernel used is the standard normal N(0, 1). The bandwidth parameter is the
    standard rule of thumb 1.06*(std)*n^(1/5). This function returns a 3x4 array of
    the bandwidth parameters corresponding to (flower type) X (predictor)."""
    return [calc_param_array(setosa), calc_param_array(versic), calc_param_array(virgin)]

def iris_probability(label, sepal_length, sepal_width, petal_length, petal_width, model):
    """Calculates the probability of a label given the four observations using the model
    created with create_model_1()."""
    indep = [sepal_length, sepal_width, petal_length, petal_width]
    if label == iris_labels[0]:
        z = [(indep[i] - model[0][i])/model[3][i] for i in range(4)]
    elif label == iris_labels[1]:
        z = [(indep[i] - model[1][i])/model[4][i] for i in range(4)]
    elif label == iris_labels[2]:
        z = [(indep[i] - model[2][i])/model[5][i] for i in range(4)]
    else:
        return -1
    probs = [st.norm.cdf(z[i]) if z[i] < 0 else 1 - st.norm.cdf(z[i]) for i in range(4)]
    return functools.reduce(lambda x, y: x*y, probs)

def iris_classify(sepal_length, sepal_width, petal_length, petal_width, model):
    """Uses the model from create_model_1() to classify an observation. Returns the
    corresponding string from iris_labels"""
    p_setosa = iris_probability(iris_labels[0], sepal_length, sepal_width, petal_length, petal_width, model)
    p_versic = iris_probability(iris_labels[1], sepal_length, sepal_width, petal_length, petal_width, model)
    p_virgin = iris_probability(iris_labels[2], sepal_length, sepal_width, petal_length, petal_width, model)
    if p_setosa > p_versic:
        if p_setosa > p_virgin:
            return iris_labels[0]
        else:
            return iris_labels[2]
    else:
        if p_versic > p_virgin:
            return iris_labels[1]
        else:
            return iris_labels[2]

def kernel_density(x, data, h):
    """Calculates the kernel density estimate of x with K = N(0, 1), where data is
    the values used to derive the estimate, and h is the bandwidth parameter. This
    function is very slow and is the bottleneck for all calculations involving the
    second model."""
    fx = 0
    for xi in data:
        fx = fx + st.norm.pdf((x - xi)/h)
    fx = (fx/data.count())/h
    return fx

def iris_probability_2(label, sepal_length, sepal_width, petal_length, petal_width, setosa, versic, virgin, h):
    """Calculates the probability of a label given the four observations using the model
    created with create_model_2()."""
    indep = [sepal_length, sepal_width, petal_length, petal_width]
    if label == iris_labels[0]:
        p =   kernel_density(sepal_length, setosa[iris_header[0]], h[0][0])
        p = p*kernel_density( sepal_width, setosa[iris_header[1]], h[0][1])
        p = p*kernel_density(petal_length, setosa[iris_header[2]], h[0][2])
        p = p*kernel_density( petal_width, setosa[iris_header[3]], h[0][3])
    elif label == iris_labels[1]:
        p =   kernel_density(sepal_length, versic[iris_header[0]], h[1][0])
        p = p*kernel_density( sepal_width, versic[iris_header[1]], h[1][1])
        p = p*kernel_density(petal_length, versic[iris_header[2]], h[1][2])
        p = p*kernel_density( petal_width, versic[iris_header[3]], h[1][3])
    elif label == iris_labels[2]:
        p =   kernel_density(sepal_length, virgin[iris_header[0]], h[2][0])
        p = p*kernel_density( sepal_width, virgin[iris_header[1]], h[2][1])
        p = p*kernel_density(petal_length, virgin[iris_header[2]], h[2][2])
        p = p*kernel_density( petal_width, virgin[iris_header[3]], h[2][3])
    else:
        return -1
    return p

def iris_classify_2(sepal_length, sepal_width, petal_length, petal_width, setosa, versic, virgin, h):
    """Uses the model from create_model_2() to classify an observation."""
    p_setosa = iris_probability_2(iris_labels[0], sepal_length, sepal_width, petal_length, petal_width, setosa, versic, virgin, h)
    p_versic = iris_probability_2(iris_labels[1], sepal_length, sepal_width, petal_length, petal_width, setosa, versic, virgin, h)
    p_virgin = iris_probability_2(iris_labels[2], sepal_length, sepal_width, petal_length, petal_width, setosa, versic, virgin, h)
    if p_setosa > p_versic:
        if p_setosa > p_virgin:
            return iris_labels[0]
        else:
            return iris_labels[2]
    else:
        if p_versic > p_virgin:
            return iris_labels[1]
        else:
            return iris_labels[2]

def add_cross(data, ax):
    """Places a red 'x' marker on top of misclassified data points in the visualization."""
    for i in range(4):
        for j in range(4):
            ax[i][j].scatter(data[i], data[j], c='red', marker='x')

def sanity_check_1(setosa, versic, virgin, iris_model, fig1, ax1):
    """Tests the first model on the data. Also updates the plot from visualize_data()
    to show misclassified points."""
    for index, row in setosa.iterrows():
        iris_type = iris_classify(row[iris_header[0]],
                                  row[iris_header[1]],
                                  row[iris_header[2]],
                                  row[iris_header[3]], iris_model)
        if iris_type != iris_labels[0]:
            print('Uh oh!', iris_labels[0], 'misclassified as', iris_type)
    for index, row in versic.iterrows():
        iris_type = iris_classify(row[iris_header[0]],
                                  row[iris_header[1]],
                                  row[iris_header[2]],
                                  row[iris_header[3]], iris_model)
        if iris_type != iris_labels[1]:
            print('Uh oh!', iris_labels[1], 'misclassified as', iris_type)
            add_cross(row, ax1)
    for index, row in virgin.iterrows():
        iris_type = iris_classify(row[iris_header[0]],
                                  row[iris_header[1]],
                                  row[iris_header[2]],
                                  row[iris_header[3]], iris_model)
        if iris_type != iris_labels[2]:
            print('Uh oh!', iris_labels[2], 'misclassified as', iris_type)
            add_cross(row, ax1)
    fig1.legend((iris_labels[0], iris_labels[1], iris_labels[2], 'Misclassified'))

def sanity_check_2(setosa, versic, virgin, h, fig1, ax1):
    """Tests the second model on the data. Also updates the plot from visualize_data()
    to show misclassified points."""
    for index, row in setosa.iterrows():
        iris_type = iris_classify_2(row[iris_header[0]],
                                    row[iris_header[1]],
                                    row[iris_header[2]],
                                    row[iris_header[3]], setosa, versic, virgin, h)
        if iris_type != iris_labels[0]:
            print('Oh no!', iris_labels[0], 'misclassified as', iris_type)
    for index, row in versic.iterrows():
        iris_type = iris_classify_2(row[iris_header[0]],
                                    row[iris_header[1]],
                                    row[iris_header[2]],
                                    row[iris_header[3]], setosa, versic, virgin, h)
        if iris_type != iris_labels[1]:
            print('Oh no!', iris_labels[1], 'misclassified as', iris_type)
            add_cross(row, ax1)
    for index, row in virgin.iterrows():
        iris_type = iris_classify_2(row[iris_header[0]],
                                    row[iris_header[1]],
                                    row[iris_header[2]],
                                    row[iris_header[3]], setosa, versic, virgin, h)
        if iris_type != iris_labels[2]:
            print('Oh no!', iris_labels[2], 'misclassified as', iris_type)
            add_cross(row, ax1)
    fig1.legend((iris_labels[0], iris_labels[1], iris_labels[2], 'Misclassified'))

def generate_flower_1(index, model, n_rows):
    """Returns a numpy.Array containing randomly generated data based on the first
    model for a particular species of iris, specified by index."""
    data = np.random.normal(loc=0.0, scale=1.0, size=(n_rows, 4))
    for i in range(4):
        data[:,i] = data[:,i] * model[index+3][i] + model[index][i]
    return data

def generate_similar_1(model):
    """Uses the first model to create a random dataset and then plots another figure
    to visually compare similarities between the actual data and the model."""
    n_rows = 50
    r_setosa = generate_flower_1(0, model, n_rows)
    r_versic = generate_flower_1(1, model, n_rows)
    r_virgin = generate_flower_1(2, model, n_rows)
    fig, ax = plt.subplots(nrows=4, ncols=4)
    fig.suptitle('Plots of Random Data 1: Sepal Length, Sepal Width, Petal Length, Petal Width (Left to Right, Top to Bottom)')
    for i in range(4):
        for j in range(4):
            ax[i][j].scatter(r_setosa[:, i], r_setosa[:, j], c='red')
            ax[i][j].scatter(r_versic[:, i], r_versic[:, j], c='blue')
            ax[i][j].scatter(r_virgin[:, i], r_virgin[:, j], c='green')
    fig.legend((iris_labels[0], iris_labels[1], iris_labels[2]))

def inverse_dist_2(y, x_values, D_values):
    """Calculates the inverse value of the distribution function to generate a random
    variable from the given distribution. Uses a binary search, since distribution
    functions are necessarily increasing. This could be improved with interpolation,
    but it isn't necessary for a basic visualization."""
    a = 0
    b = len(D_values) - 1
    c = (a + b)//2
    while b - a > 1:
        if y < D_values[c]:
            b = c
        else:
            a = c
        c = (a + b)//2
    return x_values[a] if abs(D_values[a]-y) < abs(D_values[b]-y) else x_values[b]

def sample_dist_2(data, h, n_rows, ax):
    """Returns a list of n_rows random variables sampled from the kernel density estimate.
    Can be made more accurate (and slower) by increasing the value of n_points, but
    the distribution functions don't change rapidly."""
    n_points = 300
    x_min = 1.2*data.min() - 0.2*data.max() # add 20 percent of range to either side
    x_max = 1.2*data.max() - 0.2*data.min()
    x_diff = x_max - x_min
    x_values = [x_diff*(i + 0.5)/n_points + x_min for i in range(n_points)] # estimate integral with midpoint
    d_values = [kernel_density(x, data, h) for x in x_values] # density values
    d_sum = sum(d_values) 
    d_values = [d/d_sum for d in d_values] # normalize to sum to 1
    D_values = [sum(d_values[0:i]) for i in range(n_points)] # Distribution values
    r_values = [inverse_dist_2(y, x_values, D_values) for y in np.random.uniform(size=n_rows)]
    ax.plot(x_values, d_values)
    return r_values

def generate_flower_2(data, h, n_rows, ax):
    """Returns a numpy array of n_rows random oberservations generated for a specific
    type of flower using the second model."""
    sep_len = sample_dist_2(data[iris_header[0]], h[0], n_rows, ax[0])
    sep_wid = sample_dist_2(data[iris_header[1]], h[1], n_rows, ax[1])
    pet_len = sample_dist_2(data[iris_header[2]], h[2], n_rows, ax[2])
    pet_wid = sample_dist_2(data[iris_header[3]], h[3], n_rows, ax[3])
    return np.array((sep_len, sep_wid, pet_len, pet_wid))

def generate_similar_2(setosa, versic, virgin, h):
    """Uses the kernel density estimation (second) model to create a random dataset
    and then plots another figure to visually compare with actual data."""
    n_rows = 50
    fig, ax = plt.subplots(nrows=3, ncols=4)
    fig.suptitle('Kernel Density Plots: Rows - Setosa, Versicolor, Virginica, Columns - Sepal Length, Sepal Width, Petal Length, Petal Width')
    r_setosa = generate_flower_2(setosa, h[0], n_rows, ax[0])
    r_versic = generate_flower_2(versic, h[1], n_rows, ax[1])
    r_virgin = generate_flower_2(virgin, h[2], n_rows, ax[2])
    fig, ax = plt.subplots(nrows=4, ncols=4)
    fig.suptitle('Plots of Random Data 2: Sepal Length, Sepal Width, Petal Length, Petal Width (Left to Right, Top to Bottom)')
    for i in range(4):
        for j in range(4):
            ax[i][j].scatter(r_setosa[i, :], r_setosa[j, :], c='red')
            ax[i][j].scatter(r_versic[i, :], r_versic[j, :], c='blue')
            ax[i][j].scatter(r_virgin[i, :], r_virgin[j, :], c='green')
    fig.legend((iris_labels[0], iris_labels[1], iris_labels[2]))    

def main():
    [setosa, versic, virgin] = load_data()
    fig1, ax1 = visualize_data(setosa, versic, virgin)
    fig2, ax2 = visualize_data(setosa, versic, virgin)
    model = create_model_1(setosa, versic, virgin)
    h = create_model_2(setosa, versic, virgin)
    sanity_check_1(setosa, versic, virgin, model, fig1, ax1)
    sanity_check_2(setosa, versic, virgin, h, fig2, ax2)
    generate_similar_1(model)
    generate_similar_2(setosa, versic, virgin, h)

if __name__ == "__main__":
    main()