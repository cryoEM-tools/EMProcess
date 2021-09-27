import numpy as np

def sample_gaussian(xs, A, B, sigma):
    ys = A/(sigma*np.sqrt(2*np.pi)) * np.exp((-0.5*((xs-B)/sigma)**2))
    return ys

def smooth_probabilities(
        data, populations=None, x_range=None, n_points=1000, sigma=None):
    """projects populations onto an order parameter
    
    Inputs
    ----------
    data : nd.array, shape=(n_points, )
        The value of the order parameter for each data point.
    populations : nd.array, shape=(n_states, )
        The population of each data point.
    x_range : array, shape=(2, ), default=None,
        The x-axis plot range. i.e. [1, 5].
    n_points : int, default=1000,
        Number of points to use for plotting data.
    sigma : float, default=None,
        The width to use for each gaussian. If none is supplied, defaults to
        1/20th of the `x_range`.
    
    Outputs
    ----------
    xs : nd.array, shape=(n_points, ),
        The x-axis values of the resultant projection.
    ys : nd.array, shape=(n_points, )
        The y-axis values of the resultant projection.
    """
    data_spread = data.max() - data.min()
    if populations is None:
        populations = np.ones(data.shape[0])/data.shape[0]
    if x_range is None:
        delta_0p1 = data_spread*0.1
        x_range = [data.min()-delta_0p1, data.max()+delta_0p1]
    if sigma is None:
        sigma = data_spread/20.
    range_spread = x_range[1] - x_range[0]
    xs = range_spread*(np.arange(n_points)/n_points) + x_range[0]
    ys = np.zeros(xs.shape[0])
    for n in np.arange(data.shape[0]):
        ys += sample_gaussian(xs, populations[n], data[n], sigma=sigma)
    return xs, ys
