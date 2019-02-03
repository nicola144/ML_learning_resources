import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import time
import IPython

# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# rc('text', usetex=True)

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def process_plot(fig, options=dict()):
    if 'x_label' in options.keys():
        fig.set_xlabel(options['x_label'])
    if 'y_label' in options.keys():
        fig.set_ylabel(options['y_label'])
    if 'x_lim' in options.keys():
        fig.set_ylim(options['x_lim'])
    if 'y_lim' in options.keys():
        fig.set_ylim(options['y_lim'])
    if 'title' in options.keys():
        fig.set_title(options['title'])
    if 'legend' in options.keys():
        if options['legend']:
            fig.legend(loc=options.get('legend_loc', 'best'))


def plot_data(X, Y, fig=None, options=dict()):
    # fig_data = plt.figure()
    if fig is None:
        fig = plt.subplot(111)
    fig.plot(X, Y, options.get('marker', 'b*'), 
        label=options.get('label', 'Raw data'),
        fillstyle=options.get('fillstyle', 'full'),
        ms=options.get('size', 8))
    process_plot(fig, options)


def plot_fit(X, w, fig=None, options=dict()):
    if fig is None:
        fig = plt.subplot(111)

    x_min = np.min(X[:, -2])
    x_max = np.max(X[:, -2])
    dim = w.size - 1
    x_plot = np.reshape(np.linspace(x_min, x_max, 100), [-1, 1])
    x1_plot = np.ones_like(x_plot)
    for d in range(dim):
        x1_plot = np.concatenate((np.power(x_plot, 1 + d), x1_plot), axis=1)

    y_plot = np.dot(x1_plot, w)
    fig.plot(x_plot, y_plot, 'r-', label=options.get('label', 'Regression fit'))
    process_plot(fig, options)


def plot_contour(X, Y, w_trajectory, func, fig=None, options=dict()):
    if fig is None:
        fig = plt.subplot(111)

    w_min = np.min(w_trajectory, axis=0)
    w_max = np.max(w_trajectory, axis=0)
    w_range = np.abs(w_max - w_min)
    w_range[w_range == 0] = 3
    w_end = w_trajectory[-1, :]

    [xg, yg] = np.meshgrid(np.linspace(w_end[0] - 2 * w_range[0], w_end[0] + 2 * w_range[0], 100),
                           np.linspace(w_end[1] - 2 * w_range[1], w_end[1] + 2 * w_range[1], 100))
    wg = np.concatenate((xg.reshape([-1, 1]), yg.reshape([-1, 1])), axis=1)
    zg = np.zeros((wg.shape[0], 1))
    j = 0
    for wj in wg:
        zg[j] = func(wj, X, Y)
        j += 1
    zg = np.reshape(zg, newshape=xg.shape)
    fig.contour(xg, yg, zg)

    process_plot(fig, options)


def plot_arrow(w_old, w_new, fig=None, options=dict()):
    if fig is None:
        fig = plt.subplot(111)
    length = w_new - w_old
    arrow = fig.arrow(w_old[0], w_old[1], length[0], length[1],
                      width=max(0.01, min(0.1, np.sqrt(length[0] ** 2 + length[1] ** 2))), length_includes_head=False,
                      fc='r')
    return arrow


def plot_trajectory(w_trajectory, fig=None, options=dict()):
    if fig is None:
        fig = plt.subplot(111)
    traj_plot, = fig.plot(w_trajectory[:, 0], w_trajectory[:, 1], 'r.-', alpha=0.5)
    return traj_plot


def linear_regression_progression(X, Y, w_trajectory, index_trajectory, func, contourplot=None, dataplot=None, options=dict()):
    # Plot Contour
    if contourplot is not None:
        contour_opts = options.get('contour_opts', dict())
        plot_contour(X, Y, w_trajectory, func, fig=contourplot, options=contour_opts)

    # Plot raw Data
    if dataplot is not None:
        data_opts = options.get('data_opts', dict())
        plot_data(X[:, -2], Y, fig=dataplot, options=data_opts)

    for idx in range(len(w_trajectory) - 1):
        if 'traj_plot' in locals():
            traj_plot.remove()  # Remove previous trajectory.
        if 'arrow' in locals():
            arrow.remove()  # Remove previous arrow.
        if dataplot is not None:
            while len(dataplot.lines) > 1:
                dataplot.lines.pop(-1)  # Remove previous line.

        if contourplot is not None:
            # Plot gradient arrow.
            contour_opts = options.get('contour_opts', dict())
            arrow = plot_arrow(w_trajectory[idx, :], w_trajectory[idx + 1, :], fig=contourplot, options=contour_opts)

            # Plot weight trajectory.
            traj_plot = plot_trajectory(w_trajectory[:(idx + 1), :], fig=contourplot, options=contour_opts)

        if dataplot is not None:
            # Plot best fit line. 
            data_opts = options.get('data_opts', dict())
            x_idx = index_trajectory[idx]
            #if x_idx.size == 1:
            if data_opts.get('sgd_point', False):
                opt = {'marker': 'mX', 'label': 'Current SGD point', 'size': 15}
                plot_data(X[x_idx, -2], Y[x_idx], fig=dataplot, options=opt)

            plot_fit(X, w_trajectory[idx, :], fig=dataplot, options=data_opts)

        IPython.display.clear_output(wait=True)
        IPython.display.display(plt.gcf())

        # Wait for human visualization. 
        time.sleep(0.5)
        # input("Press Enter to continue...")

    plt.close()


def kernelized_regression_progression(X, Xtr, Ytr, alpha_trayectory, index_trajectory, regressor, fig=None, options=dict()):
    if fig is None:
        fig = plt.subplot(111)

    n_iter = index_trajectory.shape[0]
    for it in range(n_iter):
        alpha = alpha_trayectory[it, :]
        while len(fig.lines) > 2:
            fig.lines.pop(-1)

        x_idx = index_trajectory[it]
        if options.get('sgd_point', False):
            opt = {'marker': 'mX', 'label': 'Current SGD point', 'size': 15}
            plot_data(Xtr[x_idx], Ytr[x_idx], fig=fig, options=opt)

        regressor.set_weights(alpha)
        Yhat = regressor.predict(X)
        fig.plot(X, Yhat, options.get('marker', 'g-'), label=options.get('label', 'Kernel'))
        IPython.display.clear_output(wait=True)
        IPython.display.display(plt.gcf())

        process_plot(fig, options)
        # Wait for human visualization. 
        time.sleep(0.1)

    plt.close()


def classification_progression(X, Y, w_trajectory, index_trajectory, classifier, contour_plot=None, error_plot=None, options=dict()):
    if contour_plot is not None:
        contour_opts = options.get('contour_opts', dict())

    if error_plot is not None:
        error_opts = options.get('error_opts', dict())
        current_error_line, = error_plot.plot([], [], error_opts.get('marker', 'g*-'), label='Current Loss') 
        train_error_line, = error_plot.plot([], [], error_opts.get('marker', 'r*-'), label='Train Loss') 
        test_error_line, = error_plot.plot([], [], error_opts.get('marker', 'b*-'), label='Test Loss')

    process_plot(contour_plot, contour_opts)

    min_x = np.min(X[:, 0]);
    max_x = np.max(X[:, 0]);
    min_y = np.min(X[:, 1]);
    max_y = np.max(X[:, 1]);
    n_points = options.get('n_points', 20)
    [xg, yg] = np.meshgrid(np.linspace(min_x, max_x, n_points),
                           np.linspace(min_y, max_y, n_points))

    x1g = np.concatenate((xg.reshape([-1, 1]),
                          yg.reshape([-1, 1]),
                          np.ones_like(xg).reshape([-1, 1])),
                         axis=1)

    n_iter = index_trajectory.shape[0]

    current_idx = []
    current_loss = []

    train_loss = []
    test_loss = []
    test_idx = []

    # error_plot.set_xlim([0, n_iter])
    # error_plot.
    for it in range(n_iter):
        if contour_plot is not None:
            while len(contour_plot.lines) > contour_opts.get('n_classes', 2):
                contour_plot.lines.pop(-1)

            if 'contour' in locals():
                for c in contour.collections:
                    c.remove()

            x_idx = index_trajectory[it]
            if contour_opts.get('sgd_point', False):
                opt = {'marker': 'mX', 'label': 'Current SGD point', 'size': 15}
                plot_data(classifier._Xtr[x_idx, 0], classifier._Xtr[x_idx, 1], fig=contour_plot, options=opt)

            w = w_trajectory[it, :]
            classifier.set_weights(w)
            zg = classifier.predict(x1g)  # Replace this by func call

            contour = contour_plot.contourf(xg, yg, np.reshape(zg, newshape=xg.shape), alpha=0.3,
                                   cmap=matplotlib.cm.jet)  # colors=('blue', 'red'))
            # if 'cb' not in locals():
            #     cb = contour_plot.colorbar(contour)

        if error_plot is not None:
            w = w_trajectory[it, :] 
            current_idx.append(it)
            current_loss.append(classifier.loss(w, index_trajectory[it]))
            current_error_line.set_data(current_idx, current_loss)

            if (it % error_opts.get('epoch', 1)) == 0:
                test_idx.append(it)
                test_loss.append(classifier.test_loss(w))
                train_loss.append(classifier.loss(w))

                test_error_line.set_data(test_idx, test_loss)
                train_error_line.set_data(test_idx, train_loss)
            
            error_plot.relim()
            error_plot.autoscale()
            error_plot.legend(loc='upper right')

        plt.draw()

        IPython.display.clear_output(wait=True)
        IPython.display.display(plt.gcf())

        time.sleep(0.1)

    plt.close()


def plot_classification_boundaries(X, classifier, fig=None, options=dict()):
    if fig is None:
        fig = plt.subplot(111)
    min_x = np.min(X[:, 0]);
    max_x = np.max(X[:, 0]);
    min_y = np.min(X[:, 1]);
    max_y = np.max(X[:, 1]);
    n_points = options.get('n_points', 20)
    [xg, yg] = np.meshgrid(np.linspace(min_x, max_x, n_points),
                           np.linspace(min_y, max_y, n_points))

    x1g = np.concatenate((xg.reshape([-1, 1]),
                          yg.reshape([-1, 1]),
                          np.ones_like(xg).reshape([-1, 1])),
                         axis=1)

    Zg = classifier.predict(x1g)
    contour = fig.contourf(xg, yg, np.reshape(Zg, newshape=xg.shape), alpha=0.3,
                           cmap=matplotlib.cm.jet)  # colors=('blue', 'red'))
    cb = plt.colorbar(contour)
