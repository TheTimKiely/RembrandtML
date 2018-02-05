import numpy as np
from matplotlib import  cm
import matplotlib.pyplot as plt
import seaborn as sns


class PlottingData(object):
    def __init__(self, name, history, series_styles=('b','r')):
        self.Name = name
        self.History = history
        self.SeriesStyles = series_styles


class PlotData:
    def __init__(self):
        self._legend = True


class SeriesData:
    def __init__(self):
        self._x_label = None;

class Plotter(object):
    def __init__(self):
        self.Style = 'ggplot'

    def plot(self, x, y):
        plt.plot(x, y)

    def save(self, name):
        plt.savefig(name)

    def show(self):
        if self.Style:
            plt.style.use(self.Style)
        plt.show()

    def display_plot(self, alpha_space, cv_scores, cv_scores_std):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(alpha_space, cv_scores)

        std_error = cv_scores_std / np.sqrt(10)

        ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
        ax.set_ylabel('CV Score +/- Std Error')
        ax.set_xlabel('Alpha')
        ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
        ax.set_xlim([alpha_space[0], alpha_space[-1]])
        ax.set_xscale('log')

    def boxplot(data_frame, x_column_name, y_column_name, rot):
        data_frame.boxplot(x_column_name, y_column_name, rot=rot)

    def plot_model_complexity(neighbors, train_accuracy, test_accuracy):
            # Generate plot
            plt.title('k-NN: Varying Number of Neighbors')
            plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
            plt.plot(neighbors, train_accuracy, label='Training Accuracy')
            plt.legend()
            plt.xlabel('Number of Neighbors')
            plt.ylabel('Accuracy')

    def heatmap(corr, square=True, cmap='RdYlGn'):
        sns.heatmap(corr, square=True, cmap='RdYlGn')

    def plot_scatter(self, X, y, xlabel=None, ylabel=None, color=None):
        plt.scatter(X, y, color=color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def plot_image(self, data):
        image = data.reshape(28, 28)
        plt.imshow(image, cmap=cm.binary,
                   interpolation="nearest")
        plt.axis("off")

    def plot_images(self, instances, images_per_row=10, **options):
        size = 28
        images_per_row = min(len(instances), images_per_row)
        images = [instance.reshape(size, size) for instance in instances]
        n_rows = (len(instances) - 1) // images_per_row + 1
        row_images = []
        n_empty = n_rows * images_per_row - len(instances)
        images.append(np.zeros((size, size * n_empty)))
        for row in range(n_rows):
            rimages = images[row * images_per_row: (row + 1) * images_per_row]
            row_images.append(np.concatenate(rimages, axis=1))
        image = np.concatenate(row_images, axis=0)
        plt.imshow(image, cmap=cm.binary, **options)
        plt.axis("off")

    def plot(self, X, y, color = 'blue', linewidth=2):
        plt.plot(X, y, color=color, linewidth=linewidth)

    def clear(self):
        plt.clf()

class MetricsPlotter(Plotter):
    def __init__(self):
        self.Style = 'ggplot'

    def build_series(self, epochs, data, style, label):
        series_data = SeriesData()
        series_data.x_data = epochs
        series_data.y_data = data
        series_data.series_style = style
        series_data.series_label = label
        return series_data

    def plot_histories(self, histories, figures):
        for history in histories:
            self.plot_metrics(history, figures)

    def plot_metrics(self, metrics, figures):
        history = metrics.History.history
        # change to metrics['loss']
        for layout in figures:
            plt.figure(layout[0])
            for i, series_name in enumerate(layout[1]):
                series_data = history[series_name]
                epochs = range(1, len(series_data) + 1)
                series = self.build_series(epochs, series_data, metrics.SeriesStyles[i], f'{metrics.Name} {series_name}')
                self.add_series(series)

        '''        
        loss = history['loss']
        val_loss = history['val_loss']
        epochs = range(1, len(loss) + 1)

        loss_series = self.build_series(epochs, loss, 'bo', 'Training Loss')
        self.add_series(loss_series)
        val_loss_series = self.build_series(epochs, val_loss, 'b', 'Validation Loss')
        self.add_series(val_loss_series)
        plt.clf()
        acc = history['acc']
        val_acc = history['val_acc']
        acc_series = self.build_series(epochs, acc, 'ro', 'Training Accuracy')
        self.add_series(acc_series)
        val_acc_series = self.build_series(epochs, val_acc, 'r', 'Validation Accuracy')
        self.add_series(val_acc_series)
        '''
        plt.title('Training & Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

    def add_series(self, series_data):
        plt.plot(series_data.x_data, series_data.y_data, series_data.series_style, label=series_data.series_label)

    def show_plot(self, x, y, xlabel, ylabel, legend=False):
        plt.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if (legend == True):
            plt.legend()
        plt.show()
