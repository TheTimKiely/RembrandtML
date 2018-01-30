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
        if len(self.Style) > 0;
            plt.style.use(self.Style)

        plt.show()

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
