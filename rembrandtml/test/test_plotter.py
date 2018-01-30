from unittest import TestCase

from rembrandtml.data import DataContainer
from rembrandtml.plotting import Plotter


class TestPlotter(TestCase):
    def test_plot(self):
        self.fail()

    def test_plot_model_complexity(self):
        self.fail()

    def test_plot_scatter(self):
        data_container = DataContainer('sklearn', 'boston')
        X, y = data_container.prepare_data('boston')
        plotter = Plotter()
        plotter.plot_scatter(X, y, 'Number of Rooms', 'Value of House / 1000 ($)')
        plotter.show()