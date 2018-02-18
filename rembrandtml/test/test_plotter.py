from unittest import TestCase

from rembrandtml.data import DataContainer
from rembrandtml.visualization import Visualizer


class TestVisualizer(TestCase):
    def test_plot(self):
        self.fail()

    def test_plot_model_complexity(self):
        self.fail()

    def test_plot_scatter(self):
        data_container = DataContainer('sklearn', 'boston')
        X, y = data_container.prepare_data('boston')
        vis = Visualizer()
        vis.plot_scatter(X, y, 'Number of Rooms', 'Value of House / 1000 ($)')
        vis.show()