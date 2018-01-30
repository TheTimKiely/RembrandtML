from unittest import TestCase

from rembrandtml.data import DataContainer


class TestDataContainer(TestCase):
    def test_build_generator(self):
        self.fail()

    def test_generator(self):
        self.fail()

    def test_prepare_data(self):
        data_container = DataContainer('sklearn', 'boston')
        X, y = data_container.prepare_data('boston')

    def test_prepare_file_data(self):
        self.fail()
