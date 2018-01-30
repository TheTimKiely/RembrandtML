import sys

from rembrandtml.test_MLModelBase import TestMLModelBase


def main(params):
    tests = TestMLModelBase()
    tests.test_prepare_data()


if __name__ == '__main__':
    main(sys.argv[1:])
