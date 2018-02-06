
class RmlTest(object):

    def prepare_assert(self, expected, actual):
        """
        Creates a string that reports the expected and actual values that were passed to the assert
        :param expected:
        :param actual:
        :return:
        """
        return (expected, actual, f'Expected: {expected} Actual: {actual}')
