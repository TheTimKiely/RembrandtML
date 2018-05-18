import os

from rembrandtml import app


class TestsIntegration(object):
    def test_predict_not_bloom(self):
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        data_file = os.path.join(cur_dir, 'images', 'not-bloom.jpg')
        model_file = os.path.join(cur_dir, '..', '..', 'models', 'hab_KerasBinaryClassifier_model.h5')
        params = ['-t', 'predict', '-d', data_file, '-m', model_file]
        app.main(params)
        print('Expected 0')

    def test_predict_bloom(self):
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        data_file = os.path.join(cur_dir, 'images', 'bloom2.jpg')
        model_file = os.path.join(cur_dir, '..', '..', 'models', 'hab_KerasBinaryClassifier_model.h5')
        params = ['-t', 'predict', '-d', data_file, '-m', model_file]
        app.main(params)
        print('Expected 1')

    def test_train(self):
        data_dir = '/home/timk/code/ML/projects/cyanotracker/images/raw/train/'
        test_data_dir = '/home/timk/code/ML/projects/cyanotracker/images/raw/test/'
        output_dir = '/home/timk/code/ML/RembrandtML/models'
        params = ['-t', 'train', '-d', data_dir, '-c', test_data_dir, '-o', output_dir, '-e', '20']
        app.main(params)


if __name__ == '__main__':
    tests = TestsIntegration()
    #tests.test_train()
    tests.test_predict_bloom()
    tests.test_predict_not_bloom()