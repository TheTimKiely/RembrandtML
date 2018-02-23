import os, sys
root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
print(root_dir)
sys.path.append(root_dir)
print(f'path: {sys.path}')
from rembrandtml.configuration import DataConfig

def main():
    data_config = DataConfig('f', 'd')
    print('This is main')

if __name__ == '__main__':
    main()