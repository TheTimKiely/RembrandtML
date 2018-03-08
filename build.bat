REM build distribution
python setup.py sdist


REM build wheel
python setup.py bdist_wheel

REM upload to PyPi
REM twine upload dist/*
