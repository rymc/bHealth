# NewLib

Let's create a library.

## Installation and requirements

```
python3.6 -m venv venv
source  venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


## Unittest

Run unittests for the digihealth library with the following command

```
python -m unittest discover digihealth
```

## Coverage

In order to check that the tests are exhaustive, and we do not miss any part of
the code, we can use the coverage tool in the following way.

```
pip install coverage
coverage run --source=./digihealth -m unittest discover -s digihealth
coverage report -m
```

The report should indicate what lines of code are missing a unittest.

```
Name                            Stmts   Miss  Cover   Missing
-------------------------------------------------------------
digihealth/accelerometer_example.py      107    107     0%   1-158
digihealth/datawrapper/__init__.py         0      0   100%
digihealth/datawrapper/house.py          194    194     0%   1-280
digihealth/datawrapper/utils.py           30     30     0%   1-80
digihealth/localisation_example.py       119    119     0%   1-200
digihealth/metrics.py                     93      1    99%   138
digihealth/synthetic.py                   58     58     0%   1-147
digihealth/synthetic_example.py           92     92     0%   1-171
digihealth/synthetic_long_example.py      84     84     0%   1-161
digihealth/tests/__init__.py               0      0   100%
digihealth/tests/test_metrics.py          96      0   100%
digihealth/tests/test_transforms.py      165      0   100%
digihealth/transforms.py                  60      2    97%   60, 90
digihealth/visualisations.py             109    109     0%   1-316
-------------------------------------------------------------
TOTAL                            1207    796    34%
```


## Examples

In order to run the examples you will need to run the following commands to download a dataset[1] and place the files in the required directory. 
```
cd scripts
bash download-rssi-acc.sh
```

[1] Byrne D, Kozlowski M, Santos-Rodriguez R, Piechocki R, Craddock I. Residential wearable RSSI and accelerometer measurements with detailed location annotations. Scientific data. 2018 Aug 21;5:180168.
