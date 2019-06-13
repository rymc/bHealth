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

Run unittests for the src library with the following command

```
python -m unittest discover src
```


## Examples

In order to run the examples you will need to run the following commands to download a dataset[1] and place the files in the required directory. 
```
wget -O rssi-acc-data.zip https://ndownloader.figshare.com/files/14220518
unzip rssi-acc-data.zip
mkdir data
mkdir data/acc_loc_data
mv ble-accelerometer-indoor-localisation-measurements data/acc_loc_data
```

[1] Byrne D, Kozlowski M, Santos-Rodriguez R, Piechocki R, Craddock I. Residential wearable RSSI and accelerometer measurements with detailed location annotations. Scientific data. 2018 Aug 21;5:180168.
