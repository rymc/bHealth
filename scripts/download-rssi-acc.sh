#!/bin/sh
wget -O rssi-acc-data.zip https://ndownloader.figshare.com/files/14220518
unzip rssi-acc-data.zip
mkdir data
mkdir data/acc_loc_data
mv ble-accelerometer-indoor-localisation-measurements data/acc_loc_data
mv data/ ../
