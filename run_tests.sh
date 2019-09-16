#!/bin/bash

coverage run --source=./bhealth -m unittest discover -s bhealth
coverage report -m
