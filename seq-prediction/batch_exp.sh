#!/bin/bash

trap 'exit 1;' INT

for i in {1..100}
do
   python recurrentpredictors.py $i 10 mean_squared_error ~/dev-repos/factory-robot-data/imgs_2015-10-30/
done
