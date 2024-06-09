#!/bin/bash

. /home/sabawi/.profile
. /home/sabawi/.bashrc

/usr/bin/python3 /home/sabawi/Development/stocks_evaluator/evaluate_this.py > /home/sabawi/cron_stocks.log 2>&1 
