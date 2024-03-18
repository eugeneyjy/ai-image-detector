# Author: Eugene Yong
# Date: March 18, 2024
# Final Project: AI Image Detector
# Code Adapted From: My pass assignments in CS 579 (Trustworth Machine Learning)

import os
from pathlib import Path
from time import time

REPORT_DIR =  './reports'
CURRRENT_TIME = f'local_{int(time() * 1000)}'

def get_report_dir():
    if not os.path.exists(REPORT_DIR):
        os.mkdir(REPORT_DIR)
    if not os.path.exists(f'{REPORT_DIR}/{CURRRENT_TIME}'):
        os.mkdir(f'{REPORT_DIR}/{CURRRENT_TIME}')
    return f'{REPORT_DIR}/{CURRRENT_TIME}'
