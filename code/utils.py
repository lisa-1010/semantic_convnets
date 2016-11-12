# utils.py
# @author: Lisa Wang
# @created: Nov 12 2016
#
#===============================================================================
# DESCRIPTION: Useful functions, e.g. for creating directories on the fly.
# not specific to this project.
#===============================================================================
# CURRENT STATUS: Working
#===============================================================================
# USAGE:
# from utils import *

import os, sys, getopt

from collections import Counter

def check_if_path_exists_or_create(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:  # Guard against race condition
            return False
    return True
