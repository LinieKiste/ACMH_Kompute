#!/bin/env python3
# downloads from https://www.eth3d.net/datasets

from urllib import request
import os
import py7zr
if 'relief_dslr_undistorted.7z' not in os.listdir('data'):
    remote_url = 'https://www.eth3d.net/data/relief_dslr_undistorted.7z'
    local_file = 'data/relief_dslr_undistorted.7z'
    request.urlretrieve(remote_url, local_file)

# extract data
if 'relief' not in os.listdir('data'):
    py7zr.unpack_7zarchive('data/relief_dslr_undistorted.7z', "data")
