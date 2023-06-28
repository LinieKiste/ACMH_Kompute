# downloads from https://www.eth3d.net/datasets

from urllib import request
remote_url = 'https://www.eth3d.net/data/relief_dslr_undistorted.7z'
local_file = 'data/relief_dslr_undistorted.7z'
request.urlretrieve(remote_url, local_file)

# extract data
import os
import py7zr
py7zr.unpack_7zarchive('data/relief_dslr_undistorted.7z', "data")
