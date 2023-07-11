#!/bin/sh
# visualize point cloud with f3d

if [ -z "$1" ]
then
        echo "Usage: ./visualize.sh [.ply file]"
else
        f3d -so --point-size=0 --comp=-2 --up -Y $1
fi

