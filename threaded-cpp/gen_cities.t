#!/bin/bash

#
#   invoke as: gen_cities [n] [range]
#
#   n is the number of cites (default is 10)
#   range is the maximum a and y values (default is 500)
#
n=${1:-10}
range=${2:-500}

awk -v n=$n -v range=$range '
    END {
        min_xy = range / 200;
        for ( i = 0; i < n; i++ ) {
            x = min_xy + (range-min_xy)*rand();
            y = min_xy + (range-min_xy)*rand();
            printf "%10.2f %10.2f\n", x, y;
        }
    }' < /dev/null

