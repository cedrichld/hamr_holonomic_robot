#!/bin/bash
# run: ./extract_xy.sh points.txt

file="$1"

grep -oP "x=\K-?[0-9]+\.[0-9]+|y=\K-?[0-9]+\.[0-9]+" "$file" \
  | paste - - \
  | awk '{print "["$1", "$2"],"}'