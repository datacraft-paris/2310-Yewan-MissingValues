#!/bin/bash
test ! -f data.zip && (echo "ERROR: can not find file data.zip. Are you starting this script at the root of the directory ?" && exit 1)
unzip -n data.zip -d data
