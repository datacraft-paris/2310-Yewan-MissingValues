#!/bin/bash
test ! -d data && (echo "ERROR: can not find directory data. Are you starting this script at the root of the directory ?" && exit 1)
zip -ur data.zip data
