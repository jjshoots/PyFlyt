#!/bin/sh

rm -r ../docs
make html
cp ./docs/html/ ../docs -r
