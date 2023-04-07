#!/bin/sh

rm -r ../docs
make html
cp ./build/html/ ../docs -r
touch ../docs/.nojekyll
