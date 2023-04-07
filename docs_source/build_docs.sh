#!/bin/sh

rm -r ../docs ./build
make html
cp ./build/html/ ../docs -r
cd ../docs
touch .nojekyll
