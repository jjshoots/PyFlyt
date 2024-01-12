#!/bin/sh

# delete the old build files
rm -r ../docs ./build

# build the thing
make html linkcheck

# move the thing to the right place
mv ./build/html/ ../docs

# delete the build in the source file
rm -r ./build

# add nojekyll to the docs
cd ../docs
touch .nojekyll
