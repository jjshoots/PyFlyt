#!/bin/sh

rm -r ../docs
make html
mv docs/html ../docs
