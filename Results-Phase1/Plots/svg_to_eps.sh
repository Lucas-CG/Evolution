#!/bin/bash
for i in $(ls | awk -F. '{print $1}'); do inkscape $i.svg -o $i.eps --export-ignore-filters --export-ps-level=3; done
