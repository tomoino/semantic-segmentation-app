#!/bin/sh
docker build \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  -f ./Dockerfile \
  -t intern_tinoue_segmentation \
  --force-rm=true \
  .