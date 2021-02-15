#!/bin/sh
docker run \
  -dit \
  --gpus '"device=0"' \
  -v ~/workspace:/workspace \
  -p 8887:8501 \
  --name intern_tinoue_semantic_segmentation\
  --rm \
  --shm-size=256m \
  intern_tinoue_segmentation:latest