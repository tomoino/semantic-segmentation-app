# semantic-segmentation-app
## Setup
1. Clone this repository
```
https://github.com/tomoino/semantic-segmentation-app.git
```
1. Build docker image
```
sh docker/build.sh
```
1. Run docker container
```
sh docker/run.sh
```
1. Run setup.py
```
python3 setup.py
```

## Usage
### Train
```
python3 train.py --config ./configs/default.yml
```

### Evaluation
```
python3 train.py --config ./configs/default.yml --eval
```

### Streamlit Application
```
streamlit run app/app.py
```