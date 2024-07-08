# Environment setup

## GPU version
### Package installation
```
conda create -n env4 python=3.8.5
conda activate env4
pip install tensorflow-gpu==2.5.0
pip install adversarial-robustness-toolbox==1.13.0
conda install numpy==1.23.5  
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
pip install git+https://github.com/fra31/auto-attack
pip install foolbox==3.3.1
```

## Testing
`python test.py tensorflow torch foolbox numpy`
The output should be:
```
libary_name succeed version
tensorflow True 2.5.0
torch True 1.12.1
foolbox True 3.3.1
numpy True 1.23.5
```
