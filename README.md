 # ai-image-detector
A ResNet50 model that tell apart AI generated and Human generated images.
## Dataset 
`https://huggingface.co/datasets/eugeneyjy/ai-human-images`  
Clone with Git:  
`git clone https://huggingface.co/datasets/eugeneyjy/ai-human-images`  
Put dataset such that it locate at the parent directory of the files like `../data/ai-human-images/data` or specify `--data-dir` in command line.

## Dependencies
```
python >= 3.10
numpy
torch
torchvision
tqdm
matplotlib
scikit-learn
opencv-python
pyarrow
```

# How to Run
Do Nothing:  
`python train.py --version 1`  
Class Weighted Sampling:  
`python train.py --weighted-class --version 1`  
Cost Sensitive Learning:  
`python train.py --cost-sensitive-learning --version 1`  
Style Weighted Sampling:  
`python train.py --weighted-style --version 1`  
Augmented Weighted Sampling:  
`python train.py --weighted-style --horizontal-flip --rotation --versions 1`  
Old + New:  
`python train.py`  
Old + Frequent Sample New:  
`python train.py --weighted-version`  
