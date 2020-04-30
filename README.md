# textsnake_pytorch
Unofficial implementation of textsnake.

Most of the codes are borrowed from [TextSnake.pytorch by princewang1994](https://github.com/princewang1994/TextSnake.pytorch), which is a great repo.

## Some new features
* Added resnet50 backbone.

* Added Batchnorm-upsampling blocks for faster convergence.

* Synthtext dataset conversion script.

# What is this repo?
This repo breaks down the 1) dataset setup, 2) model setup, 3) training setup, 4) evaluation/Inference **mostly for code reading purposes.**

I added some comments in Japanese to get some understanding of how TextSnake works.

`1. Prepare Dataset.ipynb` shows how the annotation data are converted to the TextSnake format step-by-step.

`2. Prepare Model.ipynb` shows how to setup the TextNet model.

`3. Train Model.ipynb` shows how to train the model using the prepared dataset.

`4. Inference and Evaluate.ipynb` shows how to visualize and evaluate the TextNet results.

## Results
resnet50 
`Precision = 0.7379 - Recall = 0.6681 - Fscore = 0.7012`

vgg16

vgg16 original