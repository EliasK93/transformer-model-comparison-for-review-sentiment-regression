## Review Sentiment Regression: Transformer Model Comparison

Some experiments to compare the performances of some [pre-trained transformer models](https://huggingface.co/models) on a basic sentiment regression task after fine-tuning them on a sample of the dataset. 

A total of 8 transformer models were trained (fine-tuned) on product reviews from the [Amazon Review Dataset](https://nijianmo.github.io/amazon/index.html)
in the product category _Traditional Laptops_ (`Electronics` 🡒 `Computers & Accessories` 🡒 `Computers & Tablets` 🡒 `Laptops` 🡒 `Traditional Laptops`)
to predict the star rating of the review given the concatenated summary and text of a review. 

A random sample with a size of 10.000 reviews (2.000 for each of the five rating classes 1, 2, 3, 4 and 5 stars) 
was used to fine-tune each of the pre-trained models on the Laptops reviews data. 

The model fine-tuning and evaluation was implemented in [Flair](https://github.com/flairNLP/flair).
The task was defined as regression task using the `TransformerDocumentEmbeddings` class (which uses models from [huggingface](https://huggingface.co/)) and the (experimental) `TextRegressor` class.
The maximum number of training epochs for each model was set to 10.

<br>

### Results

|               Model               |  MSE<sup>1</sup> | MAE<sup>2</sup> | Pearson<sup>3</sup> | Training time<sup>4</sup> |
|:---------------------------------:|:-----------------|----------------:|:-------------------:|:-------------------------:|
|           albert-base-v2          | 0.53 (#7)        | 0.43 (#4)       | 0.86 (#8)           | 0h 56m 11s (#3)           |
|          bert-base-cased          | 0.51 (#5)        | 0.44 (#5)       | 0.88 (#6)           | 1h 05m 28s (#6)           |
|         bert-base-uncased         | 0.40 (#1)        | 0.37 (#1)       | 0.90 (#2)           | 1h 02m 30s (#4)           |
|       distilbert-base-cased       | 0.44 (#4)        | 0.40 (#3)       | 0.89 (#4)           | 0h 38m 42s (#2)           |
|      distilbert-base-uncased      | 0.42 (#2)        | 0.39 (#2)       | 0.89 (#3)           | 0h 36m 20s (#1)           |
|     microsoft/deberta-v3-base     | 0.44 (#3)        | 0.46 (#6)       | 0.91 (#1)           | 1h 51m 01s (#8)           |
|            roberta-base           | 0.54 (#8)        | 0.50 (#8)       | 0.87 (#7)           | 1h 04m 33s (#5)           |
|          xlnet-base-cased         | 0.52 (#6)        | 0.48 (#7)       | 0.88 (#5)           | 1h 33m 30s (#7)           |

<sup>1</sup>: Mean Squared Error<br>
<sup>2</sup>: Mean Absolute Error<br>
<sup>3</sup>: Pearson correlation coefficient<br>
<sup>4</sup>: Time to complete training & evaluation (NVIDIA GeForce GTX 1660 Ti)<br>

<br>

### Requirements

##### - Python >= 3.8

##### - Conda
  - `pytorch==1.7.1`
  - `cudatoolkit=10.1`

##### - pip
  - `flair`
  - `ujson`

<br>

### Notes

The uploaded versions of the training data in this repository are cut off after the first 50 rows of each file, the 
real training data contains a combined 10.000 rows. The trained model files `final-model.pt` for each model are omitted in this repository.
