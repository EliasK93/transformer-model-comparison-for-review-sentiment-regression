import os
import random
import ujson
import flair
from flair.datasets import ClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models.text_regression_model import TextRegressor
from flair.trainers import ModelTrainer


def main():
    # prepare the train and test sets and write them to /training_data
    prepare_sample_datasets_from_json(json_path="training_data/raw_files/laptops.json")

    # fine-tune each model on the train set, save them to /models and evaluate
    for model in ['albert-base-v2', 'bert-base-cased',
                  'bert-base-uncased', 'distilbert-base-cased',
                  'distilbert-base-uncased', 'microsoft/deberta-v3-base',
                  'roberta-base', 'xlnet-base-cased']:
        finetune_pretrained_model(model_name=model)


def prepare_sample_datasets_from_json(json_path: str, sample_size: int = 10000, test_split: float = 0.05, seed: int = 1):
    """
    Loads the raw JSON reviews file, takes a sample with equal distribution of each rating, splits to a train set
    and a test set and saves them as FastText-formatted text files.

    :param json_path: relative path to JSON-file
    :param sample_size: overall number of reviews to use for train and test set
    :param test_split: percentage of sample to put in the test set
    :param seed: seed to use for splitting
    """

    if os.path.exists("training_data/test.txt"):
        print("Skipped preparing training data")
        return

    random.seed(seed)

    # load raw JSON file into a dict
    with open(json_path) as json_file:
        reviews = ujson.load(json_file)

    # extract only text (replacing line-breaks with whitespaces) and rating as list of tuples
    tuples = [(r["text"].replace('\n', ' '), r["rating"]) for r in reviews]

    # convert to FastText-formatted lines while ensuring equal distribution of each of the five rating classes
    lines = []
    for rating in [1, 2, 3, 4, 5]:
        reviews_for_rating = [t for t in tuples if t[1] == rating]
        reviews_for_rating = reviews_for_rating[:int(sample_size / 5)]
        lines += [f"__label__{str(int(tup[1]))} {tup[0]}\n" for tup in reviews_for_rating]

    # write to train and test set using the defined split ratio
    random.shuffle(lines)
    test_size = int(test_split * len(lines))
    with open("training_data/test.txt", "w") as test_set_file:
        for line in lines[:test_size]:
            test_set_file.write(line)
    with open("training_data/train.txt", "w") as train_set_file:
        for line in lines[test_size:]:
            train_set_file.write(line)


def finetune_pretrained_model(model_name: str, seed: int = 1, max_epochs: int = 10):
    """
    Loads the pretrained transformer model, trains it on the train set and saves it to /models.

    :param model_name: name of the pretrained transformer model (official model_id on https://huggingface.co/models)
    :param seed: seed to use for training
    :param max_epochs: maximum number of epochs to train (fine-tune) the TransformerDocumentEmbeddings for
    """

    model_dir = "models/"+model_name.replace("/", "-")

    if os.path.exists(model_dir):
        print("Skipped training for model " + model_name)
        return

    flair.set_seed(seed)

    # internal name of the label to train on
    label_name = "rating"

    # load the corpus (FastText-formatted text files) to train and evaluate on
    corpus = ClassificationCorpus("training_data", train_file="train.txt", test_file="test.txt", label_type=label_name)

    # initialize TransformerDocumentEmbeddings using default parameters (fine-tune last layer of the Transformer
    # model using CLS as pooling strategy to calculate document level embeddings from token level embeddings)
    document_embeddings = TransformerDocumentEmbeddings(model_name, fine_tune=True, layers="-1", pooling="cls")

    # initialize TextRegressor
    model = TextRegressor(document_embeddings, label_name=label_name)

    # initialize ModelTrainer
    trainer = ModelTrainer(model, corpus)

    # fine-tune the pre-trained model on the corpus
    trainer.fine_tune(model_dir, max_epochs=max_epochs, mini_batch_size=2, embeddings_storage_mode="gpu")


if __name__ == '__main__':
    main()
