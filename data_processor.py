from sklearn.model_selection import train_test_split
import pandas as pd
import os

import args

import bert
from bert import run_classifier
from bert import tokenization



def data_processor():
    if os.path.exists("sentiment_data/train.csv"):
        x_train = pd.read_csv("sentiment_data/train.csv")
        x_test = pd.read_csv("sentiment_data/test.csv")
    else:
        data = pd.read_csv("sentiment_data/simplifyweibo_4_moods.csv")
        data["label"] = data["label"].replace(0, "happy").replace(1, "angry").replace(2, "disgust").replace(3, "sad")
        x_train,x_test, y_train, y_test = train_test_split(data,data["label"],test_size=0.4, random_state=0)
        x_test, x_dev, y_test, y_dev = train_test_split(x_test, y_test, test_size=0.4, random_state=0)
        x_train.to_csv("sentiment_data/train.csv", index = False)
        x_test.to_csv("sentiment_data/test.csv", index=False)
        x_dev.to_csv("sentiment_data/dev.csv", index=False)

    train_InputExamples = x_train.apply(lambda x: run_classifier.InputExample(guid=None,
                                                                     # Globally unique ID for bookkeeping, unused in this example
                                                                     text_a=x[args.DATA_COLUMN],
                                                                     text_b=None,
                                                                     label=x[args.LABEL_COLUMN]), axis=1)

    test_InputExamples = x_test.apply(lambda x: run_classifier.InputExample(guid=None,
                                                                   text_a=x[args.DATA_COLUMN],
                                                                   text_b=None,
                                                                   label=x[args.LABEL_COLUMN]), axis=1)


    #获取tokenizer
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)


    #获取特征
    train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, args.label_list, args.MAX_SEQ_LENGTH,
                                                                      tokenizer)
    test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, args.label_list, args.MAX_SEQ_LENGTH,
                                                                     tokenizer)

    return train_features, test_features


if __name__ == "__main__":
    data_processor()