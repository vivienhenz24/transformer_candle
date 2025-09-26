pretraining-data.

The purpose of this directory is to handle everything related to the text corpus used
for pretraining the transformer. So that means everything from cleaning the text corpus
to building some kind of pipeline to AWS S3 storage if we want to do something crazy and train
on a crazy amount of data.

For the moment lets just place a input.txt file we can use locally to test out tokenizer + embedding.