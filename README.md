# Text file topic classifier using tf-idf similarity and inverted index
- This code was written for a class assignment from COSE471, 2018-1, Korea University.
- I do not own any rights to the text content in the sample data.

## What this code does
- There are topic-labelled datasets in the "Data" folder.
- There is a document named "input_document".
- Label the input document with the most relevant topic.

## How its done
- Read text data from text files in "Data" folder.
- Stem, tokenize them in the process, and then remove stopwords from the text.
- Build an inverted index for the documents, and take only the files that have common word tokens with in the input document.
- Compute the tf-idf similarity scores from the taken documents, and find the one with the highest scores.
- Label the input document with the topic of the sample document with the highest similarity score.
