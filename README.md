# uvl-analytics-concepts-lda [![badge](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.de.html)

## About

This is a microservice that can run in a docker container and perform LDA topic detection when queried by a REST call.

The project uses the Gensim Topic Modeling Library: (https://radimrehurek.com/gensim/)

## REST API

See [swagger.yaml](../master/swagger.yaml) for details. The tool at https://editor.swagger.io/ can be used to render the swagger file.

## Method Parameter

**chunksize** - Number of documents to be used in each training chunk. Default: 2000

**passes** - Number of passes through the corpus during training. Default: 1

**iterations** - Maximum number of iterations through the corpus when inferring the topic distribution of a corpus. Default: 500

**n_topics** - The number of topics that shall be detected. Higher topic coherence indicates a better n_topics. Can be any number > 0. Default: 10

**stemming** - Include stemming in preprocessing. Default: False

**fix_random** - Set to `true` to fix random seed to `0`. This will make the results reproducible. Default: false

## License
Free use of this software is granted under the terms of the [GPL version 3](https://www.gnu.org/licenses/gpl-3.0.de.html) (GPL 3.0).
