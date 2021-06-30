"""This file starts the microservice"""

from flask import Flask, json, jsonify, request

import data_process
import method

with open('./config.json') as config_file:
    CONFIG = json.load(config_file)

app = Flask(__name__)


@app.route("/hitec/classify/concepts/lda/run", methods=["POST"])
def post_classification_result():
    app.logger.debug('/hitec/classify/concepts/lda/run called')

    content = json.loads(request.data.decode('utf-8'))

    app.logger.info(content)

    # save content
    dataset = content["dataset"]["documents"]

    # get parameter
    params = content["params"]

    # start pre-processing
    dataset = data_process.preprocess(dataset, stemming=params["stemming"] == "true")

    # start concept detection
    topics, doc_topic, metrics = method.train_eval(dataset, int(params["n_topics"]), int(params["iterations"]),
                                                   int(params["chunksize"]), int(params["passes"]),
                                                   params["fix_random"] == "true")

    # prepare results
    res = dict()

    res.update({"topics": topics})
    res.update({"doc_topic": doc_topic})
    res.update({"metrics": metrics})

    # send results back
    return jsonify(res)


@app.route("/hitec/classify/concepts/lda/status", methods=["GET"])
def get_status():
    status = {
        "status": "operational",
    }

    return jsonify(status)


if __name__ == "__main__":
    app.run(debug=False, threaded=False, host=CONFIG['HOST'], port=CONFIG['PORT'])
