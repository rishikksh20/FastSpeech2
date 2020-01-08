from flask import Flask, request, jsonify, send_file
from synthesis import infer
#from utils.task_queue import add_task
#from utils.api_services import validation as valid
from time import strftime

__author__ = "@rishikesh"

import traceback

app = Flask(__name__)

@app.route('/fastspeech',methods=['POST'])
def wavernn():
    # video: list of base64 of frames
    api_request = request.get_json()
    wav_filename = infer(api_request["text"])
    return send_file(
        wav_filename,
        mimetype="audio/wav",
        as_attachment=True,
        attachment_filename=wav_filename)


@app.route("/")
def hello():
    return "Hello World!"

@app.route("/error")
def get_nothing():
    """ Route for intentional error. """
    return "" # intentional non-existent variable


if __name__ == '__main__':
    app.run(debug=True, port=6006, host='0.0.0.0')
    #app.run(debug=True)
