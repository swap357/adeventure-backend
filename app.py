from __future__ import print_function
from config import *

from flask_cors import CORS, cross_origin
from flask import Flask, jsonify
from flask_cors import CORS, cross_origin
from flask import request
from logging.handlers import RotatingFileHandler
import tiktoken
import uuid
import sys
import logging
from game import get_response

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def create_app():
    
    tokenizer = tiktoken.get_encoding("gpt2")
    session_id = str(uuid.uuid4().hex)
    app = Flask(__name__)
    
    app.tokenizer = tokenizer
    app.session_id = session_id
    # log session id
    logging.info(f"session_id: {session_id}")
    app.config["file_text_dict"] = {}
    CORS(app, supports_credentials=True)

    return app

app = create_app()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler = RotatingFileHandler("debug.log", maxBytes=10000, backupCount=1)
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
app.logger.addHandler(handler)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.DEBUG)

@app.route(f"/game", methods=["POST"])
@cross_origin(supports_credentials=True)
def answer_question():
    try:
        params = request.get_json()
        question = params["user"]
        # session_id = params["sessionId"]
        answer = get_response(question)
        
        return answer
    except Exception as e:
        return str(e)

@app.route("/healthcheck", methods=["GET"])
@cross_origin(supports_credentials=True)
def healthcheck():
    return "OK"

if __name__ == "__main__":
    app.run(debug=True, port=SERVER_PORT, threaded=True)
