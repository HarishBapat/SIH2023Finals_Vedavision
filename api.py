from flask import Flask, request, jsonify
from flask_cors import CORS
from main import get_qa_chain

app = Flask(__name__)

CORS(app)

chain = get_qa_chain()
@app.route('/', methods=["GET"])
def home():
    response = {"message": "This is VedaVision Chatbot"}
    return jsonify(response)

@app.route('/chatbot', methods=['POST'])
def answer_query():
    data = request.get_json()
    query = data.get('query')
    try :
        # chain = get_qa_chain()
        response=chain(query)
    except:
        response = {"history": "", "query": query, "result": "Currently, we didn't find any answer. You can contact an expert for more information."}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)