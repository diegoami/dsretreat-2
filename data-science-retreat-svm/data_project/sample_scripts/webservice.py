import pickle as cPickle

import numpy as np
import sklearn.externals.joblib
from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

# parser = reqparse.RequestParser()
# parser.add_argument('task')

pipeline = sklearn.externals.joblib.load('pipeline.pkl')
label_encoder = sklearn.externals.joblib.load('label_encoder.pkl')
label_names = cPickle.load(open('train_target_names.pkl'))
print(label_names)

class NewsgroupService(Resource):
    def post(self):
        request_body = request.get_json()
        prediction = pipeline.predict([request_body['post_text']]).reshape(-1)
        print(prediction    )
        try:
            print(prediction[0])
            print(label_names[prediction[0]])

            predicted_newsgroup = label_names[prediction[0]]
        except:
            predicted_newsgroup = 'unknown'

        return {
            'newsgroup': predicted_newsgroup
        }


api.add_resource(NewsgroupService, '/')

if __name__ == '__main__':
    app.run(debug=True)
