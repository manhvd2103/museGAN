from flask import Flask, request
from musegan_gen_music import *
from flask_restful import reqparse

app = Flask(__name__)
@app.route("/gen_musegan", methods=['GET', 'POST'])
def gen():
    if request.method == 'POST':
        parser = reqparse.RequestParser()
        parser.add_argument('ticks_per_quarter', action='append')

        args = parser.parse_args()
        ticks_per_quarter = int(args['ticks_per_quarter'][0])
    elif request.method == 'GET':
        model = request.args.get('model', type=str)
        ticks_per_quarter = request.args.get('ticks_per_quarter', type=int)
    if model.lower() == 'musegan':
        result = musegan_gen(tempo=ticks_per_quarter)
        return result
    
if __name__ == "__main__":
    app.run(debug=False)