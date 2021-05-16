
from flask import Flask, render_template, request, jsonify


application = Flask(__name__)



@application.route('/test')
def test():

    print('hello')
    return jsonify('this is a test')


if __name__ == '__main__':
   application.run(port=6000,debug = True)


