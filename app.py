from flask import Flask, request, jsonify, render_template
import Game

myGame = Game.backend()
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', response_message="Hello")

@app.route('/clicked', methods = ["POST"])
def clicked():

    data = request.get_json()
    positionClicked = data.get('content')

    response = myGame.oneRound(int(positionClicked))
    return jsonify(response)

@app.route('/reset', methods = ["GET"])
def reset():
    myGame.reset()
    return '', 204

if __name__ == '__main__': 
    app.run(debug=True)
