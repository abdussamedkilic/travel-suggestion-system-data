from flask import Flask, jsonify
from flask_cors import CORS
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from Data.read_file import ReadFile

app = Flask(__name__)
CORS(app)
read_file = ReadFile("Output/")


@app.route("/ping", methods=["GET"])
def ping_v1():
    return (
        jsonify(
            {
                "success": True,
                "version": "0.0.1",
            }
        ),
        200,
    )


@app.route("/get_top_ten/<city_name>/<place_name>", methods=["GET"])
def get_data(city_name, place_name):
    response_data = read_file.read_excel_for_flask("output.xlsx", city_name, place_name)
    if response_data["success"]:
        return jsonify(response_data), 200
    return jsonify({"success": "not found"}), 400


if __name__ == "__main__":
    app.run(debug=True, port=8000)
