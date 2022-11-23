import pandas as pd
from flask import render_template,Flask,request

app=Flask(__name__)

movie = pd.read_csv("C:\\Users\\faris\\OneDrive\\Documents\\Projek UAS Alpro\\datasets\\movies.csv")

@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "GET":
        titles_select = movie['title'].values.tolist()
          
        return render_template("main.html", titles_select
        =titles_select)

if __name__ == '__main__':
    app.run(debug=True)