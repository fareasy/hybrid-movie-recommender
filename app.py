import pandas as pd
from flask import render_template,Flask,request
import pickle
import numpy as np
import itertools
from lightfm.data import Dataset
from lightfm import cross_validation,LightFM
from bs4 import BeautifulSoup
import requests
from assist import data

app=Flask(__name__)

#data
movie = pd.read_csv("datasets\movies.csv")
links = pd.read_csv("datasets\links.csv")

#create lists needed
titles_select = movie['title'].values.tolist()
title_user=[]
rating_user=[]
list_em=[]

#check whether user pick the same movie or not
def check(movname,rating):
    if movname in title_user:
        return(0)
    else:
        title_user.append(movname)
        rating=int(rating)
        rating_user.append(rating)
        return(1)

@app.route("/", methods=["POST", "GET"])
def home():
    text_ann=""
    if request.method=='POST':
        title=request.form.get("titles")
        rating=request.form.get("ratings")
        val=check(title,rating)
        if val==0:
            text_ann="You have rated this movie already"
        if val==1:
            text_ann="You gave "+str(title)+" a "+str(rating)
    datazip=zip(title_user,rating_user)
    return render_template("main.html",titles=titles_select,text_ann=text_ann,title_user=title_user,datazip=datazip)

@app.route('/reset/', methods=['GET','POST'])
def reset():
    #reset the list
        global rating_user
        global title_user
        
        rating_user = []
        title_user=[]
        text_ann='Successfull reset'
        return render_template('main.html',text_ann=text_ann,titles=titles_select)

@app.route("/output", methods=["POST", "GET"])
def output():
    movie_recs=[]
    if request.method == "GET":
        datazip2=zip(title_user,rating_user)
        for x,y in datazip2:
            userId=611
            infos = movie[(movie['title']==x)]
            Id=infos['movieId'].iloc[0]
            genre=infos['genres'].iloc[0]
            dictmov={}
            dictmov.update({"userId": userId,"movieId":Id,"rating":y,"title":x,"genres":genre})
            list_em.append(dictmov)

        if len(title_user)>4:
            #create dataframe
            data_user = pd.DataFrame(list_em, columns=['userId','movieId','rating','title','genres'])
            data_new=pd.concat([data, data_user],ignore_index=True)
            print(data_new)

            #create model
            #creating the dataset
            dataset = Dataset()
            dataset.fit(users=data_new['userId'], items=data_new['movieId'])

            #build interactions
            (interactions, weights) = dataset.build_interactions(data_new.iloc[:, 0:3].values)

            #create the train/test split (80/20)
            train_interactions, test_interactions = cross_validation.random_train_test_split(
            interactions, test_percentage=0.2,
            random_state=np.random.RandomState(40))

            #create and fit the model
            model = LightFM(loss='warp', no_components=20, 
                 learning_rate=0.05,                 
                 random_state=np.random.RandomState(40))
            model.fit(train_interactions,epochs=10)    
            n_users, n_items = interactions.shape
            #return list --contoh
            #['Star Trek: The Motion Picture (1979)' 'Dune (1984)' 'Stand by Me (1986)'
            #'Thelma & Louise (1991)' 'Easy Rider (1969)' 'The Revenant (2015)'
            #'Beach, The (2000)' 'Robin Hood: Prince of Thieves (1991)'
            #'Motorcycle Diaries, The (Diarios de motocicleta) (2004)'
            #'Lawrence of Arabia (1962)']

            #function to generate movie recommendations
            #k=jumlah rekomendasi
            def recommendME(model1,movie,dataset,user_id=None,k=5): 
                nmovie=movie.set_index('movieId')
                user_id_map = dataset.mapping()[0][user_id] # just user_id -1 
                scores = model1.predict(user_id_map, np.arange(n_items))
                rank = np.argsort(-scores)
                selected_movies =np.array(list(dataset.mapping()[2].keys()))[rank]
                top_items = nmovie.loc[selected_movies]
                return top_items['title'][:k].values

            movie_recs=recommendME(model,movie,dataset,user_id=611,k=5)
        else:
            text_ann='Not enough items'
            return render_template('main.html',text_ann=text_ann,titles=titles_select)

        #nyoba tpi bingung --klo salah hapus aja
        for movs in  movie_recs:
            ids = []
            infos = movie[(movie['title']==movs)]
            Id=infos['movieId'].iloc[0]
            urls=links[(links['movieId']==Id)]
            url_tmdb=urls['tmdbId'].iloc[0]
            ids.append(url_tmdb)
        mainUrl = "https://www.themoviedb.org"
        def getUrl(Id):
                    filmUrl = "https://www.themoviedb.org/movie/" + str(Id)
                    return str(filmUrl)
        def scrap(url):
                    needed_headers = {'User-Agent': "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"}
                    respon = requests.get(url, headers = needed_headers )
                    rawhtml = respon.text
                    soup = BeautifulSoup(rawhtml, "html.parser")
                    return soup
        def fullUrl(soup):
                    for i in soup.find_all(class_ = "poster lazyload"):
                        url = i.get("data-src")
                        return mainUrl+url
        def getJudul(soup):
            for i in soup.find_all("h2"):
                judul = i.get_text()
            return judul
        def urlLengkap(id):
            filmUrl = str(getUrl(id))
            film = scrap(filmUrl)
            url = fullUrl(film)
            return url
        def judul(id):
            filmUrl = str(getUrl(id))
            film = scrap(filmUrl)
            judul = getJudul(film)
            return judul
        def getSinopsis(soup):
            for i in soup.find_all(class_ = "overview"):
                    sinopsis = i.get_text()
            return sinopsis
        def sinopsis(id):
            filmUrl = str(getUrl(id))
            film = scrap(filmUrl)
            sinopsis = getSinopsis(film)
            return sinopsis

        url_list=[]
        url_judul=[]
        url_film=[]
        url_syn=[]

        for id in ids:
            url = urlLengkap(id)
            judul_url = judul(id)
            film_url = getUrl(id)
            sinopsis_url = sinopsis(id)
            url_list.append(url)
            url_judul.append(judul_url)
            url_film.append(film_url)
            url_syn.append(sinopsis_url)
        
        return render_template("rec.html", movie_recs=movie_recs,url_list=url_list,url_judul=url_judul,url_film=url_film,url_syn=url_syn)

if __name__ == '__main__':
    app.run(debug=True,threaded=True)