from django.shortcuts import render,redirect
from django.http.request import QueryDict
from django.http import HttpResponseRedirect


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer


def signin(request):
    return render(request,'signin.html') 

def login(request):
    return render(request,'login.html')


def plantrip(request):
    
    if request.method == 'POST':
      cat = request.POST.get('category')
      dist = request.POST.get('district')
      price = request.POST.get('price')
      data=pd.read_csv(r"F:\final project\fyp\Tourist Destinations (4).csv")
      data0=pd.read_csv(r"F:\final project\fyp\form (1).csv")

      clear=pd.merge(data,data0,on='id')

      clear['pr'] = clear['pr'].astype(str)
      clear['tag']=clear['_key_x'] + clear['category_x'] +clear['district_x'] +clear['pr']
      ps = PorterStemmer()
    #  Check if the value is a string before applying the stemmer
      def stem(text):
        if isinstance(text, str):
          stemmed_text = ' '.join([ps.stem(word) for word in text.split()])
          return stemmed_text
        else:
          return text
      clear['tag'] = clear['tag'].apply(stem)
      clear['tag'] = clear['tag'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    
      cv = CountVectorizer(max_features=5000,stop_words='english')

      clear['tag'] = clear['tag'].fillna('') 
      cv = CountVectorizer()
      vectors = cv.fit_transform(clear['tag']).toarray()
    
      similarity = cosine_similarity(vectors)
          
      def reco():
        category = request.POST.get('category')
        district = request.POST.get('district')
        price = request.POST.get('price')

        tour = category + ' ' + district + ' ' + price
      
        tours = [tour] + [str(c) + ' ' + str(d) + ' ' + str(p) for c, d, p in zip(clear['category_x'], clear['district_x'], clear['price'])]
      
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(tours)
        similarity = vectors * vectors.T

        t_index = 0
        distances = similarity[t_index].toarray().flatten()
        t_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
      
        recommendations = []
      
        for i in t_list:
          recommended_place = clear.iloc[i[0]]['_key_y']
          recommended_category = clear.iloc[i[0]]['category_x']
          recommended_district = clear.iloc[i[0]]['district_x']
          recommended_link = clear.iloc[i[0]]['Unnamed: 7_x']
          recommended_decs = clear.iloc[i[0]]['Desc']
          recommended_price = clear.iloc[i[0]]['price']
          
          recommendations.append({
          'place': recommended_place,
          'category': recommended_category,
          'district': recommended_district,
          'link': recommended_link,
          'description': recommended_decs,
          'price': recommended_price
          })
          
          print(f"Place: {recommended_place}")
          print(f"Category: {recommended_category}")
          print(f"District: {recommended_district}")
          print(f"Link: {recommended_link}")
          print(f"Description: {recommended_decs}")
          print(f"Price: {recommended_price}")  

        return recommendations

      recomendation = reco()

      context = {'recommendations': recomendation}

      print(context)
      return render(request, 'plantrip.html', context=context)

    return render(request,'plantrip.html')


def index(request):
     places=pd.read_csv(r"F:\final project\fyp\Tourist Destinations (4).csv")
     rate=pd.read_csv(r"F:\final project\fyp\rate.csv")
     places_rate=places.merge(rate,on='id')
     clean=places_rate.drop(columns=['longitude','latitude'])
     clean=places_rate.drop(columns=['Unnamed: 5','Unnamed: 6','Unnamed: 7_y','Unnamed: 8','Unnamed: 9'])
     v=clean['vote_count']
     c=clean['vote_average'].mean()
     r=clean['vote_average']
     clean['vote_count'] = pd.to_numeric(clean['vote_count'], errors='coerce').fillna(0).astype(int)
     clean['vote_count'].quantile(0.9)
     threshold=clean['vote_count'].quantile(0.9)
     m=threshold
     clean[clean['vote_count']>=7.0]
     d=clean[clean['vote_count']>=7.0]
     def weighted_avg_rating(x,m=m,c=c):
      v=x['vote_count']
      r=x['vote_average']
      return ((r*v)+(c*m))/(v+m )
     d_copy = d.copy()
     d_copy['weighted_avg'] = d_copy.apply(weighted_avg_rating, axis=1).values
     sorted_ranking=d_copy.sort_values('weighted_avg',ascending=False )
     text=sorted_ranking[['_key','category','district','weighted_avg','vote_average','vote_count','Unnamed: 7_x']].head(10)
     
     
  
     return render(request, 'index.html',
                            {'data': list(zip(text['_key'], text['category'], text['district'], text['Unnamed: 7_x']))})    
    
