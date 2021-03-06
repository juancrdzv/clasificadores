import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC

gustos = [('La pizza no esta buena',"malo"),
          ('La pizza esta buena',"bueno"),
          ('La pizza no me gusto',"malo"),
          ('La pizza me gusto',"bueno"),
          ('Esto sabe rete feo',"malo"),
          ('La pizza esta rete buena',"bueno")]

labels = ['sentimiento','valor']

df = pd.DataFrame.from_records(gustos, columns=labels)

X_train, X_test, y_train, y_test = train_test_split(df['sentimiento'], df['valor'], random_state = 0)

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

X_train_counts = count_vect.fit_transform(X_train)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


svm = SVC(C=1000000.0, gamma='auto', kernel='rbf')
svm.fit(X_train_tfidf, y_train)

print(svm.predict(count_vect.transform(["La pizza esta muy super buena."])))
