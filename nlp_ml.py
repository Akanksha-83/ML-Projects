import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split

df=pd.read_csv('spam_train.csv')

x=df['Emails']

y=df['Class']
print(x)
print(y)


#Convertion of Numeric Format
vectorizer = TfidfVectorizer()

vec = vectorizer.fit_transform(x)

print(vec.toarray())

print(vectorizer.get_feature_names())


tfidf = TfidfVectorizer(stop_words='english', use_idf=False, smooth_idf=False)  # TF-IDF

X=tfidf.fit_transform(x)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

clf_rf=RandomForestClassifier()

clf_rf.fit(X_train,y_train)

pre=clf_rf.predict(X_test)

acc=accuracy_score(y_test,pre)
print(acc)

pre_score=precision_score(y_test,pre,pos_label='spam')
print(pre_score)

rec_score=recall_score(y_test,pre,pos_label='spam')
print(rec_score)

f1score=f1_score(y_test,pre,pos_label='spam')
print(f1score)


#testdata=["Fine if that is the way u feel. That is the way its gota b"]
testdata=["FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, 1.50 to rcv"]
X_test=tfidf.transform(testdata)
print("xtes=",X_test)
pre_res=clf_rf.predict(X_test)
print(pre_res[0])


