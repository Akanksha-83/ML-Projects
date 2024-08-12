import  os
import numpy as np
from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import img_to_array,load_img
from keras.preprocessing.image import img_to_array,load_img
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score,confusion_matrix
def imgprocessing():

    DIRECTORY = "../ImageClassification/flowers_dataset"
    CATEGORIES=['daisy','rose','sunflower']

    imgdata = []
    image_lebel = []

    for category in CATEGORIES:

        path = os.path.join(DIRECTORY, category)


        # image preprocessing
        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            #print(img_path)
            img = load_img(img_path, target_size=(128, 128))
            img = img_to_array(img)
            img = img / 255
            imgdata.append(img)
            image_lebel.append(category)

    x = np.array(imgdata)
    x= x.reshape(len(x), -1)

    y= np.array(image_lebel)

    #print("image data:",x_train)
    #print("image_labels:",y_train)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=30)

    clf_rf = RandomForestClassifier()

    clf_rf.fit(X_train, y_train)

    pre = clf_rf.predict(X_test)

    with open('rf.model', 'wb') as f:
        pickle.dump(clf_rf, f)

    print("RF algorithm:")

    acc_score = accuracy_score(y_test, pre) * 100
    print(acc_score)

    pre_score = precision_score(y_test, pre,pos_label='positive',average='micro') * 100
    print(pre_score)

    rec_score = recall_score(y_test,pre,pos_label='positive',average='micro') * 100
    print(rec_score)

    f1score = f1_score(y_test, pre,pos_label='positive',average='micro') * 100
    print(f1score)


imgprocessing()