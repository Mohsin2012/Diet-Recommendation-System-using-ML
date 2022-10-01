from flask import Blueprint, render_template, redirect, url_for, request, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, login_required, logout_user
from .models import User
from . import db
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

auth = Blueprint('auth', __name__)


@auth.route('/login')
def login():
    return render_template('login.html')


@auth.route('/login', methods=['POST'])
def login_post():

    email = request.form.get('email')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False

    user = User.query.filter_by(email=email).first()

    if not user or not check_password_hash(user.password, password):
        flash('Please check your login details and try again.')
        return redirect(url_for('auth.login'))

    login_user(user)
    return redirect(url_for('main.index'))


@auth.route('/signup')
def signup():
    return render_template('signup.html')


@auth.route('/signup', methods=['POST'])
def signup_post():
    email = request.form.get('email')
    password = request.form.get('password')
    name = request.form.get('name')
    user = User.query.filter_by(email=email).first()

    if user:
        flash('Email address already exists')
        return redirect(url_for('auth.signup'))
    new_user = User(email=email, password=generate_password_hash(
        password, method='sha256'), name=name)

    db.session.add(new_user)
    db.session.commit()

    return redirect(url_for('auth.login'))


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.home'))

# ML model


style = "healthy"


@auth.route('/result', methods=['GET', 'POST'])
def result():

    age = 0
    weight = 0
    height = 0
    result = []
    if request.method == 'POST':
        age = int(request.form['test-age'])
        weight = float(request.form['test-weight'])
        height = float(request.form['test-height'])
        type = str(request.form['type'])
        global style
        style = str(request.form['style'])

        # call function according to style
        if style == 'weightloss':
            result = Weight_Loss(age, weight, height, type)
        elif style == 'weightgain':
            result = Weight_Gain(age, weight, height, type)
        elif style == 'healthy':
            result = Healthy(age, weight, height, type)

        if len(result) < 3:
            result.append("Avocados")
            result.append("Grapefruit")
            result.append("Pomegranate")
            result.append("Orange")
            result.append("Cereals-Corn Flakes")

        return render_template('result.html', result=result)


def Weight_Loss(age, weight, height, type):
    bmi = round(weight/(height**2), 2)
    result = []
    result.append(bmi)
    agewiseinp = 0
    # print(" Age: %s\n Weight%s\n Hight%s\n" % (age, weight, height))

    data = pd.read_csv(r'E:\Diet Rec\\project\\input1.csv')

    Breakfastdata = data['Breakfast']
    BreakfastdataNumpy = Breakfastdata.to_numpy()

    Lunchdata = data['Lunch']
    LunchdataNumpy = Lunchdata.to_numpy()

    Dinnerdata = data['Dinner']
    DinnerdataNumpy = Dinnerdata.to_numpy()

    Food_itemsdata = data['Food_items']
    breakfastfoodseparated = []
    Lunchfoodseparated = []
    Dinnerfoodseparated = []

    breakfastfoodseparatedID = []
    LunchfoodseparatedID = []
    DinnerfoodseparatedID = []

    for i in range(len(Breakfastdata)):
        if BreakfastdataNumpy[i] == 1:
            breakfastfoodseparated.append(Food_itemsdata[i])
            breakfastfoodseparatedID.append(i)
        if LunchdataNumpy[i] == 1:
            Lunchfoodseparated.append(Food_itemsdata[i])
            LunchfoodseparatedID.append(i)
        if DinnerdataNumpy[i] == 1:
            Dinnerfoodseparated.append(Food_itemsdata[i])
            DinnerfoodseparatedID.append(i)

    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID]
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.T
    val = list(np.arange(5, 16))
    Valapnd = [0]+[4]+val
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.iloc[Valapnd]
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.T

    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.T
    val = list(np.arange(5, 16))
    Valapnd = [0]+[4]+val
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.iloc[Valapnd]
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.T

    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID]
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.T
    val = list(np.arange(5, 16))
    Valapnd = [0]+[4]+val
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.iloc[Valapnd]
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.T

    for lp in range(0, 80, 20):
        test_list = np.arange(lp, lp+20)
        for i in test_list:
            if(i == age):
                # print('age is between', str(lp), str(lp+10))
                agecl = round(lp/20)

    # print("Your body mass index is: ", bmi)
    HealthCondition = ""
    if (bmi < 16):
        HealthCondition = "Severely underweight"
        clbmi = 4
    elif (bmi >= 16 and bmi < 18.5):
        HealthCondition = "Underweight"
        clbmi = 3
    elif (bmi >= 18.5 and bmi < 25):
        HealthCondition = "Healthy"
        clbmi = 2
    elif (bmi >= 25 and bmi < 30):
        HealthCondition = "overweight"
        clbmi = 1
    elif (bmi >= 30):
        HealthCondition = "Severely Overweight"
        clbmi = 0

    result.append(HealthCondition)

    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.to_numpy()
    ti = (clbmi+agecl)/2

    # K-Means Based  Dinner Food
    Datacalorie = DinnerfoodseparatedIDdata[1:, 1:len(
        DinnerfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    XValu = np.arange(0, len(kmeans.labels_))
    dnrlbl = kmeans.labels_

    # K-Means Based  Lunch Food
    Datacalorie = LunchfoodseparatedIDdata[1:, 1:len(LunchfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    XValu = np.arange(0, len(kmeans.labels_))
    lnchlbl = kmeans.labels_

    # K-Means Based  Breakfast Food
    Datacalorie = breakfastfoodseparatedIDdata[1:, 1:len(
        breakfastfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    XValu = np.arange(0, len(kmeans.labels_))
    brklbl = kmeans.labels_

    # Reading of the Dataset
    datafin = pd.read_csv(
        r'E:\Diet Rec\\project\\inputfin.csv')

    dataTog = datafin.T

    bmicls = [0, 1, 2, 3, 4]
    agecls = [0, 1, 2, 3, 4]

    weightlosscat = dataTog.iloc[[1, 2, 7, 8]]
    weightlosscat = weightlosscat.T
    weightgaincat = dataTog.iloc[[0, 1, 2, 3, 4, 7, 9, 10]]
    weightgaincat = weightgaincat.T
    healthycat = dataTog.iloc[[1, 2, 3, 4, 6, 7, 9]]
    healthycat = healthycat.T
    weightlosscatDdata = weightlosscat.to_numpy()
    weightgaincatDdata = weightgaincat.to_numpy()
    healthycatDdata = healthycat.to_numpy()
    weightlosscat = weightlosscatDdata[1:, 0:len(weightlosscatDdata)]
    weightgaincat = weightgaincatDdata[1:, 0:len(weightgaincatDdata)]
    healthycat = healthycatDdata[1:, 0:len(healthycatDdata)]

    weightlossfin = np.zeros((len(weightlosscat)*5, 6), dtype=np.float32)
    weightgainfin = np.zeros((len(weightgaincat)*5, 10), dtype=np.float32)
    healthycatfin = np.zeros((len(healthycat)*5, 9), dtype=np.float32)

    t = 0
    r = 0
    s = 0
    yt = []
    yr = []
    ys = []
    for zz in range(5):
        for jj in range(len(weightlosscat)):
            valloc = list(weightlosscat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[t] = np.array(valloc)
            yt.append(brklbl[jj])
            t += 1

        for jj in range(len(weightlosscat)):
            valloc = list(weightlosscat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[r] = np.array(valloc)
            yr.append(lnchlbl[jj])
            r += 1

        for jj in range(len(weightlosscat)):
            valloc = list(weightlosscat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[s] = np.array(valloc)
            ys.append(dnrlbl[jj])
            s += 1

    X_test = np.zeros((len(weightlosscat), 6), dtype=np.float32)

    for jj in range(len(weightlosscat)):
        valloc = list(weightlosscat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj] = np.array(valloc)*ti

    from sklearn.model_selection import train_test_split

    if type == 'Breakfast':
        X_train = weightlossfin
        y_train = yt

    elif type == 'Lunch':
        X_train = weightlossfin
        y_train = yr

    elif type == 'Dinner':
        X_train = weightlossfin
        y_train = ys

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # print('SUGGESTED FOOD ITEMS ::')
    for ii in range(len(y_pred)):
        if y_pred[ii] == 2:
            result.append(Food_itemsdata[ii])
            # print(Food_itemsdata[ii])
    return result


def Weight_Gain(age, weight, height, type):
    bmi = round(weight/(height**2), 2)
    result = []
    result.append(bmi)
    agewiseinp = 0

    data = pd.read_csv(r'E:\Diet Rec\\project\\input1.csv')
    data.head(5)
    Breakfastdata = data['Breakfast']
    BreakfastdataNumpy = Breakfastdata.to_numpy()

    Lunchdata = data['Lunch']
    LunchdataNumpy = Lunchdata.to_numpy()

    Dinnerdata = data['Dinner']
    DinnerdataNumpy = Dinnerdata.to_numpy()

    Food_itemsdata = data['Food_items']
    breakfastfoodseparated = []
    Lunchfoodseparated = []
    Dinnerfoodseparated = []

    breakfastfoodseparatedID = []
    LunchfoodseparatedID = []
    DinnerfoodseparatedID = []

    for i in range(len(Breakfastdata)):
        if BreakfastdataNumpy[i] == 1:
            breakfastfoodseparated.append(Food_itemsdata[i])
            breakfastfoodseparatedID.append(i)
        if LunchdataNumpy[i] == 1:
            Lunchfoodseparated.append(Food_itemsdata[i])
            LunchfoodseparatedID.append(i)
        if DinnerdataNumpy[i] == 1:
            Dinnerfoodseparated.append(Food_itemsdata[i])
            DinnerfoodseparatedID.append(i)

    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID]
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.T
    val = list(np.arange(5, 15))
    Valapnd = [0]+val
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.iloc[Valapnd]
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.T

    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.T
    val = list(np.arange(5, 15))
    Valapnd = [0]+val
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.iloc[Valapnd]
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.T

    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID]
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.T
    val = list(np.arange(5, 15))
    Valapnd = [0]+val
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.iloc[Valapnd]
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.T

    for lp in range(0, 80, 20):
        test_list = np.arange(lp, lp+20)
        for i in test_list:
            if(i == age):
                # print('age is between', str(lp), str(lp+10))
                tr = round(lp/20)
                agecl = round(lp/20)

    HealthCondition = ""
    if (bmi < 16):
        HealthCondition = "Severely underweight"
        clbmi = 4
    elif (bmi >= 16 and bmi < 18.5):
        HealthCondition = "Underweight"
        clbmi = 3
    elif (bmi >= 18.5 and bmi < 25):
        HealthCondition = "Healthy"
        clbmi = 2
    elif (bmi >= 25 and bmi < 30):
        HealthCondition = "overweight"
        clbmi = 1
    elif (bmi >= 30):
        HealthCondition = "Severely Overweight"
        clbmi = 0
    result.append(HealthCondition)

    val1 = DinnerfoodseparatedIDdata.describe()
    valTog = val1.T
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.to_numpy()
    ti = (bmi+agecl)/2

    # K-Means Based  Dinner Food
    Datacalorie = DinnerfoodseparatedIDdata[1:, 1:len(
        DinnerfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    XValu = np.arange(0, len(kmeans.labels_))
    dnrlbl = kmeans.labels_

    # K-Means Based  lunch Food
    Datacalorie = LunchfoodseparatedIDdata[1:, 1:len(LunchfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    XValu = np.arange(0, len(kmeans.labels_))
    lnchlbl = kmeans.labels_

    # K-Means Based  lunch Food
    Datacalorie = breakfastfoodseparatedIDdata[1:, 1:len(
        breakfastfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    XValu = np.arange(0, len(kmeans.labels_))
    brklbl = kmeans.labels_

    inp = []
    datafin = pd.read_csv(
        r'E:\Diet Rec\\project\\inputfin.csv')
    datafin.head(5)
    dataTog = datafin.T
    bmicls = [0, 1, 2, 3, 4]
    agecls = [0, 1, 2, 3, 4]
    weightlosscat = dataTog.iloc[[1, 2, 7, 8]]
    weightlosscat = weightlosscat.T
    weightgaincat = dataTog.iloc[[0, 1, 2, 3, 4, 7, 9, 10]]
    weightgaincat = weightgaincat.T
    healthycat = dataTog.iloc[[1, 2, 3, 4, 6, 7, 9]]
    healthycat = healthycat.T
    weightlosscatDdata = weightlosscat.to_numpy()
    weightgaincatDdata = weightgaincat.to_numpy()
    healthycatDdata = healthycat.to_numpy()
    weightlosscat = weightlosscatDdata[1:, 0:len(weightlosscatDdata)]
    weightgaincat = weightgaincatDdata[1:, 0:len(weightgaincatDdata)]
    healthycat = healthycatDdata[1:, 0:len(healthycatDdata)]

    weightlossfin = np.zeros((len(weightlosscat)*5, 6), dtype=np.float32)
    weightgainfin = np.zeros((len(weightgaincat)*5, 10), dtype=np.float32)
    healthycatfin = np.zeros((len(healthycat)*5, 9), dtype=np.float32)
    t = 0
    r = 0
    s = 0
    yt = []
    yr = []
    ys = []
    for zz in range(5):
        for jj in range(len(weightgaincat)):
            valloc = list(weightgaincat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[t] = np.array(valloc)
            yt.append(brklbl[jj])
            t += 1
        for jj in range(len(weightgaincat)):
            valloc = list(weightgaincat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[r] = np.array(valloc)
            yr.append(lnchlbl[jj])
            r += 1
        for jj in range(len(weightgaincat)):
            valloc = list(weightgaincat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[s] = np.array(valloc)
            ys.append(dnrlbl[jj])
            s += 1

    X_test = np.zeros((len(weightgaincat), 10), dtype=np.float32)

    for jj in range(len(weightgaincat)):
        valloc = list(weightgaincat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj] = np.array(valloc)*ti

    from sklearn.model_selection import train_test_split

    if type == 'Breakfast':
        X_train = weightgainfin
        y_train = yt

    elif type == 'Lunch':
        X_train = weightgainfin
        y_train = yr

    elif type == 'Dinner':
        X_train = weightgainfin
        y_train = ys

    clf = RandomForestClassifier(n_estimators=100)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # print('SUGGESTED FOOD ITEMS ::')
    for ii in range(len(y_pred)):
        if y_pred[ii] == 2:
            result.append(Food_itemsdata[ii])
            # print(Food_itemsdata[ii])
    return result


def Healthy(age, weight, height, type):
    bmi = round(weight/(height**2), 2)
    result = []
    result.append(bmi)
    agewiseinp = 0
    # print(" Age: %s\n Weight%s\n Height%s\n" % (age, weight, height))

    data = pd.read_csv(r'E:\Diet Rec\\project\\input1.csv')
    data.head(5)
    Breakfastdata = data['Breakfast']
    BreakfastdataNumpy = Breakfastdata.to_numpy()

    Lunchdata = data['Lunch']
    LunchdataNumpy = Lunchdata.to_numpy()

    Dinnerdata = data['Dinner']
    DinnerdataNumpy = Dinnerdata.to_numpy()

    Food_itemsdata = data['Food_items']
    breakfastfoodseparated = []
    Lunchfoodseparated = []
    Dinnerfoodseparated = []

    breakfastfoodseparatedID = []
    LunchfoodseparatedID = []
    DinnerfoodseparatedID = []

    for i in range(len(Breakfastdata)):
        if BreakfastdataNumpy[i] == 1:
            breakfastfoodseparated.append(Food_itemsdata[i])
            breakfastfoodseparatedID.append(i)
        if LunchdataNumpy[i] == 1:
            Lunchfoodseparated.append(Food_itemsdata[i])
            LunchfoodseparatedID.append(i)
        if DinnerdataNumpy[i] == 1:
            Dinnerfoodseparated.append(Food_itemsdata[i])
            DinnerfoodseparatedID.append(i)

    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID]
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.T
    val = list(np.arange(5, 15))
    Valapnd = [0]+val
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.iloc[Valapnd]
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.T

    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.T
    val = list(np.arange(5, 15))
    Valapnd = [0]+val
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.iloc[Valapnd]
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.T

    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID]
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.T
    val = list(np.arange(5, 15))
    Valapnd = [0]+val
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.iloc[Valapnd]
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.T

    for lp in range(0, 80, 20):
        test_list = np.arange(lp, lp+20)
        for i in test_list:
            if(i == age):
                # print('age is between', str(lp), str(lp+10))
                tr = round(lp/20)
                agecl = round(lp/20)

    #print("Your body mass index is: ", bmi)
    HealthCondition = ""
    if (bmi < 16):
        HealthCondition = "Severely underweight"
        clbmi = 4
    elif (bmi >= 16 and bmi < 18.5):
        HealthCondition = "Underweight"
        clbmi = 3
    elif (bmi >= 18.5 and bmi < 25):
        HealthCondition = "Healthy"
        clbmi = 2
    elif (bmi >= 25 and bmi < 30):
        HealthCondition = "overweight"
        clbmi = 1
    elif (bmi >= 30):
        HealthCondition = "Severely Overweight"
        clbmi = 0
    result.append(HealthCondition)

    val1 = DinnerfoodseparatedIDdata.describe()
    valTog = val1.T
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.to_numpy()
    ti = (bmi+agecl)/2

    # K-Means Based  Dinner Food
    Datacalorie = DinnerfoodseparatedIDdata[1:, 1:len(
        DinnerfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    XValu = np.arange(0, len(kmeans.labels_))
    dnrlbl = kmeans.labels_

    # K-Means Based  lunch Food
    Datacalorie = LunchfoodseparatedIDdata[1:, 1:len(LunchfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    XValu = np.arange(0, len(kmeans.labels_))
    lnchlbl = kmeans.labels_

    # K-Means Based  lunch Food
    Datacalorie = breakfastfoodseparatedIDdata[1:, 1:len(
        breakfastfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    XValu = np.arange(0, len(kmeans.labels_))
    brklbl = kmeans.labels_
    inp = []
    datafin = pd.read_csv(
        r'E:\Diet Rec\\project\\inputfin.csv')
    datafin.head(5)
    dataTog = datafin.T
    bmicls = [0, 1, 2, 3, 4]
    agecls = [0, 1, 2, 3, 4]
    weightlosscat = dataTog.iloc[[1, 2, 7, 8]]
    weightlosscat = weightlosscat.T
    weightgaincat = dataTog.iloc[[0, 1, 2, 3, 4, 7, 9, 10]]
    weightgaincat = weightgaincat.T
    healthycat = dataTog.iloc[[1, 2, 3, 4, 6, 7, 9]]
    healthycat = healthycat.T
    weightlosscatDdata = weightlosscat.to_numpy()
    weightgaincatDdata = weightgaincat.to_numpy()
    healthycatDdata = healthycat.to_numpy()
    weightlosscat = weightlosscatDdata[1:, 0:len(weightlosscatDdata)]
    weightgaincat = weightgaincatDdata[1:, 0:len(weightgaincatDdata)]
    healthycat = healthycatDdata[1:, 0:len(healthycatDdata)]

    weightlossfin = np.zeros((len(weightlosscat)*5, 6), dtype=np.float32)
    weightgainfin = np.zeros((len(weightgaincat)*5, 10), dtype=np.float32)
    healthycatfin = np.zeros((len(healthycat)*5, 9), dtype=np.float32)
    t = 0
    r = 0
    s = 0
    yt = []
    yr = []
    ys = []
    for zz in range(5):
        for jj in range(len(healthycat)):
            valloc = list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthycatfin[t] = np.array(valloc)
            yt.append(brklbl[jj])
            t += 1
        for jj in range(len(healthycat)):
            valloc = list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthycatfin[r] = np.array(valloc)
            yr.append(lnchlbl[jj])
            r += 1
        for jj in range(len(healthycat)):
            valloc = list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthycatfin[s] = np.array(valloc)
            ys.append(dnrlbl[jj])
            s += 1

    X_test = np.zeros((len(healthycat)*5, 9), dtype=np.float32)
    for jj in range(len(healthycat)):
        valloc = list(healthycat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj] = np.array(valloc)*ti

    from sklearn.model_selection import train_test_split

    #val = int(input("Enter 1 for Breakfast, 2 for Lunch, 3 for Dinner"))

    if type == 'Breakfast':
        X_train = healthycatfin
        y_train = yt

    elif type == 'Lunch':
        X_train = healthycatfin
        y_train = yt

    elif type == 'Dinner':
        X_train = healthycatfin
        y_train = ys

    clf = RandomForestClassifier(n_estimators=100)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print('SUGGESTED FOOD ITEMS ::')
    result = []
    for ii in range(len(y_pred)):
        if y_pred[ii] == 2:
            result.append(Food_itemsdata[ii])
            # print(Food_itemsdata[ii])

    return result


@auth.route('/workout', methods=['GET', 'POST'])
def workout():
    return render_template('workout.html', style=style)
