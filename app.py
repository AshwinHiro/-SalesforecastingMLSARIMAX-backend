import pickle
from flask import Flask , request , Blueprint
from flask_cors import CORS 
from flask_pymongo import pymongo
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import datetime
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
import base64
from io import BytesIO



con_string = "mongodb+srv://Ashwin-Forecast:root@cluster0.wheso4t.mongodb.net/test"

client = pymongo.MongoClient(con_string)

db = client.get_database('sales-forecast-db')

user_collection = pymongo.collection.Collection(db,'users')

print("Mongo Db Connected Successfully")


def create_web_app():
    web_app = Flask(__name__)
    
    CORS(web_app)

    api_blueprint = Blueprint('api_blueprint',__name__)
    
    @api_blueprint.route('/')
    def index():
        return "This is an example app"
        
    @api_blueprint.route('/hello',methods=['GET'])
    def hello():
        print("Hello World")
        return "Hello World"

    @api_blueprint.route('/register_user',methods=['POST'])
    def register_user():
        res = {}
        try:
            req_body = request.json
            user_collection.insert_one(req_body)
            print(req_body)
            print("User successfully Registered")
            status = {
                "message" : "success",
                "statusMsg" : "user successfully created"
            }
        except Exception as e:
            print(e)
            status = {
                "message" : "success",
                "error" : "400"
            }
        finally:
            print("register function completed")
        res["status"] = status
        return res
    
    @api_blueprint.route('/login_user/',methods=['GET'])
    def login_user():
        res = {}
        try:
            print(request.args)
            user = {}
            user["email"] = request.args.get("email")
            user["password"] = request.args.get("password")
            dbuser = user_collection.find(user)
            final_user = list(dbuser)
            if(len(final_user) == 0):
                status = {
                    "message" : "failed",
                    "statusMsg" : "not an existing user"
                }
                res["status"] = status
                return res
            status = {
                "message" : "success",
                "statusMsg" : "existing user"
            }
            data = [{'_id' : str(user['_id']),'name' : user["name"] , 'email' : user["email"]} for user in final_user]
            res["data"] = data
        except Exception as e:
            print(e)
            status = {
                "message" : "success",
                "error" : "400"
            }
        finally:
            print("get user completed")
        res["status"] = status
        print(res)
        return res
    
    @api_blueprint.route("/uploadcsv",methods=["POST"])
    def uploadCsv():
        req = [request.files.get("file"), request.form.get("date"), request.form.get('fields')]
        print(req)
        res = {}


        # dataset = pd.read_csv(req[0])
        # model_score = model.predict(dataset)
        # print(model_score)
        # url = requests.get("https://drive.google.com/u/0/uc?id=1BO7P9JkRa6TZZXghp3wYcYjq9TK3Wyuz&export=download")
        # model = pickle.load(url)
        # model.score()
        res["status"] = {
            "status" : "Backend Msg"
        }
        
        readyTest = timeseries(req[0])
        print(readyTest)

        binaryFile = open("assets/binary.pkl","rb")
        model = pickle.load(binaryFile)
        model_score = model.predict(readyTest)
        print(model_score)
        return res

    def timeseries(file):
        train = pd.read_csv("assets/Train.csv")
        test = pd.read_csv(file)
        
        data = train.append(test,sort=False,ignore_index=True)

        data.info()

        l = dict(data.Name[data.Publisher.isnull()])

        data.isnull().sum()

        data = data[data.Genre.notnull()]

        data.Publisher[data.Name==l[475]] = "THQ"
        data.Publisher[data.Name==l[1301]] = "Electronic Arts"
        data.Publisher[data.Name==l[1667]] ="Take-Two Interactive"
        data.Publisher[data.Name==l[2212]] ="Sony Computer Entertainment"
        data.Publisher[data.Name==l[3148]] ="Majesco Entertainment"
        data.Publisher[data.Name==l[3159]] ="Majesco Entertainment"
        data.Publisher[data.Name==l[3755]] ="Majesco Entertainment"
        data.Publisher[data.Name==l[4127]] ="Sega"
        data.Publisher[data.Name==l[4508]] ="THQ"
        data.Publisher[data.Name==l[4615]] ="THQ"
        data.Publisher[data.Name==l[5294]] ="Nintendo"
        data.Publisher[data.Name==l[5633]] ="Majesco Entertainment"
        data.Publisher[data.Name==l[6256]] ="Nintendo"
        data.Publisher[data.Name==l[6415]] ="Majesco Entertainment"
        data.Publisher[data.Name==l[6532]] ="Gotham Games"
        data.Publisher[data.Name==l[6626]] ="Nintendo"
        data.Publisher[data.Name==l[6831]] ="THQ"
        data.Publisher[data.Name==l[7189]] ="Nintendo"
        data.Publisher[data.Name==l[7333]] ="Atari"
        data.Publisher[data.Name==l[7477]] ="Nintendo"
        data.Publisher[data.Name==l[7930]] ="THQ"
        data.Publisher[data.Name==l[8317]]="Nintendo"
        data.Publisher[data.Name==l[8328]]="Global Star"
        data.Publisher[data.Name==l[8358]]="Konami Digital Entertainment"
        data.Publisher[data.Name==l[8494]]="Unknown"
        data.Publisher[data.Name==l[8754]]="Majesco Entertainment"
        data.Publisher[data.Name==l[8837]]="Nintendo"
        data.Publisher[data.Name==l[8883]]="Nintendo"
        data.Publisher[data.Name==l[9703]]="Majesco Entertainment"
        data.Publisher[data.Name==l[9812]]="Konami Digital Entertainment"
        data.Publisher[data.Name==l[10418]]="Habro Interactive"
        data.Publisher[data.Name==l[11175]]="Microsoft Game Studios"
        data.Publisher[data.Name==l[11569]]="Hudson soft"
        data.Publisher[data.Name==l[11985]]="Unknown"
        data.Publisher[data.Name==l[12209]]="Nintendo"
        data.Publisher[data.Name==l[12544]]="Wargaming.net"
        data.Publisher[data.Name==l[12629]]="Nintendo"
        data.Publisher[data.Name==l[12881]]="Sony Computer Entertainment"
        data.Publisher[data.Name==l[13197]] = "Nintendo"
        data.Publisher[data.Name==l[13274]] = "Nintendo"
        data.Publisher[data.Name==l[13299]] = "Nintendo"
        data.Publisher[data.Name==l[13537]] = "Unknown"
        data.Publisher[data.Name==l[13876]] = "Zoo Games"
        data.Publisher[data.Name==l[14137]] = "Gearbox Software"
        data.Publisher[data.Name==l[14153]] = "Unknown"
        data.Publisher[data.Name==l[14163]] = "Majesco Entertainment"
        data.Publisher[data.Name==l[14189]] = "Ubisoft"
        data.Publisher[data.Name==l[14196]] = "Unknown"
        data.Publisher[data.Name==l[14519]] = "Excalibur publishing"
        data.Publisher[data.Name==l[14747]] = "Unknown"
        data.Publisher[data.Name==l[14975]] = "Unknown"
        data.Publisher[data.Name==l[15621]] = "Majesco Entertainment"
        data.Publisher[data.Name==l[16600]] = "Nintendo"
        data.Publisher[data.Name==l[16644]] = "Nintendo"

        data.drop(['Name','Year_of_Release'],axis=1,inplace=True)

        data.User_Score.unique()

        data.User_Score[data.User_Score=='tbd']='0'

        data.User_Score = data.User_Score.astype('float64')

        data.User_Score.fillna(data.User_Score.mean(),inplace=True)

        data.User_Count.fillna(data.User_Count.median(),inplace=True)

        data.Critic_Score.fillna(data.Critic_Score.median(),inplace=True)

        data.Critic_Count.fillna(data.Critic_Count.median(),inplace=True)

        data.isnull().sum()

        data['Genre']= LabelEncoder().fit_transform(data.Genre)

        data['Platform']= LabelEncoder().fit_transform(data.Platform)

        data['Publisher']= LabelEncoder().fit_transform(data.Publisher)

        data['Rating']= LabelEncoder().fit_transform(data.Rating)

        data.drop('Developer',axis=1,inplace=True)

        x = data.drop('Global_Sales',axis=1)
        y = data.Global_Sales

        scaler = MinMaxScaler()
        d = scaler.fit_transform(x)

        df = pd.DataFrame(d)

        df.columns = x.columns

        # checked the testing
      
        # newtrainx = df.iloc[0:14574,0:]
        # newtestx = df.iloc[14574:,0:]
        # newtrainy = y.iloc[0:14574]
        # newtesty = y.iloc[14574:]

        # trainx,testx,trainy,testy = train_test_split(newtrainx,newtrainy,test_size = 0.2)

        # model1 = LinearRegression().fit(trainx,trainy)


        # ans = model1.score(testx,testy)

        # print(ans)

        test_x = df.iloc[14574:,0:]
        # test_y = y.iloc[14574:]

        return test_x
    
    @api_blueprint.route("/csvupload",methods=["POST"])
    def csvupload():
        req = request.files.get("file")
        yearOffset = request.form.get("offsetYears")
        if(int(yearOffset) > 0):
            req = create_new_dataset(req, int(yearOffset) - 1)

        data = training(req,yearOffset)
        res = {}
        res["status"] = {
            "status": 400,
            "data" : data
        }
        return res
        
    def create_new_dataset(file, yearOffset):

        dt = datetime.datetime(2018, 1, 1)
        end = datetime.datetime(2018 + yearOffset, 12, 30, 23, 59, 59)
        step = datetime.timedelta(days=1)

        result = []

        while dt < end:
            result.append(dt.strftime('%Y-%m-%d %H:%M:%S'))
            dt += step
            
        result = pd.DataFrame(result)
        
        result.rename(columns={0: 'date'}, inplace=True)
        
        result.date = pd.to_datetime(result.date, format = "%Y-%m-%d")
        
        result = pd.DataFrame(result)

        csvFile = pd.read_csv(file)

        predictdata = csvFile.append(result,sort=False,ignore_index=True)
        
        predictdata.date = pd.to_datetime(predictdata.date, format = "%Y-%m-%d")

        return predictdata
        

    def training(csvFile, yearOffset):
        if(int(yearOffset) == 0):
            train = pd.read_csv(csvFile)
        else:
            train = csvFile
        temp = pd.DataFrame()
        temp["Date"] = 0
        l = []
        for i in train.date:
            l.append(str(i)[0:7])
        temp["Date"] = l
        
        train.date = temp.Date
        train.date = pd.to_datetime(train.date)
        grouped1 = train.groupby('date')['sales'].sum()
        
    #     plt.hist(pd.DataFrame(grouped1).sales)
    #     plt.xlabel("Sales")

        grouped1= pd.DataFrame(grouped1)
        
        second = train.groupby(train.date.dt.year)['sales'].mean().reset_index()
        
        second.date = pd.to_datetime(second.date, format='%Y')
        
    #     sns.lineplot((second.date + datetime.timedelta(6*365/12)), 'sales', data=second,color='red', label='Mean Sales')  
        
        # sns.lineplot('date', 'sales', data=train,color='mediumblue', label='Total Sales')
        # second = train.groupby(train.date.dt.year)['sales'].mean().reset_index()
        # second.date = pd.to_datetime(second.date, format='%Y')
        # sns.lineplot((second.date + datetime.timedelta(6*365/12)), 'sales', data=second,color='red', label='Mean Sales')  
        
        grouped1.sales[grouped1.sales==0.0] = np.NaN
        
        sar = sm.tsa.statespace.SARIMAX(grouped1.sales, order=(12,0,0), seasonal_order=(0,1,0,12), trend='c').fit()
        
        start, end, dynamic = 40, 100, 7
        
        plotimage = BytesIO()

        grouped1['forecast'] = sar.predict(start=start, end=end, dynamic=dynamic)
        
        pred_df = grouped1.forecast[start+dynamic:end]
        
        grouped1[['sales', 'forecast']].plot(color=['mediumblue', 'Red'])

        plt.savefig(plotimage , format="png")

        plt.close()

        plotimage.seek(0)

        print(plotimage)

        predictedURL = base64.b64encode(plotimage.getvalue()).decode('utf')

        returnData = {} 

        returnData["predictedimage"] = "data:image/png;base64," + predictedURL

        final_data = grouped1.to_csv()

        final_columns = [x.split(',') for x in final_data.split('\r\n')]

        final_columns = final_columns[:-1]

        returnData["resCsv"] = final_columns

        return returnData


        
    web_app.register_blueprint( api_blueprint , url_prefix = "/api")
    return web_app

        

app = create_web_app()



if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0")
