from flask import Flask , render_template , request
import numpy as np
import pickle
import json
import pandas as pd
# from sklearn.datasets import make_classification
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.tree import DecisionTreeClassifier


app = Flask(__name__)
#model = pickle.load(open(r'C:\Users\chirag.uthra\Downloads\kagglemodel.pkl' , 'rb'))
model = pickle.load(open('./kagglemodel.pkl' , 'rb'))





@app.route('/')
def index():
    return render_template('Kaggle_HTML_new.html')


@app.route('/predict' , methods=['POST'])
def predict():
    values = [x for x in request.form.values()]
    final = {}
    final['city'] = values[0]
    final['gender'] = values[1]
    final['enrolled_university'] = values[2]
    final['education_level'] = values[3]
    final['major_discipline'] = values[4]
    final['experience'] = values[5]
    final['company_size'] = values[6]
    final['company_type'] = values[7]
    final['last_new_job'] = values[8]

    trained_feature_list = ['enrollee_id', 'city_development_index', 'training_hours',
       'city_city_100', 'city_city_102', 'city_city_103', 'city_city_104',
       'city_city_11', 'city_city_114', 'city_city_136', 'city_city_16',
       'city_city_160', 'city_city_21', 'city_city_23', 'city_city_28',
       'city_city_36', 'city_city_61', 'city_city_65', 'city_city_67',
       'city_city_71', 'city_city_73', 'city_city_75', 'city_city_90',
       'gender_Male', 'gender_OTHERS', 'gender_Other',
       'relevent_experience_No relevent experience',
       'enrolled_university_Part time course',
       'enrolled_university_no_enrollment', 'education_level_High School',
       'education_level_Masters', 'education_level_OTHERS',
       'education_level_Phd', 'education_level_Primary School',
       'major_discipline_Business Degree', 'major_discipline_Humanities',
       'major_discipline_No Major', 'major_discipline_Other',
       'major_discipline_STEM', 'experience_1', 'experience_10',
       'experience_11', 'experience_12', 'experience_13', 'experience_14',
       'experience_15', 'experience_16', 'experience_17', 'experience_18',
       'experience_19', 'experience_2', 'experience_20', 'experience_3',
       'experience_4', 'experience_5', 'experience_6', 'experience_7',
       'experience_8', 'experience_9', 'experience_<1', 'experience_>20',
       'company_size_10/49', 'company_size_100-500', 'company_size_1000-4999',
       'company_size_10000+', 'company_size_50-99', 'company_size_500-999',
       'company_size_5000-9999', 'company_size_<10',
       'company_type_Funded Startup', 'company_type_NGO',
       'company_type_OTHERS', 'company_type_Other',
       'company_type_Public Sector', 'company_type_Pvt Ltd', 'last_new_job_1',
       'last_new_job_2', 'last_new_job_3', 'last_new_job_4',
       'last_new_job_>4']
    
    df = pd.DataFrame(final , index = [0])  

    df.loc[~df["city"].isin(['city_103','city_21' , 'city_16']), "city"] = "city_103"
    df.loc[~df["gender"].isin(['Male','Female']), "gender"] = "OTHERS"
    df.loc[~df["enrolled_university"].isin(['Full time course','enrolled_university']), "enrolled_university"] = "no_enrollment"
    df.loc[~df["education_level"].isin(['Graduate','Masters' , 'High_School' , 'Phd' , 'Primary_School']), "education_level"] = "OTHERS"
    df.loc[~df["major_discipline"].isin(['STEM','Humanities' , 'Business Degree' , 'Arts' , 'No Major']), "major_discipline"] = "Other"
    df.loc[~df["experience"].isin(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18']), "experience"] = "OTHERS"
    df.loc[~df["company_size"].isin(['10/49','50-99']), "company_size"] = "5000-9999"
    df.loc[~df["company_type"].isin(['Pvt Ltd','Funded Startup']), "company_type"] = "OTHERS"
    df.loc[~df["last_new_job"].isin(['1','2']), "last_new_job"] = "1"  
    
    df_final = pd.get_dummies(df)
    
    delta_list = set(trained_feature_list) - set(df_final.columns)

    for l in delta_list:
        df_final[l] = 0

    pred = model.predict(df_final)

    for i in pred.tolist():
        if i == 0:
            return 'Will not Stay'
        else:
            return 'Will Stay'

#    return pred.tolist()

#    pred_op = model.predict(features)



if __name__ == '__main__':
    app.run(port = '5001' , debug = True)