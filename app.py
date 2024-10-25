import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
import time
import streamlit as st
from db import *

pickleFile=open("weights.pkl","rb")
regressor=pickle.load(pickleFile) # our model

# **2. Loading Dataset**

df = pd.read_csv(r'mldata.csv')

df.head()


def coding_quiz():
    st.subheader("Coding Skills Quiz")

    # Quiz questions
    questions = {
        "Question 1: What is the time complexity of binary search?": {
            "options": ["O(n)", "O(log n)", "O(n^2)", "O(n log n)"],
            "answer": "O(log n)"
        },
        "Question 2: What does 'OOP' stand for in programming?": {
            "options": ["Objective-Oriented Programming", "Object-Oriented Programming", "Operating System Oriented Programming", "Open-Oriented Programming"],
            "answer": "Object-Oriented Programming"
        },
        "Question 3: Which data structure works as FIFO (First In First Out)?": {
            "options": ["Stack", "Queue", "Tree", "Graph"],
            "answer": "Queue"
        }
    }

    # Initialize or retrieve session state for storing answers
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
        st.session_state.correct_answers = 0
    
    # Display quiz if not submitted
    if not st.session_state.quiz_submitted:
        correct_answers = 0
        for question, data in questions.items():
            user_answer = st.radio(question, data["options"], key=question)
            if user_answer == data["answer"]:
                correct_answers += 1

        if st.button("Submit Coding Quiz"):
            st.session_state.quiz_submitted = True
            st.session_state.correct_answers = correct_answers

    # Calculate and show the score if the quiz is submitted
    if st.session_state.quiz_submitted:
        quiz_score = st.session_state.correct_answers / len(questions) * 10
        st.success(f'Your Coding Skills Score: {quiz_score} / 10')

def logical_quotient_quiz():
    st.subheader("Logical Quotient Quiz")

    # Quiz questions
    questions = {
        "Question 1: If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies?": {
            "options": ["Yes", "No", "Maybe", "Cannot Say"],
            "answer": "Yes"
        },
        "Question 2: A clock shows the time as 3:00. If the minute hand moves 90 degrees, what is the time?": {
            "options": ["3:15", "3:30", "4:30", "4:15"],
            "answer": "3:15"
        },
        "Question 3: Which number comes next in the series: 1, 4, 9, 16, 25, ___?": {
            "options": ["30", "36", "49", "40"],
            "answer": "36"
        }
    }

    # Initialize or retrieve session state for logical quiz
    if 'logical_quiz_submitted' not in st.session_state:
        st.session_state.logical_quiz_submitted = False
        st.session_state.logical_correct_answers = 0

    # Display logical quiz if not submitted
    if not st.session_state.logical_quiz_submitted:
        logical_correct_answers = 0
        for question, data in questions.items():
            user_answer = st.radio(question, data["options"], key=question + "_logical")
            if user_answer == data["answer"]:
                logical_correct_answers += 1

        if st.button("Submit Logical Quiz"):
            st.session_state.logical_quiz_submitted = True
            st.session_state.logical_correct_answers = logical_correct_answers

    # Calculate and show the score if the logical quiz is submitted
    if st.session_state.logical_quiz_submitted:
        logical_quiz_score = st.session_state.logical_correct_answers / len(questions) * 10
        st.success(f'Your Logical Quotient Score: {logical_quiz_score} / 10')
df['workshops'] = df['workshops'].replace(['testing'],'Testing')
df.head()

print(df.columns.unique)

n = df['Suggested Job Role'].unique()
print(len(n))

print('The shape of our training set: %s professionals and %s features'%(df.shape[0],df.shape[1]))


# **5. Feature Engineering**

## (a) Binary Encoding for Categorical Variables

newdf = df
newdf.head(10)

cols = df[["self-learning capability?", "Extra-courses did","Taken inputs from seniors or elders", "worked in teams ever?", "Introvert"]]
for i in cols:
    print(i)
    cleanup_nums = {i: {"yes": 1, "no": 0}}
    df = df.replace(cleanup_nums)

print("\n\nList of Categorical features: \n" , df.select_dtypes(include=['object']).columns.tolist())

## (b) Number Encoding for Categorical 

mycol = df[["reading and writing skills", "memory capability score"]]
for i in mycol:
    print(i)    
    cleanup_nums = {i: {"poor": 0, "medium": 1, "excellent": 2}}
    df = df.replace(cleanup_nums)

category_cols = df[['certifications', 'workshops', 'Interested subjects', 'interested career area ', 'Type of company want to settle in?', 
                    'Interested Type of Books']]
for i in category_cols:
    df[i] = df[i].astype('category')
    df[i + "_code"] = df[i].cat.codes

print("\n\nList of Categorical features: \n" , df.select_dtypes(include=['object']).columns.tolist())

## (c) Dummy Variable Encoding

print(df['Management or Technical'].unique())
print(df['hard/smart worker'].unique())

df = pd.get_dummies(df, columns=["Management or Technical", "hard/smart worker"], prefix=["A", "B"])
df.head()

df.sort_values(by=['certifications'])

print("List of Numerical features: \n" , df.select_dtypes(include=np.number).columns.tolist())


category_cols = df[['certifications', 'workshops', 'Interested subjects', 'interested career area ', 'Type of company want to settle in?', 'Interested Type of Books']]
for i in category_cols:
  print(i)

Certifi = list(df['certifications'].unique())
print(Certifi)
certi_code = list(df['certifications_code'].unique())
print(certi_code)

Workshops = list(df['workshops'].unique())
print(Workshops)
Workshops_code = list(df['workshops_code'].unique())
print(Workshops_code)

Certi_l = list(df['certifications'].unique())
certi_code = list(df['certifications_code'].unique())
C = dict(zip(Certi_l,certi_code))

Workshops = list(df['workshops'].unique())
print(Workshops)
Workshops_code = list(df['workshops_code'].unique())
print(Workshops_code)
W = dict(zip(Workshops,Workshops_code))

Interested_subjects = list(df['Interested subjects'].unique())
print(Interested_subjects)
Interested_subjects_code = list(df['Interested subjects_code'].unique())
ISC = dict(zip(Interested_subjects,Interested_subjects_code))

interested_career_area = list(df['interested career area '].unique())
print(interested_career_area)
interested_career_area_code = list(df['interested career area _code'].unique())
ICA = dict(zip(interested_career_area,interested_career_area_code))

Typeofcompany = list(df['Type of company want to settle in?'].unique())
print(Typeofcompany)
Typeofcompany_code = list(df['Type of company want to settle in?_code'].unique())
TOCO = dict(zip(Typeofcompany,Typeofcompany_code))

Interested_Books = list(df['Interested Type of Books'].unique())
print(Interested_subjects)
Interested_Books_code = list(df['Interested Type of Books_code'].unique())
IB = dict(zip(Interested_Books,Interested_Books_code))

Range_dict = {"poor": 0, "medium": 1, "excellent": 2}
print(Range_dict)


A = 'yes'
B = 'No'
col = [A,B]
for i in col:
  if(i=='yes'):
    i = 1
  print(i)


f =[]
A = 'r programming'
clms = ['r programming',0]
for i in clms:
  for key in C:
    if(i==key):
      i = C[key]
      f.append(i)
print(f)

C = dict(zip(Certifi,certi_code))
  
print(C)

import numpy as np
array = np.array([1,2,3,4])
array.reshape(-1,1)

def inputlist(Name,Contact_Number,Email_address,
      Logical_quotient_rating, coding_skills_rating, hackathons, 
      public_speaking_points, self_learning_capability, 
      Extra_courses_did, Taken_inputs_from_seniors_or_elders,
      worked_in_teams_ever,Introvert, reading_and_writing_skills,
      memory_capability_score, smart_or_hard_work, Management_or_Techinical,
      Interested_subjects, Interested_Type_of_Books,certifications, workshops, 
      Type_of_company_want_to_settle_in, interested_career_area):
  #1,1,1,1,'Yes','Yes''Yes''Yes''Yes',"poor","poor","Smart worker", "Management","programming","Series","information security"."testing","BPA","testing"
  Afeed = [Logical_quotient_rating, coding_skills_rating, hackathons, public_speaking_points]

  input_list_col = [self_learning_capability,Extra_courses_did,Taken_inputs_from_seniors_or_elders,worked_in_teams_ever,Introvert,reading_and_writing_skills,memory_capability_score,smart_or_hard_work,Management_or_Techinical,Interested_subjects,Interested_Type_of_Books,certifications,workshops,Type_of_company_want_to_settle_in,interested_career_area]
  feed = []
  K=0
  j=0
  for i in input_list_col:
    if(i=='Yes'):
      j=2
      feed.append(j)
       
      print("feed 1",i)
    
    elif(i=="No"):
      j=3
      feed.append(j)
       
      print("feed 2",j)
    
    elif(i=='Management'):
      j=1
      k=0
      feed.append(j)
      feed.append(K)
       
      print("feed 10,11",i,j,k)

    elif(i=='Technical'):
      j=0
      k=1
      feed.append(j)
      feed.append(K)
       
      print("feed 12,13",i,j,k)

    elif(i=='Smart worker'):
      j=1
      k=0
      feed.append(j)
      feed.append(K)
       
      print("feed 14,15",i,j,k)

    elif(i=='Hard Worker'):
      j=0
      k=1
      feed.append(j)
      feed.append(K)
      print("feed 16,17",i,j,k)
    
    else:
      for key in Range_dict:
        if(i==key):
          j = Range_dict[key]
          feed.append(j)
         
          print("feed 3",i,j)

      for key in C:
        if(i==key):
          j = C[key]
          feed.append(j)
          
          print("feed 4",i,j)
      
      for key in W:
        if(i==key):
          j = W[key]
          feed.append(j)
          
          print("feed 5",i,j)
      
      for key in ISC:
        if(i==key):
          j = ISC[key]
          feed.append(j)
          
          print("feed 6",i,j)

      for key in ICA:
        if(i==key):
          j = ICA[key]
          feed.append(j)
          
          print("feed 7",i,j)

      for key in TOCO:
        if(i==key):
          j = TOCO[key]
          feed.append(j)
          
          print("feed 8",i,j)

      for key in IB:
        if(i==key):
          j = IB[key]
          feed.append(j)
          
          print("feed 9",i,j)

   
       
  t = Afeed+feed    
  output = regressor.predict([t])
  
  return(output)

def main():

  # with st.spinner('Wait for it...'):
  #     time.sleep(5)
  # st.success('Done!')

  html1="""
    <div style="text-align:center; text-shadow: 3px 1px 2px purple;">
      <h1> NextStep 📈</h1>
    </div>
      """
  st.markdown(html1,unsafe_allow_html=True) #simple html 

  # Images


  html2="""
    <div style="text-align:center; text-shadow: 3px 1px 2px purple;">
      <h2>Your friendly Career Compass<h2>
    </div>
      """
  st.markdown(html2,unsafe_allow_html=True) #simple html 
 
  st.sidebar.title("Your Information")

  Name = st.sidebar.text_input("Full Name")

  Contact_Number = st.sidebar.text_input("Contact Number")

  Email_address = st.sidebar.text_input("Email address")

  if not Name and Email_address:
    st.sidebar.warning("Please fill out your name and EmailID")

  if Name and Contact_Number and Email_address:
    st.sidebar.success("Thanks!")

  Logical_quotient_rating = logical_quotient_quiz()

  coding_skills_rating = coding_quiz()

  hackathons = st.slider(
    'Enter number of Hackathons participated',0,10,1)
  st.write(hackathons)

  public_speaking_points = st.slider(
    'Rate Your Public Speaking', 0,10,1)
  st.write(public_speaking_points)

  self_learning_capability = st.selectbox(
    'Self Learning Capability',
    ('Yes', 'No')
    )
  # st.write('You selected:', self_learning_capability)

  Extra_courses_did = st.selectbox(
    'Extra courses',
  ('Yes', 'No')
  )
  # st.write('You selected:', Extra_courses_did)

  Taken_inputs_from_seniors_or_elders = st.selectbox(
    'Took advice from seniors or elders',
    ('Yes', 'No')
    )
  # st.write('You selected:', Taken_inputs_from_seniors_or_elders)

  worked_in_teams_ever = st.selectbox(
    'Team Co-ordination Skill',
    ('Yes', 'No')
    )
  # st.write('You selected:', worked_in_teams_ever)

  Introvert = st.selectbox(
    'Introvert',
    ('Yes', 'No')
    )
  # st.write('You selected:', Introvert)

  reading_and_writing_skills = st.selectbox(
    'Reading and writing skills',
    ('poor','medium','excellent')
    )
  st.write('You selected: **{}**' .format(reading_and_writing_skills))

  memory_capability_score = st.selectbox(
    'Memory capability score',
    ('poor','medium','excellent')
    )
  st.write('You selected: **{}**' .format(memory_capability_score))

  smart_or_hard_work = st.selectbox(
    'Smart or Hard Work',
    ('Smart worker', 'Hard Worker')
    )
  st.write('You selected: **{}**' .format(smart_or_hard_work))

  Management_or_Techinical = st.selectbox(
    'Management or Techinical',
    ('Management', 'Technical')
    )
  st.write('You selected: **{}**' .format(Management_or_Techinical))

  Interested_subjects = st.selectbox(
    'Interested Subjects',
    ('programming', 'Management', 'data engineering', 'networks', 'Software Engineering', 'cloud computing', 'parallel computing', 'IOT', 'Computer Architecture', 'hacking')
    )
  st.write('You selected: **{}**' .format(Interested_subjects))

  Interested_Type_of_Books = st.selectbox(
    'Interested Books Category',
    ('Series', 'Autobiographies', 'Travel', 'Guide', 'Health', 'Journals', 'Anthology', 'Dictionaries', 'Prayer books', 'Art', 'Encyclopedias', 'Religion-Spirituality', 'Action and Adventure', 'Comics', 'Horror', 'Satire', 'Self help', 'History', 'Cookbooks', 'Math', 'Biographies', 'Drama', 'Diaries', 'Science fiction', 'Poetry', 'Romance', 'Science', 'Trilogy', 'Fantasy', 'Childrens', 'Mystery')
    )
  st.write('You selected: **{}**' .format(Interested_Type_of_Books))

  certifications = st.selectbox(
    'Interested_Type_of_Books',
    ('information security', 'shell programming', 'r programming', 'distro making', 'machine learning', 'full stack', 'hadoop', 'app development', 'python')
    )
  st.write('You selected: **{}**' .format(certifications))

  workshops = st.selectbox(
    'Workshops Attended',
    ('Testing', 'database security', 'game development', 'data science', 'system designing', 'hacking', 'cloud computing', 'web technologies')
    )
  st.write('You selected: **{}**' .format(workshops))
  
  Type_of_company_want_to_settle_in = st.selectbox(
    'Type of Company You Want to Settle In ',
    ('BPA', 'Cloud Services', 'product development', 'Testing and Maintainance Services', 'SAaS services', 'Web Services', 'Finance', 'Sales and Marketing', 'Product based', 'Service Based')
    )
  st.write('You selected: **{}**' .format(Type_of_company_want_to_settle_in))
  
  interested_career_area = st.selectbox(
    'Interested Career Area',
    ('testing', 'system developer', 'Business process analyst', 'security', 'developer', 'cloud computing')
    )
  st.write('You selected: **{}**' .format(interested_career_area))
  
  result=""
  
  if st.button("Predict"):
    result=inputlist(Name,Contact_Number,Email_address,Logical_quotient_rating, coding_skills_rating, hackathons, 
                    public_speaking_points, self_learning_capability,Extra_courses_did, 
                     Taken_inputs_from_seniors_or_elders,worked_in_teams_ever, Introvert,
                     reading_and_writing_skills,memory_capability_score, smart_or_hard_work, 
                     Management_or_Techinical,Interested_subjects, Interested_Type_of_Books,
                     certifications, workshops, Type_of_company_want_to_settle_in, interested_career_area) 

    # Progress bar
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.05)
        my_bar.progress(percent_complete + 1)

    # Balloons
    st.balloons()

    #result will be displayed if button is pressed
    st.success("Predicted Career Option : "
               "{}".format(result))

    # Plot
    corr = df[['Logical quotient rating', 'hackathons', 
           'coding skills rating', 'public speaking points']].corr()
    f,axes = plt.subplots(1,1,figsize = (10,10))
    sns.heatmap(corr,square=True,annot = True,linewidth = .4,center = 2,ax = axes)
    st.subheader("Here are some nerdy analytics 😁")
    st.text("Correlation Between Numerical Features")
    st.pyplot(f)

    # Expander
    with st.expander("See explanation"):
     st.write("""
         The plot above shows the correlation of the features.
         As we can see, no highly correlated pair is found!
     """)

    create_table()
    add_data(Name,Contact_Number,Email_address,Logical_quotient_rating, coding_skills_rating, hackathons, 
            public_speaking_points, self_learning_capability,Extra_courses_did, 
            Taken_inputs_from_seniors_or_elders,worked_in_teams_ever, Introvert,
            reading_and_writing_skills,memory_capability_score, smart_or_hard_work, 
            Management_or_Techinical,Interested_subjects, Interested_Type_of_Books,
            certifications, workshops, Type_of_company_want_to_settle_in, interested_career_area)

  # if choice == "Add Post":
  #     st.subheader("Add Your Article")
  #     create_table()
  #     blog_title = st.text_input('Enter Post Title')
  #     blog_author = st.text_input("Enter Author Name",max_chars=50)
  #     blog_article = st.text_area("Enter Your Message",height=200)
  #     blog_post_date = st.date_input("Post Date")
  #     if st.button("Add"):
  #       add_data(blog_author,blog_title,blog_article,blog_post_date)
  #       st.success("Post::'{}' Saved".format(blog_title))

 
if __name__=='__main__':
    main()
