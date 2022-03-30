#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pickle 


# In[4]:


lg_model = pickle.load(open('lg_model.pkl' , 'rb'))
log_model = pickle.load(open('log_model.pkl' , 'rb'))
RFC_model = pickle.load(open('RFC_model.pkl' , 'rb'))
SVM_model = pickle.load(open('SVM_model.pkl' , 'rb'))


# In[5]:


def classification(num):
    if num<0.5:
        return 'Setosa'
    elif num<1.5:
        return 'Versicolor'
    else:
        return 'Virginica'


# In[6]:


def main():
    
    st.title("Streamlit Implementation")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Iris Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Linear Regression','Logistic Regression','SVM', 'RFC']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    sl=st.slider('Select Sepal Length', 0.0, 10.0)
    sw=st.slider('Select Sepal Width', 0.0, 10.0)
    pl=st.slider('Select Petal Length', 0.0, 10.0)
    pw=st.slider('Select Petal Width', 0.0, 10.0)
    inputs=[[sl,sw,pl,pw]]
    if st.button('Classification'):
        if option=='Linear Regression':
            st.success(classification(lg_model.predict(inputs)))
        elif option=='Logistic Regression':
            st.success(classification(log_model.predict(inputs)))
        elif option=='Support Vectore Machine':
            st.success(classification(SVM_model.predict(inputs)))
        else:
            st.success(classification(RFC_model.predict(inputs)))


if __name__=='__main__':
    main()


# In[ ]:




