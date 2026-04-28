import streamlit as st
import cv2
st.title("ML")
st.write("Hello world")

st.write("DL")
#a=st.number_input("Enter a number")
#st.text(a)
img=cv2.imread("SRH-logo.jpg")
st.image(img)
st.write("Resize img")
Height=st.slider("Select the Height",100,500)
Weidth=st.slider("Select the Weidth",100,500)

img1=cv2.resize(img,(Height,Weidth))
st.image(img1)