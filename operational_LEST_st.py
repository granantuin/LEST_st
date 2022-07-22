import numpy as np
import pandas as pd
from help_functions import get_meteogalicia_model, get_metar, get_table_download_link
import pickle
import streamlit as st
import plotly.express as px
from st_aggrid import AgGrid

st.set_page_config(page_title="LEST Machine Learning",layout="wide")
#open algorithm
alg=pickle.load(open("algorithms/vis_LEST_d0.al","rb"))

#load raw meteorological model and get model variables
meteo_model=get_meteogalicia_model(alg["coor"])


#map Vigo airport
if st.checkbox("model points map?"):
  #map
  st.write("#### **Santiago airport and WRF Meteogalicia model**") 
  px.set_mapbox_access_token("pk.eyJ1IjoiZ3JhbmFudHVpbiIsImEiOiJja3B4dGU4OTkwMTFmMm9ycnNhMjJvaGJqIn0.VWzx_PkD9A5cSUVsn_ijCA")
  dist_map=px.scatter_mapbox(alg["coor"], hover_data=['distance'],lat='lat', lon='lon',color='distance',
                             color_continuous_scale=px.colors.cyclical.IceFire,)
  st.plotly_chart(dist_map)

 
#get metar today
metar_df=get_metar("LEST")

#select x _var
model_x_var=meteo_model[:24][alg["x_var"]]

#forecast machine learning  horizontal visibility meters
vis_ml=(pd.DataFrame(alg["pipe"].predict_proba(model_x_var))).iloc[:,0].map("{:.0%}".format).values


#open new algorithm
alg=pickle.load(open("algorithms/temp_LEST_d0.al","rb"))

#select x _var
model_x_var=meteo_model[:24][alg["x_var"]]

#forecast machine learning  wind direction
temp_ml= alg["pipe"].predict(model_x_var)

#show results wind and temperature
st.write("#### **Results visibility and temperature forecast  D0**")
   
df_for0=pd.DataFrame({"time UTC":meteo_model[:24].index,
                     "visibility <=1000m (prob)":vis_ml,
                     "Temperature ml":temp_ml,
                     "Temperature WRF":round(model_x_var["temp4"]-273.16,0)})

df_all=pd.concat([df_for0.set_index("time UTC"),metar_df],axis=1).reset_index()
df_all=df_all.rename(columns={"index": "Time UTC"})
AgGrid(df_all)




#download report
with open("reports/vis_LEST_d0.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()
st.download_button(label="Download visibility report",
                    data=PDFbyte,
                    file_name="LEST_visibilityd0_report.pdf",
                    mime='application/octet-stream')
