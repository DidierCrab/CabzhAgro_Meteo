import os
import glob
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import folium
import matplotlib.pyplot as plt
from threading import RLock

from streamlit_folium import st_folium

st.set_page_config(layout="wide")

# Cette fonction ne sera exécutée qu'une seule fois, puis mise en cache
@st.cache_data
def load_dataset(path_file_xz):
    # Opération coûteuse, comme charger un grand CSV
    return pd.read_csv(path_file_xz,compression="xz",delimiter=';')

#d0 = lambda x:x if x>0 else 0
#d6 = lambda x:0 if x<=6 else (x-6 if x<=30 else 24) 

for j in ["stations","stations_df","tntxm","rr_cumul","champs_t","station_t","champs_rr","station_rr"]:
    if j not in st.session_state:
        exec(f"st.session_state.{j}=[]")

#print(st.__version__)

min_date=datetime.datetime(2024,1,1)
max_date=datetime.date.today()

# Récupération des données
rep_data = r"C:\Users\debroizd35r\Documents\Meteo_Donnees"

tntxm = load_dataset("https://calcul-rsh-bretagne.com/data/TNTXM_TN_TM.csv.xz")
print(len(tntxm))
#st.session_state.tntxm = pd.read_csv(os.path.join(rep_data,"TNTXM_TN_TM.csv"),delimiter=';')
#st.session_state.tntxm = pd.read_csv("https://calcul-rsh-bretagne.com/data/TNTXM_TN_TM.csv",delimiter=';')

#st.session_state.tntxm['STA'] = st.session_state.tntxm["CITY_LATLON"].apply(lambda x:x.split('_(')[0])
tntxm['DOB'] = tntxm["AAAAMMJJ"].apply(lambda x:datetime.date(int(str(x)[:4]),int(str(x)[4:6]),int(str(x)[6:8])))

st.session_state.rr_cumul = pd.read_csv(os.path.join(rep_data,"RR_RR_cumul.csv"),delimiter=';')
#st.session_state.rr_cumul = pd.read_csv("https://calcul-rsh-bretagne.com/data/RR_RR_cumul.csv",delimiter=';')
#st.session_state.rr_cumul['STA'] = st.session_state.rr_cumul["CITY_LATLON"].apply(lambda x:x.split('_(')[0])
st.session_state.rr_cumul['DOB'] = st.session_state.rr_cumul["AAAAMMJJ"].apply(lambda x:datetime.date(int(str(x)[:4]),int(str(x)[4:6]),int(str(x)[6:8])))



col1, col2 = st.columns([1,8]) 
with col1:
    st.image("https://calcul-rsh-bretagne.com/img/logo_moyen.jpg", width=100)#,size="large")#os.path.join(rep,'logocrab_blanc.pnt'))
with col2:
    st.title("Données météofrance")
    st.write("Valorisation des données mises à disposition par Météofrance via meteo.data.gouv.fr")

st.subheader("Carte des stations météo disponibles")
st.session_state.stations_df=pd.read_csv("https://calcul-rsh-bretagne.com/data/station_dep.csv",delimiter=';')
st.session_state.stations=[str(sta['Dep'])+"_"+sta['Station'] for n,sta in st.session_state.stations_df.iterrows()]


m = folium.Map(location=[48., -2.7], zoom_start=8.2)#,control_scale=True) 

for n,sta in st.session_state.stations_df.iterrows():
    folium.Marker((float(sta['Lat']),float(sta['Lon'])), tooltip=str(sta['Dep'])+"_"+sta['Station']).add_to(m)
 
st_map=st_folium(m,width=800)

#col1, col2, col3,col4 = st.columns([1,1,0.2,1])

col1, right_section = st.columns([1, 2])

with col1:
    with st.form(key = 'Temp'):
        st.subheader('Graphique des températures')

        st.session_state.station = st.selectbox("Choisissez la station météo",st.session_state.stations)

        #st.subheader('choix_Date')
        dob = st.date_input("Choisissez la date de début",min_value = min_date,max_value = max_date)
        doe = st.date_input("Choisissez la date de fin",min_value = min_date,max_value = max_date)

        ## Champs à choisir pour insertion dans le graphique
        #st.session_state.champs_t = st.multiselect("Quels données insérer dans le graphique", ['TNTXM', 'N_TNTXM_size','N_TNTXM_median', 'N_TNTXM_mean', 'N_TNTXM_q1', 'N_TNTXM_q3'])
        st.session_state.champs_t = st.multiselect("Choisissez quelles données insérer dans le graphique", ['TNTXM', 'N_TNTXM_median', 'N_TNTXM_mean',       'N_TNTXM_d2', 'N_TNTXM_d8', 'TN','N_TNTXM_size',  'N_TN_size', 'N_TN_median', 'N_TN_mean', 'N_TN_d2', 'N_TN_d8', 'TM', 'N_TM_size', 'N_TM_median', 'N_TM_mean', 'N_TM_d2', 'N_TM_d8'])
        #print(type(champs))
        #st.write("You selected:", options)

        st.form_submit_button()

with right_section:
    col_chart, col_data = st.columns([1.2, 0.8],gap='large')
    with col_chart:
        chart_data=tntxm[(tntxm['STA']==st.session_state.station) & (tntxm['DOB']>=dob) & (tntxm['DOB']<=doe)][["DOB"]+st.session_state.champs_t]
        st.write(chart_data)


    with col_data:
        #st.title("Chart Data")
        #chart_data=pd.DataFrame(np.random.randn(100,2), columns=['lat','lon'])
        data_cumult=tntxm[(tntxm['STA']==st.session_state.station) & (tntxm['DOB']>=dob) & (tntxm['DOB']<=doe)]['TNTXM']

        #chart_data=st.session_state.tntxm[(st.session_state.tntxm['STA']==st.session_state.station) & (st.session_state.tntxm['DOB']>=dob) & (st.session_state.tntxm['DOB']<=doe)][["DOB"]+st.session_state.champs_t]
        #data_cumult=st.session_state.tntxm[(st.session_state.tntxm['STA']==st.session_state.station) & (st.session_state.tntxm['DOB']>=dob) & (st.session_state.tntxm['DOB']<=doe)]['TNTXM']
        #chart_data = chart_data.set_index('DOB')
        #chart_data = chart_data.rename(columns={'index': 'x'})
        #chart_data=tntxm[(tntxm['STA']==station)]['TNTXM']

        #st.subheader('Graphique')
        #st.line_chart(chart_data)#,y_label="Evolution des températures : "

        #source = pd.DataFrame(chart_data,columns=st.session_state.champs_t, index=pd.RangeIndex(len(chart_data), name='x'))
        #source = source.reset_index().melt('x', var_name='Type', value_name='y')
        source = pd.melt(chart_data,id_vars=['DOB'],value_vars=st.session_state.champs_t, var_name='Type', value_name='y')   

        line_chart = alt.Chart(source).mark_line(interpolate='basis').encode(
            alt.X('DOB', title='Date'),
            alt.Y('y', title='Température'),
            color='Type:N').properties(title='Evolution des températures : '+st.session_state.station,width=640,height=480)
        st.altair_chart(line_chart)   


st.write("Cumul des températures base 0 entre le ",f"{dob:%d/%m/%Y}"," et le ",f"{doe:%d/%m/%Y}"," : ",f"{data_cumult.apply(lambda x:x if x>0 else 0).sum():0.1f}", "°.")
st.write("Cumul des températures base 6 entre le ",f"{dob:%d/%m/%Y}"," et le ",f"{doe:%d/%m/%Y}"," : ",f"{data_cumult.apply(lambda x:0 if x<=6 else (x-6 if x<=30 else 24)).sum():0.1f}", "°.")
 
#col3, col4 = st.columns(2)
col1, col2, col3,col4 = st.columns([1,1,0.2,1])


with col1:
    with st.form(key = 'Pluvio'):
        st.subheader('Graphique des pluviométries')

        st.session_state.station_rr = st.selectbox("Choisissez la station météo",st.session_state.stations)

        #st.subheader('choix_Date')
        dob = st.date_input("Choisissez la date de début",min_value = min_date,max_value = max_date)
        doe = st.date_input("Choisissez la date de fin",min_value = min_date,max_value = max_date)

        ## Champs à choisir pour insertion dans le graphique
        st.session_state.champs_rr = st.multiselect("Choisissez quelles données insérer dans le graphique", ['RR','RR_cumul','N_RR_size','N_RR_median','N_RR_mean','N_RR_d2','N_RR_d8','N_RR_cumul_size','N_RR_cumul_median','N_RR_cumul_mean','N_RR_cumul_d2','N_RR_cumul_d8'])
        #print(type(champs))
        #st.write("You selected:", options)

        st.form_submit_button()
with col2:
    #chart_data=tntxm[(tntxm['STA']==st.session_state.station) & (tntxm['DOB']>=dob) & (tntxm['DOB']<=doe)][["DOB"]+st.session_state.champs_t]

    #st.subheader('Graphique')
    rr_df=st.session_state.rr_cumul[(st.session_state.rr_cumul['STA']==st.session_state.station_rr) & (st.session_state.rr_cumul['DOB']>=dob - datetime.timedelta(days=1)) & (st.session_state.rr_cumul['DOB']<=doe)][["DOB","MMJJ"]+st.session_state.champs_rr]#.reset_index()
    #print(st.session_state.rr_cumul.columns,rr_df.columns)
    for ch in st.session_state.champs_rr:
        if '_cumul' in ch:
            if 1230 not in rr_df['MMJJ'].values:# or "N_" not in ch:
                rr_df[ch]=rr_df[ch]-rr_df[ch].iloc[0]
                #print(rr_df['MMJJ'].iloc[3])
            else:
                #if "N_" in ch:
                print('ok -----------')
                #print(type(rr_df[rr_df['MMJJ']==1231]['DOB'].iloc[0]))
                date1231=rr_df[rr_df['MMJJ']==1231]['DOB'].iloc[0]
                rr_df[ch]=rr_df.apply(lambda x:x[ch]-rr_df[ch].iloc[0] if x['DOB']<=date1231 else x[ch]-rr_df[ch].iloc[0]+rr_df[rr_df['DOB']==date1231][ch].iloc[0],axis=1)
                print(ch,rr_df[ch].iloc[0],rr_df[rr_df['DOB']==date1231][ch],rr_df[ch].iloc[0]+rr_df[rr_df['DOB']==date1231][ch].iloc[0],date1231)
                #rr_df[ch]=rr_df[ch]-rr_df[ch].iloc[0]

    rr_df=rr_df[rr_df['DOB']>=dob]        
    _lock = RLock()
    #x,y=[1,2,3],[4,5,6]
    #with _lock:
    #    fig, ax = plt.subplots()
    #    ax.scatter(x, y)
    #    st.pyplot(fig)

    with _lock:
        fig, ax = plt.subplots()
        for ch in st.session_state.champs_rr:
            if '_cumul' in ch:
                    ax.plot(rr_df['DOB'], rr_df[ch],label=ch)
            elif '_cumul' not in ch:
                    ax.bar(rr_df['DOB'], rr_df[ch],label=ch)

        #ax.set_xticks(rr_df['DOB'], rr_df['DOB'], rotation=45)
        ax.tick_params("x", rotation=45)
        #ax.tick_params(axis='x', rotation=45, labelright=False, labelbottom=True, ha='right')
        for tick in ax.get_xticklabels():
           tick.set_horizontalalignment('right')

        ax.set_xlabel("Date")
        ax.set_ylabel("Pluviométrie : mm")
        ax.legend()
        ax.set_title('Pluviométrie : '+st.session_state.station_rr)
        st.pyplot(fig)

with col3:
    st.empty()

with col4:
    st.write(rr_df)
