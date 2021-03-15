import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title('College Football Passing Stats Explorer')

st.markdown("""
This app performs simple webscraping of College Football player passing data.
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** [sports-reference.com/cfb/](https://www.sports-reference.com/cfb/).
* **Thanks to the Data Professor for the tutorial:** https://www.youtube.com/watch?v=zYSDlbr-8V8
""")

st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1990,2021))))

# https://www.sports-reference.com/cfb/years/2020/rushing.htm
@st.cache
def load_data(year):
    url = "https://www.sports-reference.com/cfb/years/" + str(year) + "-passing.html"
    html = pd.read_html(url, header = 1)
    df = html[0]
    print(df)
    #raw = df.drop(df[df.Age == 'Age'].index) # Deletes repeating headers in content
    #raw = raw.fillna(0)
    #playerstats = raw.drop(['Rk'], axis=1)
    playerstats = df
    return playerstats
playerstats = load_data(selected_year)

# Sidebar - Team selection
sorted_unique_team = sorted(playerstats.School.unique())
selected_team = st.sidebar.multiselect('School', sorted_unique_team, sorted_unique_team)

# Sidebar - Position selection
unique_pos = ['ACC', 'American', 'Big Ten', 'Big 12', 'CUSA', 'Ind', 'MAC', 'MWC', 'Pac-12', 'SEC', 'Sun Belt']
selected_pos = st.sidebar.multiselect('Conference', unique_pos, unique_pos)

# Filtering data
df_selected_team = playerstats[(playerstats.School.isin(selected_team)) & (playerstats.Conf.isin(selected_pos))]

st.header('Display Player Stats of Selected Team(s)')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
st.dataframe(df_selected_team)

# Download NBA player stats data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

# Heatmap
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    st.set_option('deprecation.showPyplotGlobalUse', False)
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot()
