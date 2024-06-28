import streamlit as st
from .constructor.dashboard_constructor import dashboard_constructor
from .constructor.prediction_constructor import predict_constructor


def start() -> None:
    st.set_page_config(layout='wide', page_title='Stock Analysis', page_icon=':dollar:')
    #while True:
    page = st.sidebar.radio('Pages', ['Dashboard', 'Prediction'])
    if page == 'Dashboard':
        dashboard_constructor()
    elif page == 'Prediction':
        predict_constructor()
    