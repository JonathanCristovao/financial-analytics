import streamlit as st
from src.controllers.prediction_controller import PredictionController

class PredictionView:
    def __init__(self):
        self.controller = PredictionController()

    def display(self):
        self.controller.run()
