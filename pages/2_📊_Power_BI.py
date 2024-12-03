import streamlit as st

import streamlit.components.v1 as components

st.set_page_config(page_title="Power BI", page_icon="📊")

st.markdown("# Power BI")


components.iframe("https://app.powerbi.com/view?r=eyJrIjoiNmE0NmYwOGYtZTQ1OS00ZDY1LThkNTctMGIxNTViMjFmMjI5IiwidCI6IjFjZWQ5ZDZjLWM2OTktNGMxYS1hYjlkLTI1YWY1MzI0ZGMyYSJ9",width=800,height=600,scrolling=False)

