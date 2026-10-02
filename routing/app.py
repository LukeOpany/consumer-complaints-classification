import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import streamlit as st
from routing.model import route
from routing.queue import add,pending,resolve

st.set_page_config(page_title='Complaint routing',layout='wide')
st.title('Complaint routing')
st.caption('Financial product suggestions with a local human-review queue')
model_path=Path(os.getenv('ROUTING_MODEL','artifacts/routing.joblib'))
queue_path=os.getenv('REVIEW_QUEUE','data/review-queue.sqlite')
if not model_path.exists():
    st.info('Train the model first: python -m routing.train --csv data/complaints.csv');st.stop()
# Only load a locally trained/trusted bundle; the UI does not accept uploaded pickle files.
bundle=joblib.load(model_path)
if not bundle['auto_enabled']: st.info('Validation did not establish a routing threshold. All submissions require review.')
left,right=st.columns([3,2])
with left:
    with st.form('classify'):
        text=st.text_area('Complaint narrative',height=180)
        submitted=st.form_submit_button('Suggest product')
    if submitted: st.session_state['result']=(text,route(bundle,text))
    if 'result' in st.session_state:
        original,result=st.session_state['result']
        st.write('Suggested product:',result['product'] or 'Not available')
        if result['score'] is not None: st.metric('Model score',f"{result['score']:.2f}")
        st.write('Decision:',result['decision'].replace('_',' '));st.caption(result['reason'])
        if st.button('Save to manual-review queue',disabled=len(original.strip())<20):
            identifier=add(queue_path,original,result);st.success(f'Saved as review #{identifier}')
            del st.session_state['result']
with right:
    st.subheader('How to interpret the result')
    st.write('The score is a model output, not a calibrated probability. Product coverage is limited to the classes present in the training report.')
    st.caption('Saving a review stores the narrative in a local SQLite database. This prototype has no shared-user authentication; use public or synthetic examples.')
st.subheader('Pending manual reviews')
rows=pending(queue_path)
if not rows: st.info('No pending reviews.')
else:
    st.dataframe([{k:v for k,v in row.items() if k!='narrative'} for row in rows],hide_index=True)
    selection=st.selectbox('Review ID',[row['id'] for row in rows])
    record=next(row for row in rows if row['id']==selection)
    st.text(record['narrative'])
    product=st.selectbox('Reviewed product',list(bundle['pipeline'].classes_)+['Out of scope / other'])
    if st.button('Complete review'):
        resolve(queue_path,selection,product);st.rerun()
