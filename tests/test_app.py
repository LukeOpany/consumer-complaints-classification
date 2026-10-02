from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_missing_model_has_setup_message(monkeypatch,tmp_path):
    monkeypatch.setenv('ROUTING_MODEL',str(tmp_path/'missing.joblib'))
    app=AppTest.from_file(Path('routing/app.py').resolve()).run(timeout=30)
    assert not app.exception
    assert app.info


def test_review_workflow(monkeypatch,tmp_path):
    import pandas as pd
    from routing.model import train,TEXT,LABEL
    rows=[{TEXT:f'Mortgage escrow loan payment problem number {i}',LABEL:'mortgage'} for i in range(30)]
    rows += [{TEXT:f'Credit card disputed merchant transaction number {i}',LABEL:'card'} for i in range(30)]
    source=tmp_path/'input.csv';pd.DataFrame(rows).to_csv(source,index=False)
    train(source,tmp_path/'model',tmp_path/'reports')
    monkeypatch.setenv('ROUTING_MODEL',str(tmp_path/'model/routing.joblib'))
    monkeypatch.setenv('REVIEW_QUEUE',str(tmp_path/'queue.sqlite'))
    app=AppTest.from_file(Path('routing/app.py').resolve()).run(timeout=30)
    app.text_area[0].set_value('My mortgage escrow loan payment is incorrect')
    app.button[0].click().run(timeout=30)
    assert not app.exception
    next(b for b in app.button if b.label=='Save to manual-review queue').click().run(timeout=30)
    assert not app.exception
    next(b for b in app.button if b.label=='Complete review').click().run(timeout=30)
    assert not app.exception
    assert any('No pending reviews' in i.value for i in app.info)
