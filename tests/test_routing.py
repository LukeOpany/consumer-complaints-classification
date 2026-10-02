import numpy as np
import pandas as pd
from routing.model import TEXT,LABEL,prepare,choose_threshold,route
from routing.queue import add,pending,resolve


def test_duplicate_and_conflicting_narratives_removed():
    data=[{TEXT:f'Mortgage payment narrative number {i}',LABEL:'mortgage'} for i in range(20)]
    data += [{TEXT:f'Credit card disputed narrative number {i}',LABEL:'card'} for i in range(20)]
    data += [data[0],{TEXT:data[1][TEXT],LABEL:'card'}]
    frame,_=prepare(pd.DataFrame(data),min_class=10)
    assert len(frame)==39
    assert not frame[TEXT].duplicated().any()


def test_threshold_disables_routing_without_evidence():
    threshold,enabled=choose_threshold(np.array([[.99,.01]]*30),np.array(['a','b']),['b']*30)
    assert not enabled and threshold==1


def test_queue_resolution(tmp_path):
    path=tmp_path/'queue.sqlite'
    identifier=add(path,'A public example narrative',dict(product='mortgage',score=.4))
    assert len(pending(path))==1
    resolve(path,identifier,'card')
    assert pending(path)==[]


def test_empty_input_goes_to_review():
    assert route({},'')['decision']=='manual_review'


def test_train_persist_and_route(tmp_path):
    import joblib
    from routing.model import train
    rows=[{TEXT:f'Mortgage loan escrow payment account problem number {i}',LABEL:'mortgage'} for i in range(30)]
    rows += [{TEXT:f'Credit card disputed merchant charge transaction number {i}',LABEL:'card'} for i in range(30)]
    source=tmp_path/'sample.csv';pd.DataFrame(rows).to_csv(source,index=False)
    report=train(source,tmp_path/'model',tmp_path/'reports')
    bundle=joblib.load(tmp_path/'model/routing.joblib')
    result=route(bundle,'My mortgage escrow payment is incorrect and the loan balance is wrong')
    assert result['product']=='mortgage'
    assert sum(report['split_rows'].values())==60
    assert (tmp_path/'reports/confusion-matrix.csv').exists()
    # With fewer than 20 validation records, routing must remain disabled.
    assert result['decision']=='manual_review'
