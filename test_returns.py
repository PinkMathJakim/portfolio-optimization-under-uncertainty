import numpy as np, pandas as pd
from src.returns import simple_returns, log_returns, annualized_volatility
def test_return_calculations():
 p=pd.DataFrame({'A':[100,110,99]})
 assert np.isclose(simple_returns(p).iloc[1,0],.1)
 assert np.isclose(log_returns(p).iloc[1,0],np.log(1.1))
def test_annualized_volatility_positive():
 assert annualized_volatility(pd.Series([-.01,.01,-.02,.02]))>0
