import numpy as np
from src.optimization import equal_weight, minimum_variance, max_sharpe, risk_parity
def test_constraints_all_strategies():
 cov=np.array([[.04,.01,.005],[.01,.09,.01],[.005,.01,.06]])
 mu=np.array([.0005,.0007,.0004]); cap=.5
 for w in [equal_weight(3,cap),minimum_variance(cov,cap),max_sharpe(mu,cov,0,cap),risk_parity(cov,cap)]:
  assert np.isclose(w.sum(),1,atol=1e-7) and w.min()>=-1e-8 and w.max()<=cap+1e-6
def test_infeasible_cap_raises():
 try: minimum_variance(np.eye(3),.2)
 except ValueError: pass
 else: raise AssertionError('Expected infeasible cap error')
