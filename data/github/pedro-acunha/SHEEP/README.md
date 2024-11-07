# https://github.com/pedro-acunha/SHEEP

```console
clf_1vsall.py:        #"tree_method":'gpu_hist',
clf_1vsall.py:              colsample_bytree=0.9643171868050568, gamma=0, gpu_id=-1,
clf_1vsall.py:              tree_method='gpu_hist', use_label_encoder=False,
clf_1vsall.py:                            n_estimators= 290, task_type = 'GPU', verbose=0)
clf_1vsall.py:              colsample_bytree=0.4972272561013618, gamma=0, gpu_id=-1,
clf_1vsall.py:              subsample=0.9462723804220237, tree_method='gpu_hist',
clf_1vsall.py:                            n_estimators= 122, task_type = 'GPU', verbose=0)
clf_1vsall.py:              gamma=0, gpu_id=0, importance_type='gain',
clf_1vsall.py:              tree_method='gpu_hist', validate_parameters=1, verbosity=None)
clf_1vsall.py:                             n_estimators= 167, task_type = 'GPU',verbose=0)
clf_1vsall.py:              gamma=0, gpu_id=0, importance_type='gain',
clf_1vsall.py:              tree_method='gpu_hist', validate_parameters=1, verbosity=None)
photo_z.py:xgb_clf = xgb.XGBRegressor(n_estimators=1500, n_jobs=-1,tree_method='gpu_hist', random_state=24)
photo_z.py:cb_model = CatBoostRegressor(iterations=1500, max_depth=10, task_type="GPU",random_seed=0, verbose=0)
photo_z.py:models = {'xgb':xgb.XGBRegressor(n_estimators=1500, n_jobs=-1,tree_method='gpu_hist', random_state=24),
photo_z.py:        'cb':CatBoostRegressor(iterations=1500, task_type="GPU",random_seed=0, verbose=0),
photo_z.py:models = {'xgb':xgb.XGBRegressor(n_estimators=1500, n_jobs=-1,tree_method='gpu_hist', random_state=24),
photo_z.py:        'cb':CatBoostRegressor(iterations=1500, task_type="GPU",random_seed=0, verbose=0),
clf_multi.py:              colsample_bytree=0.671168787703373, gamma=0, gpu_id=-1,
clf_multi.py:              tree_method='gpu_hist')
clf_multi.py:                            n_estimators= 131,verbose=0, task_type="GPU")
clf_multi.py:                            tree_method='gpu_hist', validate_parameters=1, verbosity=None)
clf_multi.py:              colsample_bytree=0.671168787703373, gamma=0, gpu_id=-1,
clf_multi.py:              tree_method='gpu_hist')

```
