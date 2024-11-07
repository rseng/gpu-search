# https://github.com/Beuth-Erdelt/DBMS-Benchmarker

```console
dbmsbenchmarker/scripts/dashboardcli.py:    elif filter_by == 'GPU':
dbmsbenchmarker/scripts/dashboardcli.py:        connections_by_filter = e.get_experiment_list_connections_by_hostsystem('GPU')
dbmsbenchmarker/scripts/dashboardcli.py:     Output('dropdown_gpu', 'options'),
dbmsbenchmarker/scripts/dashboardcli.py:     Output('dropdown_gpu', 'optionHeight'),
dbmsbenchmarker/scripts/dashboardcli.py:             e.get_experiment_list_connections_by_hostsystem('GPU'),
dbmsbenchmarker/scripts/dashboardcli.py:     Output('dropdown_gpu', 'value'),
dbmsbenchmarker/scripts/dashboardcli.py:     State('dropdown_gpu', 'value'),
dbmsbenchmarker/scripts/dashboardcli.py:def filter_connections(nc1, nc2, nc3, active_graph, store_dashboard_config, dbms, node, client, gpu, cpu,
dbmsbenchmarker/scripts/dashboardcli.py:        dict_filter_values = {'DBMS': dbms, 'Node': node, 'Client': client, 'GPU': gpu, 'CPU': cpu}
dbmsbenchmarker/layout.py:                html.Label("GPU"),
dbmsbenchmarker/layout.py:                    id='dropdown_gpu',
dbmsbenchmarker/layout.py:                        options=[{'label': x, 'value': x} for x in ['DBMS', 'Node', 'Script', 'CPU Limit', 'Client', 'GPU', 'CPU', 'Docker Image', 'Experiment Run']],
dbmsbenchmarker/evaluator.py:                        if 'total_gpu_power' in hardwareAverages[c]:
dbmsbenchmarker/evaluator.py:                            evaluation['dbms'][c]['hardwaremetrics']['total_gpu_energy'] = hardwareAverages[c]['total_gpu_power']*times[c]/3600000
dbmsbenchmarker/evaluator.py:    if 'CUDA' in df2.columns:
dbmsbenchmarker/evaluator.py:        df2 = df2.drop(['CUDA'],axis=1)
dbmsbenchmarker/evaluator.py:    if 'GPUIDs' in df2.columns:
dbmsbenchmarker/evaluator.py:        df2 = df2.drop(['GPUIDs'],axis=1)
dbmsbenchmarker/evaluator.py:    df = df1.merge(df2,left_index=True,right_index=True).drop(['host','CPU','GPU','RAM','Cores'],axis=1)
dbmsbenchmarker/evaluator.py:    #df3=df1.merge(df2,left_index=True,right_index=True).drop(['CUDA','host','CPU','GPU','instance','RAM','Cores'],axis=1)
dbmsbenchmarker/tools.py:        if 'CUDA' in df2.columns:
dbmsbenchmarker/tools.py:            df2 = df2.drop(['CUDA'],axis=1)
dbmsbenchmarker/tools.py:        if 'GPUIDs' in df2.columns:
dbmsbenchmarker/tools.py:            df2 = df2.drop(['GPUIDs'],axis=1)
dbmsbenchmarker/tools.py:        df = df1.merge(df2,left_index=True,right_index=True).drop(['host','CPU','GPU','RAM','Cores'],axis=1)
dbmsbenchmarker/tools.py:        #df3=df1.merge(df2,left_index=True,right_index=True).drop(['CUDA','host','CPU','GPU','instance','RAM','Cores'],axis=1)
dbmsbenchmarker/tools.py:        if 'CUDA' in df.columns:
dbmsbenchmarker/tools.py:            df = df.drop(['CUDA'],axis=1)
dbmsbenchmarker/tools.py:        if 'GPUIDs' in df.columns:
dbmsbenchmarker/tools.py:            df = df.drop(['GPUIDs'],axis=1)
dbmsbenchmarker/tools.py:        df = df.drop(['host','CPU','GPU'],axis=1)#,'RAM','Cores'
dbmsbenchmarker/monitor.py:        'total_gpu_util': {
dbmsbenchmarker/monitor.py:            'query': 'sum(dcgm_gpu_utilization)',
dbmsbenchmarker/monitor.py:            'title': 'GPU Util [%]'
dbmsbenchmarker/monitor.py:        'total_gpu_power': {
dbmsbenchmarker/monitor.py:            'title': 'GPU Power Usage [W]'
dbmsbenchmarker/monitor.py:        'total_gpu_memory': {
dbmsbenchmarker/monitor.py:            'title': 'GPU Memory [MiB]'
dbmsbenchmarker/monitor.py:    # helps evaluating GPU util
docs/Dashboard.md:      * GPU
docs/Options.md:  'info': 'It runs on a P100 GPU',
docs/Options.md:  'info': 'It runs on a P100 GPU',
docs/Options.md:* `result`: Compare complete result set. Every cell is trimmed. Floats can be rounded to a given `precision` (decimal places). This is important for example for comparing CPU and GPU based DBMS.
docs/Options.md:      'CUDA': ' NVIDIA-SMI 410.79       Driver Version: 410.79       CUDA Version: 10.0',
docs/Concept.md:'title': 'GPU Util [%]'
docs/Concept.md:'query': 'DCGM_FI_DEV_GPU_UTIL{UUID=~"GPU-4d1c2617-649d-40f1-9430-2c9ab3297b79"}'
docs/Concept.md:'title': 'GPU Power Usage [W]'
docs/Concept.md:'query': 'DCGM_FI_DEV_POWER_USAGE{UUID=~"GPU-4d1c2617-"}'
docs/Concept.md:'title': 'GPU Memory [MiB]'
docs/Concept.md:'query': 'DCGM_FI_DEV_FB_USED{UUID=~"GPU-4d1c2617-"}'
docs/Concept.md:We also can have various hardware metrics like CPU and GPU utilization, CPU throttling, memory caching and working set.
test-result.py:        list_connections_gpu = evaluate.get_experiment_list_connections_by_hostsystem('GPU')
dashboard.py:    elif filter_by == 'GPU':
dashboard.py:        connections_by_filter = e.get_experiment_list_connections_by_hostsystem('GPU')
dashboard.py:     Output('dropdown_gpu', 'options'),
dashboard.py:     Output('dropdown_gpu', 'optionHeight'),
dashboard.py:             e.get_experiment_list_connections_by_hostsystem('GPU'),
dashboard.py:     Output('dropdown_gpu', 'value'),
dashboard.py:     State('dropdown_gpu', 'value'),
dashboard.py:def filter_connections(nc1, nc2, nc3, active_graph, store_dashboard_config, dbms, node, client, gpu, cpu,
dashboard.py:        dict_filter_values = {'DBMS': dbms, 'Node': node, 'Client': client, 'GPU': gpu, 'CPU': cpu}
README.md:  * connects to all DBMS having a JDBC interface - including GPU-enhanced DBMS
layout.py:                html.Label("GPU"),
layout.py:                    id='dropdown_gpu',
layout.py:                        options=[{'label': x, 'value': x} for x in ['DBMS', 'Node', 'Script', 'CPU Limit', 'Client', 'GPU', 'CPU', 'Docker Image', 'Experiment Run']],
demo-interactive-inspection.py:list_connections_gpu = evaluate.get_experiment_list_connections_by_hostsystem('GPU')
demo-interactive-inspection.py:evaluate.get_aggregated_query_statistics(type='monitoring', name='total_gpu_util', query_aggregate='Mean')
demo-interactive-inspection.py:evaluate.get_aggregated_query_statistics(type='monitoring', name='total_gpu_memory', query_aggregate='Max')
demo-interactive-inspection.py:evaluate.get_aggregated_query_statistics(type='monitoring', name='total_gpu_memory', query_aggregate='factor', factor_base='Max', dbms_filter=list_omnisci)
demo-interactive-inspection.py:evaluate.get_aggregated_experiment_statistics(type='monitoring', name='total_gpu_memory', query_aggregate='Mean', total_aggregate='Mean')
demo-interactive-inspection.py:evaluate.get_measures_and_statistics(numQuery, type='monitoring', name='total_gpu_util')
demo-interactive-inspection.py:evaluate.get_measures_and_statistics(numQuery, type='monitoring', name='total_gpu_memory')
demo-interactive-inspection.py:evaluate.get_measures_and_statistics(numQuery, type='monitoring', name='total_gpu_memory', dbms_filter=list_omnisci, factor_base='Min')
paper.md:Their types can be divided into, for example, row-wise, column-wise, in-memory, distributed, and GPU-enhanced. 
paper.md:We also can have various hardware metrics like CPU and GPU utilization, CPU throttling, memory caching, and working set.

```
