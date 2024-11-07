# https://github.com/BorgwardtLab/simbsig

```console
docs/index.rst:**SIMBSIG**, a Python package which provides a scikit-learn-like interface for out-of-core, GPU-enabled similarity searches, principal component analysis, and clustering. Due to the PyTorch backend it is highly modular and particularly tailored to many data types with a particular focus on biobank data analysis.
docs/Advanced.rst:            If GPU is available and a SIMBSIG neighbors module (NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor,
docs/Advanced.rst:            RadiusNeighborsClassifier, RadiusNeighborsRegressor) is instantiated with device=='gpu', x1 and x2 will
docs/Advanced.rst:            be handed over to custom_metric the GPU.
docs/Advanced.rst:	    # 1. Compute pairwise distances between points in x1 and x2 using torch.tensor operations for GPU acceleration
docs/Advanced.rst:	    # If the GPU acceleration speedup is not required, moving x1 and x2 off the gpu and using for example
docs/Advanced.rst:	    If GPU is available and a simbsig neighbors module (NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor,
docs/Advanced.rst:	    RadiusNeighborsClassifier, RadiusNeighborsRegressor) is instantiated with device=='gpu', x1 and x2 will
docs/Advanced.rst:	    be handed over to custom_metric the GPU.
docs/Advanced.rst:	    # 1. Compute pairwise distances between points in x1 and x2 using torch.tensor operations for GPU acceleration
docs/Advanced.rst:	    # If the GPU acceleration speedup is not required, moving x1 and x2 off the gpu and using for example
docs/_build/html/searchindex.js:Search.setIndex({docnames:["Advanced","KNeighborsClassifier","KNeighborsRegressor","MiniBatchKMeans","NearestNeighbors","PCA","Quickstart","RadiusNeighborsClassifier","RadiusNeighborsRegressor","index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["Advanced.rst","KNeighborsClassifier.rst","KNeighborsRegressor.rst","MiniBatchKMeans.rst","NearestNeighbors.rst","PCA.rst","Quickstart.rst","RadiusNeighborsClassifier.rst","RadiusNeighborsRegressor.rst","index.rst"],objects:{"simbsig.cluster.MiniBatchKMeans":[[3,0,1,"","MiniBatchKMeans"]],"simbsig.cluster.MiniBatchKMeans.MiniBatchKMeans":[[3,0,1,"","fit"],[3,0,1,"","fit_predict"],[3,0,1,"","predict"]],"simbsig.decomposition.PCA":[[5,0,1,"","PCA"]],"simbsig.decomposition.PCA.PCA":[[5,0,1,"","fit"],[5,0,1,"","fit_transform"],[5,0,1,"","transform"]],"simbsig.neighbors.KNeighborsClassifier":[[1,0,1,"","KNeighborsClassifier"]],"simbsig.neighbors.KNeighborsClassifier.KNeighborsClassifier":[[1,0,1,"","fit"],[1,0,1,"","predict"],[1,0,1,"","predict_proba"]],"simbsig.neighbors.KNeighborsRegressor":[[2,0,1,"","KNeighborsRegressor"]],"simbsig.neighbors.KNeighborsRegressor.KNeighborsRegressor":[[2,0,1,"","fit"],[2,0,1,"","predict"]],"simbsig.neighbors.NearestNeighbors":[[4,0,1,"","NearestNeighbors"]],"simbsig.neighbors.NearestNeighbors.NearestNeighbors":[[4,0,1,"","fit"],[4,0,1,"","kneighbors"],[4,0,1,"","radius_neighbors"]],"simbsig.neighbors.RadiusNeighborsClassifier":[[7,0,1,"","RadiusNeighborsClassifier"]],"simbsig.neighbors.RadiusNeighborsClassifier.RadiusNeighborsClassifier":[[7,0,1,"","fit"],[7,0,1,"","predict"],[7,0,1,"","predict_proba"]],"simbsig.neighbors.RadiusNeighborsRegressor":[[8,0,1,"","RadiusNeighborsRegressor"]],"simbsig.neighbors.RadiusNeighborsRegressor.RadiusNeighborsRegressor":[[8,0,1,"","fit"],[8,0,1,"","predict"]]},objnames:{"0":["py","function","Python function"]},objtypes:{"0":"py:function"},terms:{"0":[1,2,3,4,5,6,7,8],"01":3,"1":[0,1,2,3,4,5,6,7,8],"100":3,"19th":3,"1d":4,"1e":3,"2":[0,1,2,4,5,6,7,8],"2010":3,"2011":5,"2580":5,"2594":5,"3":[0,6],"33":5,"333":6,"4":6,"5":[1,2,3,4,5],"666":6,"9":6,"95":3,"class":[0,1,3,5,7],"default":[1,2,3,4,5,7,8],"float":[1,2,3,4,7,8],"function":[0,1,2,7,8],"import":6,"int":[0,1,2,3,4,5,7,8],"return":[0,1,2,3,4,5,7,8],"true":[1,2,3,4,5,7,8],As:0,By:4,For:[1,2,4,7,8],If:[0,1,2,3,4,5,7,8],It:[1,2,7,8],The:[1,2,3,4,5,6,7,8],These:0,To:[0,4],With:0,acceler:[0,3,4,5],accord:[1,7],activ:[1,2,3,4,5,7,8,9],addit:[1,2,3,4,7,8],advanc:9,after:3,al:5,algorithm:[3,5],all:[1,2,7,8],allow:0,alpha:3,also:6,altern:3,among:[1,7],an:[0,1,2,3,4,5,7,8],analysi:[5,9],ani:[0,1,2,4,7,8],api:6,appli:[1,2,3,4,7,8],approxim:4,ar:[1,2,3,4,5,7,8],arbitrari:[1,2,4,7,8],argument:[1,2,3,4,7,8],around:4,arrai:[0,1,2,3,4,5,7,8],avail:[0,1,2,3,4,7,8,9],backend:9,ball:4,base:[1,2,7,8],batch:[1,2,3,4,5,7,8],batch_siz:[1,2,3,4,5,7,8],been:5,being:[1,2,3,4,7,8],belong:3,between:[0,1,2,3,4,7,8],big:[1,2,3,4,5,7,8],biobank:9,bool:[1,2,3,4,5,7,8],border:4,both:0,boundari:4,callabl:[1,2,3,4,7,8],can:[0,3,6],cdist:0,center:[3,5],chosen:0,chunk:[1,2,3,4,5,7,8],classifi:[1,7],clone:6,closer:[1,2,7,8],cluster:[3,9],collect:4,compon:[5,9],comput:[0,1,2,3,4,5,7,8],confer:3,constructor:[0,4],contain:4,convent:[3,4,5],convert:0,core:9,correspond:[4,6],cosin:[1,2,3,4,7,8],could:0,cpu:[0,1,2,3,4,5,7,8],custom:9,custom_metr:0,custom_rbf_metr:0,data:[1,2,3,4,5,7,8,9],dataload:[1,2,3,4,5,7,8],dataset:[1,2,3,4,5,7,8],datatset:[1,7],david:3,decomposit:5,def:0,defin:[1,2,3,4,7,8],describ:3,detail:6,develop:9,devic:[0,1,2,3,4,5,7,8],dict:[1,2,3,4,7,8],dictionari:0,differ:4,dimens:[0,3,4,5,7,8],directori:6,disk:[1,2,3,4,5,7,8],dist_mat:0,distanc:[0,1,2,3,4,5,7,8],document:6,doe:0,doubl:8,dtype:8,due:9,dure:[0,1,2,3,4,5,7,8],each:[1,2,4,7,8],earlier:3,effici:4,enabl:9,entir:[1,2,3,4,5,7,8],equal:[0,1,2,7,8],equival:[1,2,4,7,8],error:[1,2,3,4,5,7,8],estim:[1,2,4,7],et:5,euclidean:[0,1,2,3,4,7,8],euclidean_dist:0,euclidean_dist_mat:0,everi:[1,2,3,4,7,8],exampl:9,exclud:4,execut:6,exp:0,fals:[4,5],favor:[1,2,3,4,5,7,8],featur:[0,1,2,3,4,5,7,8],feature_weight:[0,1,2,3,4,7,8],file:[1,2,3,4,5,7,8],find:4,first:0,fit:[1,2,3,4,5,6,7,8],fit_predict:3,fit_transform:5,fix:8,focu:9,follow:0,format:[1,2,7,8],found:6,fraction:[1,2,3,4,7,8],from:[1,2,3,4,6,7,8],github:9,give:[1,2,3,4,7,8],given:4,gpu:[0,1,2,3,4,5,7,8,9],greater:[1,2,7,8],h5py:[1,2,3,4,5,7,8],halko:5,hand:0,handl:[1,2,3,4,5],have:[0,1,2,5,7,8],hdf5:[1,2,3,4,5,7,8],highli:9,how:0,howev:0,i:5,identifi:[1,2,7,8],ignor:[3,4,5],implement:[3,4,5],includ:[0,1,2,3,4,7,8],increas:4,index:[4,9],indic:4,influenc:[1,2,7,8],inform:[1,2,3,4,5,7,8],init:3,initi:3,input:[1,2,3,4,5,7,8],instal:9,instanti:0,integ:3,interfac:9,intern:3,invers:[],iter:[3,5],iterated_pow:5,its:[3,4],itself:4,job:[1,2,3,4,5,7,8],journal:5,k:[1,2,3,4],kei:0,kept:5,kernel:0,kernelis:0,keyword:[1,2,3,4,7,8],kmean:3,kneighbor:[1,2,4],kneighborsclassifi:[0,6,9],kneighborsregressor:[0,9],knn_classifi:6,kwarg:[1,2,3,4,5,7,8],l1:[1,2,4,7,8],l2:[1,2,4,7,8],l:5,l_p:[1,2,4,7,8],label:[1,7],larg:5,latter:[1,2,3,4,5,7,8],learn:9,learner:4,length:[1,2,3,4,7,8],lexicograph:[1,7],lie:4,like:[1,2,3,4,5,7,8,9],list:[1,2,3,4,5,7,8],load:[1,2,3,4,5,7,8],local:6,locat:[7,8],log:[1,2,3,4,5,7,8],ly:4,m_sampl:0,mahalanobi:[1,2,3,4,7,8],mai:[0,1,2,3,4,5,7,8],manhattan:[1,2,3,4,7,8],mani:[3,9],match:0,matrix:[1,2,3,4,7,8],max_it:3,maximum:3,mean:3,memori:[1,2,3,4,5,7,8],meth:[1,2],method:5,metric:[1,2,3,4,5,7,8,9],metric_param:[0,1,2,3,4,7,8],might:[3,4],minibatchkmean:9,minkowski:[1,2,3,4,7,8],mode:[1,2,3,4,5,7,8],modul:[0,6,9],modular:9,more:6,move:0,must:[1,2,3,4,7,8],n_class:[1,7],n_cluster:3,n_compon:5,n_featur:[0,1,2,3,4,5,7,8],n_features_in:[1,2,3,4,7,8],n_index:[1,2,4,7,8],n_job:[1,2,3,4,5,7,8],n_neighbor:[0,1,2,4,6],n_output:[1,7],n_oversampl:5,n_queri:[1,2,4,7,8],n_regress:[2,8],n_sampl:[0,1,2,3,4,5,7,8],name:0,nathan:5,ndarrai:[1,2,4,7,8],nearest:[1,2,4,7,8],nearestneighbor:[0,1,2,9],neigh_dist:4,neigh_ind:4,neighbor:[0,1,2,4,6,7,8],neighborhood:[1,2,7,8],nn_simbsig:0,none:[0,1,2,3,4,5,7,8],note:[1,2,4,7,8],notic:0,np:[0,1,2,3,4,7,8],number:[1,2,3,4,5,7,8],numpi:0,obj:3,object:[1,2,3,4,5,7,8],off:0,onc:[1,2,3,4,5,7,8],onli:[3,4,5],oper:0,optim:[1,2,3,4,5,7,8],option:[0,1,2,3,4,5,7,8],order:[1,7],other:[1,2,3,4,7,8],otherwis:[1,2],out:9,outlier_label:7,over:0,overcom:4,p:[0,1,2,4,7,8],packag:[6,9],page:9,pairwis:0,paper:5,paramet:[0,1,2,3,4,5,7,8],particular:9,particularli:9,pass:[0,1,2,3,4,7,8],pca:9,perform:[3,4,5],pip:6,poetri:6,point:[0,1,2,3,4,7,8],popul:4,possibl:0,pow:0,power:5,precomput:[1,2,3,4,5,7,8],predict:[1,2,3,6,7,8],predict_proba:[1,6,7],present:[3,4,5],princip:[5,9],principl:5,print:6,probabl:[1,7],problem:4,proceed:3,process:[1,2,3,4,5,7,8],produc:[1,2,3,4,5,7,8],progress:[1,2,3,4,5,7,8],project:9,proport:[1,2,7,8],provid:[1,2,3,4,7,8,9],pypi:6,python:9,pytorch:9,quantifi:[1,2,3,4,7,8],queri:[1,2,4,7,8],quickstart:9,radiu:[4,7,8],radius_neighbor:[4,7,8],radiusneighborsclassifi:[0,9],radiusneighborsregressor:[0,9],random:[3,5],random_st:[3,5],rbf:0,rbf_metric:[],rbf_pairwis:0,reach:3,regress:[2,8],regressor:[2,8],repositori:6,repres:4,requir:[0,1,2,7,8],respect:[1,2,4,7,8],result:[0,4],return_dist:4,s:[5,6],same:[3,4,5],sampl:[1,2,7,8],sample_weight:[1,2,7,8],satisfi:3,scale:3,scientif:5,scikit:9,scullei:3,sculli:3,search:[0,1,2,4,7,8,9],second:0,seed:[3,5],select:3,self:[1,2,3,4,5,7,8],set:[1,2,3,5,7,8],shape:[1,2,3,4,5,7,8],should:[0,1,2,3,4,5,7,8],show:0,siam:5,sigma:0,simbsig:[0,1,2,3,4,5,6,7,8],similar:[0,1,2,3,4,6,7,8,9],simliar:0,size:[1,2,3,4,5,7,8],sklearn:6,some_oper:0,sort:4,sort_result:4,space:[4,7,8],spars:[7,8],specif:[1,2,3,4,7,8],speedup:0,squar:[1,2,3,4,7,8],standard:4,state:[3,5],step:0,stop:3,store:4,str:[1,2,3,4,5,7,8],support:[1,2,3,4,5,7,8],tailor:9,take:[1,2,7,8],target:[1,2,7,8],tensor:[0,1,2,3,4,5,7,8],termin:3,test:[1,2,3,7,8],therefor:4,thi:[0,4,9],tol:3,toler:3,torch:[0,1,2,3,4,5,7,8],train:[1,2,3,4,5,7,8],transform:5,type:9,under:[6,9],uniform:[1,2,3,4,7,8],uniformli:3,unsupervis:4,updat:[1,2,3,4,5,7,8],upon:3,us:[0,1,2,3,4,5,6,7,8],user:[1,2,3,4,7,8],valu:[2,4,7,8],vector:[1,2,3,4,7,8],verbos:[1,2,3,4,5,7,8],veri:6,via:6,vote:[1,7],we:0,web:3,weight:[0,1,2,3,4,7,8],when:[1,2,3,4,5,7,8],whether:[1,2,3,4,5,7,8],which:[0,1,2,3,4,5,6,7,8,9],wide:3,within:4,world:3,x1:0,x2:0,x:[1,2,3,4,5,6,7,8],x_transform:5,y:[1,2,3,4,5,6,7,8],you:6},titles:["Advanced","KNeighborsClassifier","KNeighborsRegressor","MiniBatchKMeans","NearestNeighbors","PCA","Quickstart","RadiusNeighborsClassifier","RadiusNeighborsRegressor","Welcome to SIMBSIG\u2019s documentation!"],titleterms:{advanc:0,content:9,custom:0,document:9,exampl:[0,6],gener:0,indic:9,instal:6,interfac:0,kneighborsclassifi:1,kneighborsregressor:2,metric:0,minibatchkmean:3,nearestneighbor:4,pca:5,quickstart:6,radiusneighborsclassifi:7,radiusneighborsregressor:8,s:9,simbsig:9,tabl:9,welcom:9}})
docs/_build/html/_sources/index.rst.txt:**SIMBSIG**, a Python package which provides a scikit-learn-like interface for out-of-core, GPU-enabled similarity searches, principal component analysis, and clustering. Due to the PyTorch backend it is highly modular and particularly tailored to many data types with a particular focus on biobank data analysis.
docs/_build/html/_sources/Advanced.rst.txt:            If GPU is available and a SIMBSIG neighbors module (NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor,
docs/_build/html/_sources/Advanced.rst.txt:            RadiusNeighborsClassifier, RadiusNeighborsRegressor) is instantiated with device=='gpu', x1 and x2 will
docs/_build/html/_sources/Advanced.rst.txt:            be handed over to custom_metric the GPU.
docs/_build/html/_sources/Advanced.rst.txt:	    # 1. Compute pairwise distances between points in x1 and x2 using torch.tensor operations for GPU acceleration
docs/_build/html/_sources/Advanced.rst.txt:	    # If the GPU acceleration speedup is not required, moving x1 and x2 off the gpu and using for example
docs/_build/html/_sources/Advanced.rst.txt:	    If GPU is available and a simbsig neighbors module (NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor,
docs/_build/html/_sources/Advanced.rst.txt:	    RadiusNeighborsClassifier, RadiusNeighborsRegressor) is instantiated with device=='gpu', x1 and x2 will
docs/_build/html/_sources/Advanced.rst.txt:	    be handed over to custom_metric the GPU.
docs/_build/html/_sources/Advanced.rst.txt:	    # 1. Compute pairwise distances between points in x1 and x2 using torch.tensor operations for GPU acceleration
docs/_build/html/_sources/Advanced.rst.txt:	    # If the GPU acceleration speedup is not required, moving x1 and x2 off the gpu and using for example
benchmarking/runtime_benchmark_cuml.py:def test_in_core_gpu(dataset,queryset):
benchmarking/runtime_benchmark_cuml.py:    runtime_df_gpu = test_in_core_gpu(hf,hf_query)
benchmarking/runtime_benchmark_cuml.py:    runtime_df_gpu.to_csv('runtime_df_gpu_cuml.csv')
benchmarking/runtime_benchmark.py:def test_in_core_gpu(dataset,queryset):
benchmarking/runtime_benchmark.py:        neigh_own = NearestNeighborsOwn(n_neighbors=4,metric='euclidean',device='gpu',batch_size=batch_size)
benchmarking/runtime_benchmark.py:        kmeans_simbsig = MiniBatchKMeansOwn(n_clusters=4,init=init,random_state=47,device='gpu',batch_size=batch_size)
benchmarking/runtime_benchmark.py:        pca_simbsig = PCAOwn(n_components=4,device='gpu',iterated_power=0,n_oversamples=6,batch_size=batch_size,random_state=47)
benchmarking/runtime_benchmark.py:    out_df = pd.DataFrame(index=patients_step,columns=['neighbors_ooc_cpu','neighbors_ooc_gpu','kmeans_ooc_cpu','kmeans_ooc_gpu','pca_ooc_cpu','pca_ooc_gpu'])
benchmarking/runtime_benchmark.py:    for device in ['cpu','gpu']:
benchmarking/runtime_benchmark.py:        n_jobs = cpu_count()# if device == 'gpu' else cpu_count()-1
benchmarking/runtime_benchmark.py:        print('In core GPU')
benchmarking/runtime_benchmark.py:        runtime_df_gpu = test_in_core_gpu(hf,hf_query)
benchmarking/runtime_benchmark.py:        runtime_df_gpu.to_csv(os.path.join(DATA_PATH,f'runtime_df_gpu_{run}.csv'))
benchmarking/runtime_benchmark.py:        # print(runtime_df_gpu)
benchmarking/create_figures.py:df_list_gpu = [pd.read_csv(os.path.join(DATA_PATH,f'runtime_df_gpu_{i}.csv'),index_col=0) for i in range(n_trials)]
benchmarking/create_figures.py:cols_to_drop = [c for c in df_list_gpu[0].columns if 'cuml' in c]
benchmarking/create_figures.py:cols_rename = {c:c+'_gpu' for c in df_list_gpu[0].columns if 'cuml' not in c}
benchmarking/create_figures.py:df_list_gpu = [df.drop(cols_to_drop,axis=1).rename(columns=cols_rename) for df in df_list_gpu]
benchmarking/create_figures.py:df_gpu_all = pd.concat(df_list_gpu,axis=1)
benchmarking/create_figures.py:df_figure = pd.concat([df_cpu_all,df_gpu_all,df_ooc_all],axis=1) #.loc[:int(1e5)]
benchmarking/create_figures.py:        plt.legend(loc='lower right', labels=['Scikit-learn', 'SIMBSIG CPU', 'SIMBSIG GPU', 'SIMBSIG CPU OOC', 'SIMBSIG GPU OOC'],fontsize=10)
README.md:# SIMBSIG = SIMilarity Batched Search Integrated Gpu-based
README.md:SIMBSIG is a GPU accelerated software tool for neighborhood queries, KMeans and PCA which mimics the sklearn API.
README.md:The algorithm for batchwise data loading and GPU usage follows the principle of [1]. The algorithm for KMeans follows the Mini-batch KMeans described by Scully [2]. The PCA algorithm follows Halko's method [3].
README.md:  [1] Gutiérrez, P. D., Lastra, M., Bacardit, J., Benítez, J. M., & Herrera, F. (2016). GPU-SME-kNN: Scalable and memory efficient kNN and lazy learning using GPUs. Information Sciences, 373, 165-182.
testing/MiniBatchKMeans_tests.py:                               for device in ['cpu']:#,'gpu']:
testing/RadiusNeighborsRegressor_tests.py:                    for device in ['cpu']:#,'gpu']:
testing/KNeighborsRegressor_tests.py:                  for device in ['cpu']:#,'gpu']:
testing/Callable_metric_tests.py:            If GPU is available and a simbsig neighbors module (NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor,
testing/Callable_metric_tests.py:            RadiusNeighborsClassifier, RadiusNeighborsRegressor) is instantiated with device=='gpu', x1 and x2 will
testing/Callable_metric_tests.py:            be handed over to custom_metric the GPU.
testing/Callable_metric_tests.py:            # 1. Compute pairwise distances between points in x1 and x2 using torch.tensor operations for GPU acceleration
testing/Callable_metric_tests.py:            # If the GPU acceleration speedup is not required, moving x1 and x2 off the gpu and using for example
testing/Callable_metric_tests.py:                for device in ['cpu']:#,'gpu']:
testing/Precomputed_tests.py:                for device in ['cpu']:#,'gpu']:
testing/KNeighborsClassifier_tests.py:                            for device in ['cpu']:#,'gpu']:
testing/Simbsig_Tests.py:# At the moment, we only test on CPU as the algorithms use the same code on CPU and GPU.
testing/Simbsig_Tests.py:# CUDA seems seems to produce marginally different outputs as CPU computations at times.
testing/Simbsig_Tests.py:# We attribute this to the different routines pytorch uses for CPU and CUDA computations.
testing/PCA_tests.py:                      for device in ['cpu']:#,'gpu']:
testing/NearestNeighbors_tests.py:                        for device in ['cpu']:#,'gpu']:
testing/RadiusNeighborsClassifier_tests.py:                        for device in ['cpu']:#,'gpu']:
pyproject.toml:keywords = ["similarity search", "kmeans", "knn", "nearest neighbors", "gpu", "pca"]
simbsig/cluster/MiniBatchKMeans.py:    optional GPU accelerated computations.
simbsig/cluster/MiniBatchKMeans.py:            Options supported are: [‘cpu’,’gpu’]
simbsig/cluster/MiniBatchKMeans.py:            optimized for dataset when using `device='gpu'`.
simbsig/cluster/MiniBatchKMeans.py:            which may return an error when using `device='gpu'`.
simbsig/cluster/MiniBatchKMeans.py:        self.device = torch.device('cuda') if device == 'gpu' else torch.device('cpu')
simbsig/base/_base.py:    """Private basis class which implements batched data loading for big datasets and optional GPU accelerated computations
simbsig/base/_base.py:            Options supported are: [‘cpu’,’gpu’]
simbsig/base/_base.py:            optimized for dataset when using `device='gpu'`.
simbsig/base/_base.py:            which may return an error when using `device='gpu'`.
simbsig/base/_base.py:        self.device = torch.device('cuda') if device == 'gpu' else torch.device('cpu')
simbsig/base/_base.py:                if self.device == torch.device('cuda'):
simbsig/decomposition/PCA.py:    optional GPU accelerated computations.
simbsig/decomposition/PCA.py:            Options supported are: [‘cpu’,’gpu’]
simbsig/decomposition/PCA.py:            optimized for dataset when using `device='gpu'`.
simbsig/decomposition/PCA.py:            which may return an error when using `device='gpu'`.
simbsig/decomposition/PCA.py:        self.device = torch.device('cuda') if device == 'gpu' else torch.device('cpu')
simbsig/neighbors/NearestNeighbors.py:    """Unsupervised learner performing neighbor searches. Implements batched data loading for big datasets and optional GPU accelerated computations
simbsig/neighbors/NearestNeighbors.py:            Options supported are: [‘cpu’,’gpu’]
simbsig/neighbors/NearestNeighbors.py:            optimized for dataset when using `device='gpu'`.
simbsig/neighbors/NearestNeighbors.py:            which may return an error when using `device='gpu'`.
simbsig/neighbors/KNeighborsRegressor.py:            Options supported are: [‘cpu’,’gpu’]
simbsig/neighbors/KNeighborsRegressor.py:            optimized for dataset when using `device='gpu'`.
simbsig/neighbors/KNeighborsRegressor.py:            which may return an error when using `device='gpu'`.
simbsig/neighbors/RadiusNeighborsClassifier.py:        Options supported are: [‘cpu’,’gpu’]
simbsig/neighbors/RadiusNeighborsClassifier.py:        optimized for dataset when using `device='gpu'`.
simbsig/neighbors/RadiusNeighborsClassifier.py:        which may return an error when using `device='gpu'`.
simbsig/neighbors/KNeighborsClassifier.py:        Options supported are: [‘cpu’,’gpu’]
simbsig/neighbors/KNeighborsClassifier.py:        optimized for dataset when using `device='gpu'`.
simbsig/neighbors/KNeighborsClassifier.py:        which may return an error when using `device='gpu'`.
simbsig/neighbors/RadiusNeighborsRegressor.py:         Options supported are: [‘cpu’,’gpu’]
simbsig/neighbors/RadiusNeighborsRegressor.py:         optimized for dataset when using `device='gpu'`.
simbsig/neighbors/RadiusNeighborsRegressor.py:         which may return an error when using `device='gpu'`.
simbsig/utils/metrics.py:        if device == 'gpu':
simbsig/utils/metrics.py:            device = torch.device('cuda')
simbsig/utils/metrics.py:    If GPU is available and a simbsig neighbors module (NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor,
simbsig/utils/metrics.py:    RadiusNeighborsClassifier, RadiusNeighborsRegressor) is instantiated with device=='gpu', x1 and x2 will
simbsig/utils/metrics.py:    be handed over to custom_metric the GPU.
simbsig/utils/metrics.py:    # 1. Compute pairwise distances between points in x1 and x2 using torch.tensor operations for GPU acceleration
simbsig/utils/metrics.py:    # If the GPU acceleration speedup is not required, moving x1 and x2 off the gpu and using for example
simbsig/utils/metrics.py:    If GPU is available and a simbsig neighbors module (NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor,
simbsig/utils/metrics.py:    RadiusNeighborsClassifier, RadiusNeighborsRegressor) is instantiated with device=='gpu', x1 and x2 will
simbsig/utils/metrics.py:    be handed over to custom_metric the GPU.
simbsig/utils/metrics.py:    # 1. Compute pairwise distances between points in x1 and x2 using torch.tensor operations for GPU acceleration
simbsig/utils/metrics.py:    # If the GPU acceleration speedup is not required, moving x1 and x2 off the gpu and using for example
poetry.lock:description = "Tensors and Dynamic neural networks in Python with strong GPU acceleration"

```
