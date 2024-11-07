# https://github.com/exoclime/HELIOS

```console
helios.py:    # create, convert and copy arrays to be used in the GPU computations
helios.py:    # conduct core computations on the GPU
helios.py:    # copy everything from the GPU back to host and write output quantities to files
docs/sections/requirements.rst:HELIOS is a GPU-accelerated software developed with parts written in CUDA. It thus requires an NVIDIA graphics card (GPU) to operate on. Any GeForce or Tesla card manufactured since 2013 and with 2 GB VRAM or more should suffice to run standard applications of HELIOS.
docs/sections/requirements.rst:CUDA
docs/sections/requirements.rst:CUDA is the NVIDIA API responsible for the communication between the graphics card (aka device) and the CPU (aka host). The software package consists of the core libraries, development utilities and the NVCC compiler to interpret C/C++ code. The CUDA toolkit can be downloaded from `here <https://developer.nvidia.com/cuda-downloads>`_.
docs/sections/requirements.rst:HELIOS has been tested with CUDA versions 7.x -- 11.x and should, in principle, also be compatible with any newer version.
docs/sections/requirements.rst:HELIOS's computational core is written in CUDA C++, but the user shell comes in Python modular format. To communicate between the host and the device the PyCUDA wrapper is used.
docs/sections/requirements.rst:* PyCUDA
docs/sections/requirements.rst:Some of them may be already included in the python distribution (e.g., Anaconda). Otherwise they can be installed with the Python package manager pip. To install, e.g., PyCUDA type::
docs/sections/requirements.rst:   pip install pycuda
docs/sections/parameters.rst:This changes the numerical precision used for the GPU calculations. However, I have never found any difference in terms of accuracy or speed when switching between single and double precision. (Perhaps I did something wrong). It is probably the best to leave it at 'double'.
docs/sections/tutorial.rst:In the following I merely expect that you possess an NVIDIA GPU and are either on a Linux or Mac Os X system (sorry, Windows users. You are on your own).
docs/sections/tutorial.rst:1. Install the newest version of the CUDA toolkit from `NVIDIA <https://developer.nvidia.com/cuda-downloads>`_. To ascertain a successful installation, type ``which nvcc``. This should provide you with the location of the nvcc compiler. If not, something went wrong with the installation. 
docs/sections/tutorial.rst:Make sure that the library and program paths are exported. On Mac Os x you should have the following entries in your .bash_profile file (shown for version 10.0 of CUDA) ::
docs/sections/tutorial.rst:	export PATH=/Developer/NVIDIA/CUDA-10.0/bin:$PATH
docs/sections/tutorial.rst:	export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-10.0/lib:$DYLD_LIBRARY_PATH
docs/sections/tutorial.rst:	export PATH=/usr/local/cuda-10.0/bin:$PATH
docs/sections/tutorial.rst:	export DYLD_LIBRARY_PATH=/usr/local/cuda-10.0/lib:$DYLD_LIBRARY_PATH
docs/sections/tutorial.rst:  to activate the environment. Then we need to uninstall and reinstall pycuda package (the pre--installed version does not work usually). Type ::
docs/sections/tutorial.rst:	pip uninstall pycuda
docs/sections/tutorial.rst:	pip install pycuda
docs/sections/structure.rst:* ``quantities.py``: contains all scalar variables and arrays. It is responsible for data management, like copying arrays between the host and the device (GPU), and allocating memory. 
docs/sections/structure.rst:* ``computation.py``: calls and co-ordinates the device kernels, i.e., functions living on the GPU. If you write a new GPU functionality (=kernel) include it here.
docs/sections/structure.rst:* ``kernels.cu``: contains the detailed computations, executed on the GPU/device. Write new kernel functions or alter existing ones here.
docs/build/html/searchindex.js:Search.setIndex({docnames:["index","sections/about","sections/ktable","sections/license","sections/parameters","sections/requirements","sections/structure","sections/tutorial"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,sphinx:56},filenames:["index.rst","sections/about.rst","sections/ktable.rst","sections/license.rst","sections/parameters.rst","sections/requirements.rst","sections/structure.rst","sections/tutorial.rst"],objects:{},objnames:{},objtypes:{},terms:{"100":[4,7],"1000":[4,7],"100th":4,"100x":7,"10x":7,"1214":7,"1214b":7,"15999":4,"1977":2,"1988":2,"1989":4,"1991":4,"1994":2,"1e8":4,"1e9":4,"2000":2,"2004":2,"2005":2,"2008":[2,4],"2009":7,"2013":[5,7],"2014":[2,4],"2016":4,"2017":[1,4],"2018":4,"2019":4,"2019a":1,"2019b":1,"2021":4,"2022":1,"285714":4,"291":4,"3996":4,"439e4":2,"439e8":2,"800":4,"8000":4,"break":4,"case":[4,5,7],"default":[2,4,6,7],"export":7,"final":[0,1,4,7],"function":[4,6,7],"g\u00f6ttingen":7,"import":[2,4,7],"long":6,"new":[0,2,4,6],"public":3,"return":[2,5,7],"short":6,"switch":4,"try":5,"while":[2,7],And:4,CGS:2,For:[1,2,3,4,5,7],Going:7,H2S:[],I2S:4,MKS:2,Not:2,That:[6,7],The:[1,2,4,5,6,7],Then:[2,7],There:[2,7],These:[4,7],Use:0,Using:[0,4],WILL:[],With:[4,7],_bf:2,_convergence_warn:4,_ff:2,_ip_:2,about:0,abs:4,abscissa:4,absorb:[2,7],absorpt:[2,4,6,7],abund:[1,7],acceler:5,accept:[1,7],access:[6,7],accord:7,account:4,accur:[2,4,7],accuraci:4,act:7,activ:[4,7],actual:2,adapt:4,add:[2,4,6,7],added:4,addit:[2,4,6,7],addition:4,additional_h:6,adiabat:[4,7],adjac:4,adjust:[0,1,7],admin:5,advanc:0,advantag:7,advect:4,aerosol:[4,7],aesthet:6,affect:4,aforement:7,after:[2,4,6,7],again:7,aka:[5,6],albedo:[1,4],albeit:4,algorithm:4,all:[2,4,6,7],alloc:6,allow:[2,4,7],almost:7,alpha:2,alreadi:[2,5,7],also:[1,4,5,7],alter:6,altern:[4,7],although:4,altitud:[4,7],alwai:[2,4],amount:4,amu:[2,7],amundsen:4,anaconda:[5,7],analog:[2,7],analysi:7,analyz:4,angl:4,ani:[0,1,4,5,6],anoth:[2,4,7],anymor:4,anyth:2,anywai:4,apart:7,api:5,app:4,appear:[2,6,7],append:4,appli:[4,7],applic:[4,5],appreci:1,approach:[2,4,7],approxim:[1,2,4],arbitrarili:2,archiv:7,archlinux:5,around:[2,4],arrai:6,arrow:[2,4],ascend:7,ascertain:7,ascii:[2,7],ask:7,assum:[2,4,7],astropi:[2,5],asymmetri:4,atmospher:[0,1,2,4,7],atom:4,attempt:7,automat:[4,7],averag:4,avid:7,avoid:4,azimuth:4,back:7,background:4,backward:4,balanc:4,band:[4,7],bar:[4,7],bare:4,base:[4,7],bash:7,bash_profil:7,basi:6,basic:7,bb_temp:7,beam:4,becaus:[2,4,7],becom:[4,7],beed:4,been:[0,4,5,7],befor:[4,5],begin:4,behind:4,being:[2,4,7],belong:7,below:[2,4,7],bern:7,best:[4,7],better:[4,7],between:[4,5,6,7],beyond:4,bin:[2,7],binari:[2,7],blackbodi:[4,7],boa:4,boltzmann:4,both:[1,4],bottom:[4,7],bound:[2,7],bracket:[2,4],briefli:6,bright:4,bring:4,bubbl:5,budget:[2,4],bug:1,build_individual_opac:2,bypass:7,c2h2:[],c_p:[0,4],calcul:[1,2,4,6,7],call:[4,5,6,7],can:[1,2,4,5,6,7],cannot:[2,4],capac:7,card:5,caus:5,cautious:4,center:4,certain:[1,4,7],cgs:[4,7],ch4:[],chain:4,chang:[2,4,6,7],check:[6,7],chem:[4,7],chem_high:[4,7],chem_low:[4,7],chemic:[1,7],chemistri:[0,1,4,6],choic:7,choos:[2,4,7],chosen:[2,4,7],chronolog:6,cia:[0,2],cia_co2co2:7,cia_h2h2:7,circumst:4,cite:1,click:7,clone:7,close:4,closer:4,closur:4,cloud:[0,6],cloud_fil:7,co2:[2,7],code:[0,1,4,5,7],coeffici:[2,4,6,7],coincid:[2,4],collis:[2,7],column:[2,4,7],com:7,combin:[2,7],come:[2,5,7],command:[2,4,6,7],comment:7,common:[2,6,7],commun:5,compar:[4,7],compat:[0,4,5],compil:[5,7],complain:7,composit:7,compromis:4,comput:[1,4,5,6],conda:7,condit:[4,7],conduct:7,config:7,configur:[6,7],congratul:7,consid:1,consist:[4,5,7],constant:[2,4,6,7],construct:7,contain:[2,4,6,7],content:[2,4,7],continu:[2,4,7],contribut:[2,4,7],conv:4,convect:[0,1,7],conveni:[2,7],converg:[4,7],convers:4,convert:[2,4,6,7],convert_to:7,coordin:4,copi:[2,6],core:[1,5],correct:[4,7],correl:[4,7],correspond:[1,7],cost:[2,4],could:[4,7],coupl:0,coupling_templ:7,cox:2,cpp:7,cpu:[5,6],creat:[0,2,4,6],criterion:4,cross:[2,7],crucial:1,cuda:[0,7],curiou:7,current:[2,4,7],cycl:7,dai:4,daili:6,damp:4,dash:[2,4],dat:[2,4,6,7],data:[1,2,4,6,7],databas:[2,6,7],dataset:4,daysid:[4,7],deal:7,debug:[4,5],decai:7,deck:[4,6,7],deep:4,deficit:4,defin:[2,4,7],definit:7,deg:4,degre:4,delad:[4,7],delad_exampl:7,delta_lambda:2,densiti:4,depend:[2,4,7],depth:4,describ:[1,2,4,6,7],design:1,desir:[4,7],detail:[2,4,6],detect:4,determin:[2,4],dev:4,develop:[5,7],deviat:4,devic:[5,6],diatom:4,did:4,differ:[0,1,2,4],diffus:4,dimens:4,dimension:1,direct:4,directli:[2,7],directori:[0,2,4,7],disabl:4,discuss:4,distanc:4,distribut:[2,3,4,5,7],divid:7,doc:6,document:[0,6,7],doe:[4,5,7],doing:7,don:[4,5,6,7],done:7,door:7,doubl:4,down:4,download:[0,2,5],downward:4,due:4,dure:[4,7],dyld_library_path:7,each:[2,4,6,7],earlier:[4,5,7],eclips:7,eddington:4,edu:[0,1],effici:4,either:[2,3,4,7],electron:2,element:7,els:4,emb:5,emiss:[1,7],emploi:2,enabl:4,encompass:4,encount:4,energi:[2,4],enter:4,entri:[2,7],entropi:7,environ:[5,7],equal:7,equat:[1,2],equilibrium:[1,2,4,7],erg:[4,7],error:[0,2,4,5,6,7],esp:1,etc:[6,7],even:[4,7],everi:[2,4],everyth:[2,4,7],evolv:4,exact:4,exactli:[4,7],exampl:[2,4,7],exclus:7,execut:[2,6,7],exhibit:7,exist:[2,4,5,6,7],exoclim:[1,7],exoplanetari:[0,1],expect:[4,7],experi:4,explain:[4,6,7],explan:2,explor:[6,7],exponenti:7,extens:[0,4],extern:[4,7],extinct:[4,7],extract:2,extrapol:7,eye:7,f_down:4,f_int:4,f_net:4,fact:4,factor:4,fail:[5,7],fastchem:[1,2,4,6,7],fastchem_input:7,fastchem_manu:7,faster:[4,7],feasibl:4,featur:[4,6,7],fed:4,feed:4,feedback:4,feel:[4,6,7],few:4,fig:4,file:[0,1],final_speci:2,find:[0,1,2,7],finish:6,first:[0,4,5],fit:7,fix:2,fixed_resolut:2,flux:[4,7],fly:[0,2,4],follow:[2,4,5,6,7],forc:4,formal:4,format:[0,1,4,5,6,7],formula:4,forward:4,found:[2,4,6,7],fraction:4,free:[2,4,6,7],freeli:2,from:[1,2,4,5,6,7],full:[1,4,7],fulli:4,further:[4,7],furthermor:4,futur:7,g_0:4,gas:[4,7],gaseou:[4,7],gaussian:2,geforc:5,gener:[0,1,3,6,7],geneva:[2,7],genfromtxt:7,geometr:4,geometri:4,get:[0,1,6],git:7,github:7,give:7,given:[1,2,4,7],gj1214:7,global:[2,4],gnu:3,goal:[2,4],goe:2,going:4,good:[4,7],gpl:3,gpu:[4,5,6,7],gradient:4,grai:[1,4],gram:7,graphic:5,graviti:4,grid:[0,2,7],guarante:7,guid:7,h2o:[2,7],h5dump:7,h5py:[5,7],had:7,hand:[1,7],hansen:4,happen:4,hard:[2,7],hardcod:7,hardwar:[0,2],harpso:7,has:[0,1,2,4,5,7],have:[1,2,4,5,7],hcn:[],hdf5:[2,4,7],header:[4,7],heat:[4,6,7],heavi:6,height:[4,7],helio:[1,2,3,4,5,6],help:[1,4],helper:6,hemispher:[1,4],henc:4,heng:4,here:[2,3,4,5,6,7],hfl:7,high:[2,7],higher:[2,4,7],horizont:4,host:[4,5,6],host_funct:6,hot:4,how:[2,4,6,7],howev:[2,4,6,7],http:7,hung:4,idea:4,ideal:4,iii:[4,7],implement:[1,4,7],improv:4,inaccuraci:0,includ:[0,1,2,4,5,6],incom:4,incorpor:4,indefinit:4,independ:4,index:0,indic:4,individu:[0,1,4,7],individual_speci:2,induc:7,infinit:4,infinitesim:4,info:[0,6,7],inform:[2,3,4],ingredi:1,initi:[4,7],input:[0,1,6],insert:[2,7],insert_target_dir:7,inspect:7,instal:[0,1,2,5,6],install_input_fil:7,instanc:[2,4],instantan:4,instead:[2,4,7],integr:4,intens:4,interest:[4,6],interfac:[2,4,7],intern:4,interpol:[2,7],interpret:5,interv:4,invers:4,ion:[0,2],irradi:[4,7],isotherm:4,isotrop:[1,4],issu:[1,4,7],iter:0,its:[1,2,6,7],job:4,john:2,jupit:4,just:[2,7],kappa:[0,4],keep:4,kelvin:4,kept:[4,6],kernel:6,kim:2,kind:[1,6,7],kinet:0,kink:4,know:4,known:4,koll:4,kretzschmar:2,ktabl:[0,1,5,6,7],laci:4,lambda:2,larg:[4,7],larger:7,last:[2,4,7],lastli:[2,7],later:7,latter:[4,7],law:4,layer:[4,7],lead:[4,7],least:[2,4,6,7],leav:4,lee:2,leewai:7,left:4,less:4,let:7,letter:[2,4],level:6,lib:7,librari:[5,7],licens:[0,6],like:[5,6,7],limit:[2,4,7],line:[2,4,6,7],linear:[4,7],link:[],linux:7,list:[2,4,7],literatur:4,live:6,load:4,local:[4,7],locat:[2,4,6,7],lock:4,lodder:7,log10:[4,7],log:7,logarithm:7,logk:7,logk_wo_ion:7,lognorm:4,longwav:4,look:[2,4,7],loop:4,low:7,lower:[2,4,7],lowest:4,mac:[5,7],machin:[5,7],made:[4,7],magic:6,magnitud:4,mai:[4,5,7],main:[0,4,6,7],main_loop:7,major:4,make:[2,4,7],malik:[0,1,4],manag:[5,6],mani:[2,4,7],manual:[2,4,7],manufactur:5,mark:2,match:7,matplotlib:[5,6],matrix:4,matter:7,maximum:4,mean:[2,4,7],measur:[4,7],memori:[4,6],mere:[2,7],met:5,metal:7,method:[0,1,2,4],micron:[2,4,7],mie:[4,6,7],might:4,min:4,minor:4,miss:7,mix:[0,2],mixed_opac_kdistr:2,mixed_opac_sampl:2,mode:[4,7],model:[1,4,7],model_main:7,model_src:7,modifi:[2,4,6,7],modul:0,modular:5,molar:[2,7],mole:7,molecul:[2,6,7],molecular:[1,2,7],moment:2,monitor:7,more:[0,2,3,4,5,6],most:[2,4,6,7],move:4,movi:4,much:[4,7],multipl:[1,4,7],multipli:4,murphi:2,muscl:7,must:4,mybubbl:7,name:[2,4,6,7],natur:4,neat:6,necessari:[2,4,7],need:[1,2,4,6,7],neg:4,neglig:4,neptun:[4,7],net:4,neutral:2,never:[2,4],newer:5,newest:[4,7],newton:7,next:[2,4,7],nh3:[],night:4,nightsid:4,no_atmospher:4,non:[1,4,7],normal:4,note:[1,2,4,5,7],now:[1,6,7],numba:5,number:[2,4,7],numer:[2,4],numpi:[5,7],nvcc:[5,7],nvidia:[5,7],obtain:[2,4,7],obvious:[2,7],occur:4,off:4,offer:7,often:7,oina:4,old:[0,4],older:4,onc:[4,7],one:[1,2,4,6,7],ones:[2,6,7],onli:[1,2,4,7],onlin:[2,7],opac:[0,1,2,6],opac_file_for_lambdagrid:7,opaqu:4,open:[1,7],oper:5,optic:4,option:[1,2,4,6,7],orbit:4,order:[2,4,6,7],ordin:6,orient:4,oscil:4,other:[1,5,6,7],otherwis:[4,5,7],out:[4,7],outcom:4,outdat:5,output:[2,4,6,7],output_fil:7,over:[4,7],overal:4,overlap:[4,7],overridden:4,overwrit:4,overwritten:4,own:[1,2,4,7],packag:[1,2,5,7],page:[0,4,7],pair:[2,7],paper:[1,4],parallel:[1,4],param:[2,4,6,7],param_kt:[2,7],paramat:7,paramet:[0,1,6,7],parameter:[4,7],parameter_fil:[2,4,6],parent:6,part:[1,5,7],particl:7,path:[2,4,6,7],pdf:7,per:[2,4,7],perfect:7,perfectli:4,perform:4,perhap:[4,7],permiss:5,perspect:6,ph3:[],phoenix:7,photochem:0,photospher:4,phys_const:6,physic:[4,6],pick:7,pip:[5,7],plancktabl:4,plane:1,planet:[4,6,7],planet_databas:[4,6],planetari:[0,1,2],platform:1,pleas:[0,1,4],plot:[4,6,7],plot_and_tweak:7,plot_spectrum:7,plot_tp:7,plu:2,point:[2,4],pop:[4,7],popul:4,possess:7,possibl:[1,4,7],post:[4,7],power:4,practic:4,pre:[0,2,4,6],precis:4,prefer:[4,7],prefix:4,premix:[1,2,4,7],prepar:5,prerun:4,pressur:[2,4,7],prevent:4,previou:4,previous:[4,7],principl:[5,6],probabl:[4,6,7],problem:4,proce:7,procedur:7,proceed:5,process:[2,4,6,7],proclaim:7,produc:[2,4,7],product:[2,7],profil:[0,1,2,4],program:[0,1,5,6,7],progress:4,propag:4,properti:[2,4,7],provid:[2,4,7],proxi:4,pt_high:7,pt_low:7,pull:[2,7],pure:[4,6],purpos:[2,4],put:7,pycuda:[5,7],python3:[2,7],python:[0,7],quadratur:4,quantiti:[4,6],question:[1,7],quick:7,quickli:4,quit:[2,7],r50_kdistr:7,r50_kdistr_solar:[],r50_kdistr_solar_eq:7,r_jup:4,r_sun:4,rad:4,radi:[0,1,4,7],radiat:0,radii:[4,7],radiu:[4,7],ran:7,random:[4,7],rang:[2,4,7],raphson:7,ratio:[2,4,7],rayleigh:[2,4,7],reach:4,read:[2,4,6,7],readi:7,readm:6,real:[4,7],realli:4,realtim:[4,6],realtime_plot:6,reason:4,rebin:4,recal:4,recommend:[4,7],recov:4,redistribut:4,reduc:[2,4],refer:[1,2,4,7],regim:4,region:4,regular:4,reinstal:7,rel:[4,7],relax:4,relev:[4,7],remain:4,remot:7,remov:7,renam:[2,6],replac:[2,4,7],report:1,repositori:7,repres:4,reproduc:4,requir:[0,2,4,7],resolut:[2,4,7],resort:4,respect:[1,2,4,7],respons:[5,6],result:4,reus:4,revis:[0,4],right:6,rock:4,rocki:4,root:6,rorr:4,rule:4,run:[0,1,2,4,5,6],runtim:4,saha:2,same:[2,4,7],sampl:[1,2,7],satisfi:4,save:[4,7],save_ascii:7,save_in_hdf5:7,save_output:7,scalar:6,scale:[4,7],scat_cross_sect:7,scatter:[0,1,2,4,6],scenario:4,scheme:[4,7],scipi:5,script:[6,7],search:0,second:[0,4,7],secondari:7,secondli:[2,7],sect:4,section:[2,7],see:[2,3,4,5,6,7],select:2,self:7,send:0,sent:1,separ:[2,6,7],sequenc:[4,7],sequenti:7,server:7,set:[0,2,6],sever:[],shape:7,sharp:4,shell:[5,7],shift:4,shorter:6,should:[2,4,5,6,7],show:[4,7],shown:[4,7],similar:4,similarli:7,simpl:[4,7],simpli:[2,4,7],simul:[1,4,7],sinc:[2,4,5,7],singl:[2,4,7],singular:4,size:[2,4,7],skip:4,sleep:4,slower:[4,7],small:[2,4,7],smaller:4,smooth:4,sneep:2,softwar:5,solar:[4,7],solid:1,solut:[4,7],solv:[1,4],solver:[1,4],some:[4,5,6,7],someth:[4,7],somewhat:4,sorri:7,sourc:[1,2,4,6,7],source_kt:2,space:[2,4,7],speak:4,speci:[0,4,6,7],special:[4,7],species_databas:[2,6,7],specif:[2,4],specifi:7,spectra:[5,7],spectral:[2,4],spectrum:[0,1,2,4,6],speed:4,spheric:4,sphinx:6,sqrt:4,squar:[2,4],stabil:4,stage:[0,4],standard:[4,5,6,7],star:[4,7],star_2022:7,star_tool:7,start:[1,2,4],state:4,std:4,stefan:4,stellar:[0,5,6],step:[4,5,7],stepsiz:[4,7],still:[0,4],stop:[2,4],store:[4,6,7],straightforward:[2,7],straightforwardli:[4,7],stream:[1,4],strength:4,strict:4,strictli:4,string:4,stronger:4,structur:[0,4,7],stubborn:4,stuck:4,student:7,studi:1,stuff:6,sub:[4,7],subdirectori:[2,6,7],substellar:4,success:[4,6,7],successfulli:[5,7],suffic:5,suffici:[4,7],suggest:7,sum:4,suppli:7,support:2,sure:7,surfac:[1,4,7],surface_grav:4,surround:7,symbol:[2,4],system:[4,5,7],t_current:4,t_int:4,t_previou:4,tabl:[1,2,4,6,7],tabul:[0,2,4],take:[4,7],taken:[2,4],temperatur:[1,2,4,7],templat:7,term:[3,4,6,7],termin:[4,7],terrestri:4,tesla:5,test:[4,5,7],text:[2,4,7],thalman:2,than:[2,4,7],thank:1,thei:[5,6,7],them:[0,2,4,5,7],theori:[4,6,7],therein:7,thi:[0,2,4,5,6,7],thick:4,thin:4,thinner:4,third:4,thoma:4,thompson:7,thomson:2,thoroughli:4,those:[1,2,7],though:[2,4,6],three:[4,7],through:[2,4,6,7],throughout:[2,4],thu:[2,4,5,7],thumb:4,tidal:4,time:[4,7],timestep:4,tini:7,toa:4,toa_pressur:4,togeth:1,too:[2,4,7],tool:[2,6],toolkit:[5,7],toon:4,top:[4,6,7],total:4,transfer:[0,1,7],transit:4,transport:4,treat:4,tri:4,tridiagon:4,turn:[4,7],tutori:[0,5],tweak:7,twice:[4,7],two:[1,2,4,7],twofold:7,type:[2,4,5,7],typic:[4,7],typo:2,ubach:2,ubuntu:5,umd:[0,1],unbound:2,uncoupl:4,under:[3,4],underneath:2,underscor:[2,4],understand:6,understood:4,unfortun:4,uninstal:7,unit:[2,4,7],uniti:4,univers:[2,7],unpack:7,unstabl:4,until:[4,7],updat:[4,5],upper:2,uppermost:4,use:[1,2,4,7],used:[1,2,4,5,6,7],useful:[4,6,7],user:[4,5,6,7],uses:[4,7],using:[2,4,7],usr:7,usual:[2,4,7],util:5,valid:2,valu:[2,4,7],vanish:4,vari:7,variabl:6,varieti:[1,7],vast:[4,7],veri:[4,7],version:[0,1,3,4,5,7],vertic:[0,4],via:[2,4,6,7],viabl:4,virtual:[5,7],visibl:7,vmr:4,volum:[4,7],vram:5,w_0:4,wagner:2,wai:[2,4,7],want:[2,4,6],warn:4,wavelength:[2,4,7],wavelength_grid:7,wavenumb:7,weaker:4,wealth:7,weight:[2,6,7],welcom:7,well:[2,4,7],went:7,were:7,wget:5,what:[4,7],when:[2,4,7],whenev:4,where:[2,4,6,7],whether:[2,7],which:[1,2,4,5,6,7],whittak:1,whole:[2,4,7],width:4,window:[4,5,7],within:[4,6,7],without:[1,2,4,7],work:[1,2,5,6,7],workflow:[0,6],world:7,would:[4,6,7],wrapper:5,write:[2,4,6,7],written:[2,4,5,6,7],wrong:[4,7],yes:[2,4,7],yet:[4,7],you:[0,1,2,4,5,6,7],young:4,your:[1,2,5,7],yourself:[5,7],zenith:4,zero:4,zip:7,zone:4},titles:["Welcome to HELIOS v3.1","<strong>About</strong>","<strong>ktable Program</strong>","<strong>License</strong>","<strong>Input Parameters</strong>","<strong>Requirements</strong>","<strong>Code Structure</strong>","<strong>Tutorial</strong>"],titleterms:{"final":2,"new":7,Use:7,Using:7,about:1,adjust:4,advanc:4,c_p:7,chemistri:7,cia:7,cloud:[4,7],code:[2,6],compat:7,convect:4,coupl:[4,7],creat:7,cuda:5,differ:7,directori:6,download:7,file:[2,4,6,7],first:[2,7],fly:7,format:2,gener:[2,4],get:7,grid:4,hardwar:5,helio:[0,7],includ:7,indic:0,individu:2,info:[2,4],input:[2,4,7],instal:7,ion:7,iter:4,kappa:7,kinet:[4,7],ktabl:2,licens:3,main:2,method:7,mix:[4,7],more:7,old:7,opac:[4,7],paramet:[2,4],photochem:[4,7],planetari:4,pre:7,profil:7,program:2,python:5,radiat:4,requir:5,run:7,sampl:[],scatter:7,second:2,set:[4,7],speci:2,spectrum:7,stage:2,stellar:[4,7],still:7,structur:[2,6],tabl:0,tabul:7,tutori:7,vertic:7,welcom:0,workflow:2}})
docs/build/html/_sources/sections/structure.rst.txt:* ``quantities.py``: contains all scalar variables and arrays. It is responsible for data management, like copying arrays between the host and the device (GPU), and allocating memory. 
docs/build/html/_sources/sections/structure.rst.txt:* ``computation.py``: calls and co-ordinates the device kernels, i.e., functions living on the GPU. If you write a new GPU functionality (=kernel) include it here.
docs/build/html/_sources/sections/structure.rst.txt:* ``kernels.cu``: contains the detailed computations, executed on the GPU/device. Write new kernel functions or alter existing ones here.
docs/build/html/_sources/sections/requirements.rst.txt:HELIOS is a GPU-accelerated software developed with parts written in CUDA. It thus requires an NVIDIA graphics card (GPU) to operate on. Any GeForce or Tesla card manufactured since 2013 and with 2 GB VRAM or more should suffice to run standard applications of HELIOS.
docs/build/html/_sources/sections/requirements.rst.txt:CUDA
docs/build/html/_sources/sections/requirements.rst.txt:CUDA is the NVIDIA API responsible for the communication between the graphics card (aka device) and the CPU (aka host). The software package consists of the core libraries, development utilities and the NVCC compiler to interpret C/C++ code. The CUDA toolkit can be downloaded from `here <https://developer.nvidia.com/cuda-downloads>`_.
docs/build/html/_sources/sections/requirements.rst.txt:HELIOS has been tested with CUDA versions 7.x -- 11.x and should, in principle, also be compatible with any newer version.
docs/build/html/_sources/sections/requirements.rst.txt:HELIOS's computational core is written in CUDA C++, but the user shell comes in Python modular format. To communicate between the host and the device the PyCUDA wrapper is used.
docs/build/html/_sources/sections/requirements.rst.txt:* PyCUDA
docs/build/html/_sources/sections/requirements.rst.txt:Some of them may be already included in the python distribution (e.g., Anaconda). Otherwise they can be installed with the Python package manager pip. To install, e.g., PyCUDA type::
docs/build/html/_sources/sections/requirements.rst.txt:   pip install pycuda
docs/build/html/_sources/sections/tutorial.rst.txt:In the following I merely expect that you possess an NVIDIA GPU and are either on a Linux or Mac Os X system (sorry, Windows users. You are on your own).
docs/build/html/_sources/sections/tutorial.rst.txt:1. Install the newest version of the CUDA toolkit from `NVIDIA <https://developer.nvidia.com/cuda-downloads>`_. To ascertain a successful installation, type ``which nvcc``. This should provide you with the location of the nvcc compiler. If not, something went wrong with the installation. 
docs/build/html/_sources/sections/tutorial.rst.txt:Make sure that the library and program paths are exported. On Mac Os x you should have the following entries in your .bash_profile file (shown for version 10.0 of CUDA) ::
docs/build/html/_sources/sections/tutorial.rst.txt:	export PATH=/Developer/NVIDIA/CUDA-10.0/bin:$PATH
docs/build/html/_sources/sections/tutorial.rst.txt:	export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-10.0/lib:$DYLD_LIBRARY_PATH
docs/build/html/_sources/sections/tutorial.rst.txt:	export PATH=/usr/local/cuda-10.0/bin:$PATH
docs/build/html/_sources/sections/tutorial.rst.txt:	export DYLD_LIBRARY_PATH=/usr/local/cuda-10.0/lib:$DYLD_LIBRARY_PATH
docs/build/html/_sources/sections/tutorial.rst.txt:  to activate the environment. Then we need to uninstall and reinstall pycuda package (the pre--installed version does not work usually). Type ::
docs/build/html/_sources/sections/tutorial.rst.txt:	pip uninstall pycuda
docs/build/html/_sources/sections/tutorial.rst.txt:	pip install pycuda
docs/build/html/_sources/sections/parameters.rst.txt:This changes the numerical precision used for the GPU calculations. However, I have never found any difference in terms of accuracy or speed when switching between single and double precision. (Perhaps I did something wrong). It is probably the best to leave it at 'double'.
docs/conf.py:	'description': 'GPU-accelerated radiative transfer code for exoplanetary atmospheres',
source/quantities.py:import pycuda.driver as cuda
source/quantities.py:import pycuda.autoinit
source/quantities.py:import pycuda.gpuarray as gpuarray
source/quantities.py:        # input arrays to be copied CPU --> GPU
source/quantities.py:        # and then copied to GPU with "gpuarray"
source/quantities.py:        # arrays to be filled with species data in the on-the-fly opacity mixing mode (i.e., CPU --> GPU)
source/quantities.py:        # arrays to be copied CPU --> GPU --> CPU
source/quantities.py:        # these are copied to GPU by "gpuarray" and copied back
source/quantities.py:        # arrays to be copied GPU --> CPU
source/quantities.py:        # for these, zero arrays of correct size are created and then copied to GPU with "gpuarray" and copied back
source/quantities.py:        # arrays exclusively used on the GPU
source/quantities.py:        # these are defined directly on the GPU and stay there. No copying required.
source/quantities.py:        # for on-the-fly opacity mixing mode (only GPU arrays)
source/quantities.py:        """ creates zero arrays of quantities to be used on the GPU with the correct length/dimension """
source/quantities.py:        self.dev_p_lay = gpuarray.to_gpu(self.p_lay)
source/quantities.py:        self.dev_p_int = gpuarray.to_gpu(self.p_int)
source/quantities.py:        self.dev_delta_colmass = gpuarray.to_gpu(self.delta_colmass)
source/quantities.py:        self.dev_delta_col_upper = gpuarray.to_gpu(self.delta_col_upper)
source/quantities.py:        self.dev_delta_col_lower = gpuarray.to_gpu(self.delta_col_lower)
source/quantities.py:        self.dev_ktemp = gpuarray.to_gpu(self.ktemp)
source/quantities.py:        self.dev_kpress = gpuarray.to_gpu(self.kpress)
source/quantities.py:        self.dev_entr_temp = gpuarray.to_gpu(self.entr_temp)
source/quantities.py:        self.dev_entr_press = gpuarray.to_gpu(self.entr_press)
source/quantities.py:        self.dev_opac_k = gpuarray.to_gpu(self.opac_k)
source/quantities.py:        self.dev_gauss_y = gpuarray.to_gpu(self.gauss_y)
source/quantities.py:        self.dev_gauss_weight = gpuarray.to_gpu(self.gauss_weight)
source/quantities.py:        self.dev_opac_wave = gpuarray.to_gpu(self.opac_wave)
source/quantities.py:        self.dev_opac_deltawave = gpuarray.to_gpu(self.opac_deltawave)
source/quantities.py:        self.dev_opac_interwave = gpuarray.to_gpu(self.opac_interwave)
source/quantities.py:        self.dev_opac_scat_cross = gpuarray.to_gpu(self.opac_scat_cross)
source/quantities.py:        self.dev_opac_meanmass = gpuarray.to_gpu(self.opac_meanmass)
source/quantities.py:        self.dev_entr_kappa = gpuarray.to_gpu(self.entr_kappa)
source/quantities.py:        self.dev_entr_c_p = gpuarray.to_gpu(self.entr_c_p)
source/quantities.py:        self.dev_entr_phase_number = gpuarray.to_gpu(self.entr_phase_number)
source/quantities.py:        self.dev_entr_entropy = gpuarray.to_gpu(self.entr_entropy)
source/quantities.py:        self.dev_c_p_lay = gpuarray.to_gpu(self.c_p_lay)
source/quantities.py:        self.dev_kappa_lay = gpuarray.to_gpu(self.kappa_lay)
source/quantities.py:        self.dev_starflux = gpuarray.to_gpu(self.starflux)
source/quantities.py:        self.dev_T_lay = gpuarray.to_gpu(self.T_lay)
source/quantities.py:        self.dev_surf_albedo = gpuarray.to_gpu(self.surf_albedo)
source/quantities.py:        self.dev_abs_cross_all_clouds_lay = gpuarray.to_gpu(self.abs_cross_all_clouds_lay)
source/quantities.py:        self.dev_scat_cross_all_clouds_lay = gpuarray.to_gpu(self.scat_cross_all_clouds_lay)
source/quantities.py:        self.dev_g_0_all_clouds_lay = gpuarray.to_gpu(self.g_0_all_clouds_lay)
source/quantities.py:        # zero arrays (copying anyway to obtain the gpuarray functionality)
source/quantities.py:        self.dev_F_up_band = gpuarray.to_gpu(self.F_up_band)
source/quantities.py:        self.dev_F_down_band = gpuarray.to_gpu(self.F_down_band)
source/quantities.py:        self.dev_F_dir_band = gpuarray.to_gpu(self.F_dir_band)
source/quantities.py:        self.dev_F_down_wg = gpuarray.to_gpu(self.F_down_wg)
source/quantities.py:        self.dev_F_up_wg = gpuarray.to_gpu(self.F_up_wg)
source/quantities.py:        self.dev_F_dir_wg = gpuarray.to_gpu(self.F_dir_wg)
source/quantities.py:        self.dev_F_up_tot = gpuarray.to_gpu(self.F_up_tot)
source/quantities.py:        self.dev_F_down_tot = gpuarray.to_gpu(self.F_down_tot)
source/quantities.py:        self.dev_F_dir_tot = gpuarray.to_gpu(self.F_dir_tot)
source/quantities.py:        self.dev_opac_band_lay = gpuarray.to_gpu(self.opac_band_lay)
source/quantities.py:        self.dev_opac_wg_lay = gpuarray.to_gpu(self.opac_wg_lay)
source/quantities.py:        self.dev_opac_wg_int = gpuarray.to_gpu(self.opac_wg_int)
source/quantities.py:        self.dev_scat_cross_lay = gpuarray.to_gpu(self.scat_cross_lay)
source/quantities.py:        self.dev_scat_cross_int = gpuarray.to_gpu(self.scat_cross_int)
source/quantities.py:        self.dev_F_net = gpuarray.to_gpu(self.F_net)
source/quantities.py:        self.dev_F_net_diff = gpuarray.to_gpu(self.F_net_diff)
source/quantities.py:        self.dev_planckband_lay = gpuarray.to_gpu(self.planckband_lay)
source/quantities.py:        self.dev_planck_opac_T_pl = gpuarray.to_gpu(self.planck_opac_T_pl)
source/quantities.py:        self.dev_ross_opac_T_pl = gpuarray.to_gpu(self.ross_opac_T_pl)
source/quantities.py:        self.dev_planck_opac_T_star = gpuarray.to_gpu(self.planck_opac_T_star)
source/quantities.py:        self.dev_ross_opac_T_star = gpuarray.to_gpu(self.ross_opac_T_star)
source/quantities.py:        self.dev_trans_band = gpuarray.to_gpu(self.trans_band)
source/quantities.py:        self.dev_delta_tau_band = gpuarray.to_gpu(self.delta_tau_band)
source/quantities.py:        self.dev_abort = gpuarray.to_gpu(self.abort)
source/quantities.py:        self.dev_meanmolmass_lay = gpuarray.to_gpu(self.meanmolmass_lay)
source/quantities.py:        self.dev_meanmolmass_int = gpuarray.to_gpu(self.meanmolmass_int)
source/quantities.py:        self.dev_entropy_lay = gpuarray.to_gpu(self.entropy_lay)
source/quantities.py:        self.dev_phase_number_lay = gpuarray.to_gpu(self.phase_number_lay)
source/quantities.py:        self.dev_trans_weight_band = gpuarray.to_gpu(self.trans_weight_band)
source/quantities.py:        self.dev_contr_func_band = gpuarray.to_gpu(self.contr_func_band)
source/quantities.py:        self.dev_delta_z_lay = gpuarray.to_gpu(self.delta_z_lay)
source/quantities.py:        self.dev_z_lay = gpuarray.to_gpu(self.z_lay)
source/quantities.py:        self.dev_g_0_tot_lay = gpuarray.to_gpu(self.g_0_tot_lay)
source/quantities.py:        self.dev_T_int = gpuarray.to_gpu(self.T_int)
source/quantities.py:        self.dev_scat_trigger = gpuarray.to_gpu(self.scat_trigger)
source/quantities.py:        self.dev_delta_tau_all_clouds = gpuarray.to_gpu(self.delta_tau_all_clouds)
source/quantities.py:        self.dev_F_add_heat_lay = gpuarray.to_gpu(self.F_add_heat_lay)
source/quantities.py:        self.dev_F_add_heat_sum = gpuarray.to_gpu(self.F_add_heat_sum)
source/quantities.py:        self.dev_F_smooth = gpuarray.to_gpu(self.F_smooth)
source/quantities.py:        self.dev_F_smooth_sum = gpuarray.to_gpu(self.F_smooth_sum)
source/quantities.py:            self.dev_planckband_int = gpuarray.to_gpu(self.planckband_int)
source/quantities.py:            self.dev_Fc_down_wg = gpuarray.to_gpu(self.Fc_down_wg)
source/quantities.py:            self.dev_Fc_up_wg = gpuarray.to_gpu(self.Fc_up_wg)
source/quantities.py:            self.dev_Fc_dir_wg = gpuarray.to_gpu(self.Fc_dir_wg)
source/quantities.py:            self.dev_g_0_tot_int = gpuarray.to_gpu(self.g_0_tot_int)
source/quantities.py:            self.dev_abs_cross_all_clouds_int = gpuarray.to_gpu(self.abs_cross_all_clouds_int)
source/quantities.py:            self.dev_scat_cross_all_clouds_int = gpuarray.to_gpu(self.scat_cross_all_clouds_int)
source/quantities.py:            self.dev_g_0_all_clouds_int = gpuarray.to_gpu(self.g_0_all_clouds_int)
source/quantities.py:            self.dev_kappa_int = gpuarray.to_gpu(self.kappa_int)
source/quantities.py:        """ allocate memory for arrays existing only on the GPU """
source/quantities.py:        self.dev_delta_t_prefactor = cuda.mem_alloc(size_nlayer_plus1)
source/quantities.py:        self.dev_T_store = cuda.mem_alloc(size_nlayer_plus1)
source/quantities.py:        self.dev_planckband_grid = cuda.mem_alloc(size_nplanckgrid)
source/quantities.py:        self.dev_opac_wg_lay = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:        self.dev_delta_tau_wg = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:        self.dev_trans_wg = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:        self.dev_w_0 = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:        self.dev_M_term = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:        self.dev_N_term = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:        self.dev_P_term = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:        self.dev_G_plus = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:        self.dev_G_minus = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:            self.dev_opac_spec_wg_lay = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:            self.dev_delta_tau_wg_upper = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:            self.dev_delta_tau_wg_lower = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:            self.dev_trans_wg_upper = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:            self.dev_trans_wg_lower = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:            self.dev_M_upper = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:            self.dev_N_upper = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:            self.dev_P_upper = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:            self.dev_M_lower = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:            self.dev_N_lower = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:            self.dev_P_lower = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:            self.dev_w_0_upper = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:            self.dev_w_0_lower = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:            self.dev_G_plus_upper = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:            self.dev_G_plus_lower = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:            self.dev_G_minus_upper = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:            self.dev_G_minus_lower = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:            self.dev_delta_tau_all_clouds_upper = cuda.mem_alloc(size_nlayer_nbin)
source/quantities.py:            self.dev_delta_tau_all_clouds_lower = cuda.mem_alloc(size_nlayer_nbin)
source/quantities.py:                self.dev_opac_spec_wg_int = cuda.mem_alloc(size_ninterface_wg_nbin)
source/quantities.py:                self.dev_alpha = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:                self.dev_beta = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:                self.dev_source_term_down = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:                self.dev_source_term_up = cuda.mem_alloc(size_nlayer_wg_nbin)
source/quantities.py:                self.dev_alpha = cuda.mem_alloc(size_2_nlayer_wg_nbin)
source/quantities.py:                self.dev_beta = cuda.mem_alloc(size_2_nlayer_wg_nbin)
source/quantities.py:                self.dev_source_term_down = cuda.mem_alloc(size_2_nlayer_wg_nbin)
source/quantities.py:                self.dev_source_term_up = cuda.mem_alloc(size_2_nlayer_wg_nbin)
source/quantities.py:            self.dev_c_prime = cuda.mem_alloc(size_nmatrix_wg_nbin)
source/quantities.py:            self.dev_d_prime = cuda.mem_alloc(size_nmatrix_wg_nbin)
source/kernels.cu:// This file contains all the device functions and CUDA kernels.
source/read.py:    def set_prec_in_cudafile(quant):
source/read.py:        with open("./source/kernels.cu", "r") as cudafile:
source/read.py:            contents = cudafile.readlines()
source/read.py:                    print("\nRewriting Cuda-sourcefile for single precision.")
source/read.py:                    print("\nRewriting Cuda-sourcefile for double precision.")
source/read.py:            with open("./source/kernels.cu", "w") as cudafile:
source/read.py:                cudafile.write(contents)
source/read.py:        self.set_prec_in_cudafile(quant)
source/read.py:                # convert to numpy arrays so they have the correct format for copying to GPU
source/read.py:                # convert to numpy array (necessary for GPU copying)
source/host_functions.py:import pycuda.driver as cuda
source/host_functions.py:import pycuda.autoinit
source/host_functions.py:import pycuda.gpuarray as gpuarray
source/host_functions.py:from pycuda.compiler import SourceModule
source/host_functions.py:#     quant.dev_F_up_tot = gpuarray.to_gpu(quant.F_up_tot)
source/host_functions.py:#     quant.dev_F_net = gpuarray.to_gpu(quant.F_net)
source/host_functions.py:#     # get arrays from GPU
source/host_functions.py:            # convert to numpy arrays in order to have the correct format for copying to GPU
source/host_functions.py:    quant.dev_meanmolmass_lay = gpuarray.to_gpu(quant.meanmolmass_lay)
source/host_functions.py:        quant.dev_meanmolmass_int = gpuarray.to_gpu(quant.meanmolmass_int)
source/host_functions.py:    quant.dev_opac_wg_lay = gpuarray.to_gpu(quant.opac_wg_lay)
source/host_functions.py:    quant.dev_opac_wg_int = gpuarray.to_gpu(quant.opac_wg_int)
source/host_functions.py:    quant.dev_scat_cross_lay = gpuarray.to_gpu(quant.scat_cross_lay)
source/host_functions.py:    quant.dev_scat_cross_int = gpuarray.to_gpu(quant.scat_cross_int)
source/computation.py:import pycuda.driver as cuda
source/computation.py:import pycuda.autoinit
source/computation.py:import pycuda.gpuarray as gpuarray
source/computation.py:from pycuda.compiler import SourceModule
source/computation.py:        cuda.Context.synchronize()
source/computation.py:            cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:            cuda.Context.synchronize()
source/computation.py:                cuda.Context.synchronize()
source/computation.py:            cuda.Context.synchronize()
source/computation.py:            cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:            cuda.Context.synchronize()
source/computation.py:        quant.dev_scat_trigger = gpuarray.to_gpu(quant.scat_trigger)
source/computation.py:        cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:            cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:        start_loop = cuda.Event()
source/computation.py:        end_loop = cuda.Event()
source/computation.py:        start_total = cuda.Event()
source/computation.py:        end_total = cuda.Event()
source/computation.py:        # start_test = cuda.Event()
source/computation.py:        # end_test = cuda.Event()
source/computation.py:                quant.dev_z_lay = gpuarray.to_gpu(quant.z_lay)
source/computation.py:                            quant.dev_F_add_heat_lay = gpuarray.to_gpu(quant.F_add_heat_lay)
source/computation.py:                            quant.dev_F_add_heat_sum = gpuarray.to_gpu(quant.F_add_heat_sum)
source/computation.py:            start_total = cuda.Event()
source/computation.py:            end_total = cuda.Event()
source/computation.py:                start_loop = cuda.Event()
source/computation.py:                end_loop = cuda.Event()
source/computation.py:                quant.dev_T_lay = gpuarray.to_gpu(quant.T_lay)
source/computation.py:                    quant.dev_z_lay = gpuarray.to_gpu(quant.z_lay)
source/computation.py:                            quant.dev_F_add_heat_lay = gpuarray.to_gpu(quant.F_add_heat_lay)
source/computation.py:                            quant.dev_F_add_heat_sum = gpuarray.to_gpu(quant.F_add_heat_sum)
source/computation.py:                    quant.dev_conv_layer = gpuarray.to_gpu(quant.conv_layer)
source/computation.py:                    quant.dev_marked_red = gpuarray.to_gpu(quant.marked_red)
source/computation.py:        cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:            cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:            cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:            cuda.Context.synchronize()
source/computation.py:        cuda.Context.synchronize()
source/computation.py:            cuda.Context.synchronize()
source/computation.py:            # copy arrays to GPU
source/computation.py:            quant.dev_vmr_spec_lay = gpuarray.to_gpu(quant.species_list[s].vmr_layer)
source/computation.py:                quant.dev_vmr_spec_int = gpuarray.to_gpu(quant.species_list[s].vmr_interface)
source/computation.py:                quant.dev_opacity_spec_pretab = gpuarray.to_gpu(quant.species_list[s].opacity_pretab)
source/computation.py:                    quant.dev_scat_cross_spec_lay = gpuarray.to_gpu(np.zeros(quant.nbin * quant.nlayer, quant.fl_prec))
source/computation.py:                        quant.dev_scat_cross_spec_int = gpuarray.to_gpu(np.zeros(quant.nbin * quant.ninterface, quant.fl_prec))
source/computation.py:                # else scattering arrays are already read in from file and just needs to be copied to GPU
source/computation.py:                    quant.dev_scat_cross_spec_lay = gpuarray.to_gpu(quant.species_list[s].scat_cross_sect_layer)
source/computation.py:                        quant.dev_scat_cross_spec_int = gpuarray.to_gpu(quant.species_list[s].scat_cross_sect_interface)
README.md:#### A GPU-ACCELERATED RADIATIVE TRANSFER CODE FOR EXOPLANETARY ATMOSPHERES ####

```
