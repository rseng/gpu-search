# https://github.com/xianggebenben/GraphSL

```console
GraphSL/GNN/SLVAE/model.py:            - device (torch.device): Device to be used for computation, cpu or cuda.
GraphSL/GNN/SLVAE/model.py:        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GraphSL/GNN/SLVAE/model.py:        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GraphSL/GNN/SLVAE/main.py:        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GraphSL/GNN/SLVAE/main.py:        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GraphSL/GNN/SLVAE/main.py:        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GraphSL/GNN/IVGD/diffusion_model.py:        # Determine whether to use GPU or CPU
GraphSL/GNN/IVGD/diffusion_model.py:        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GraphSL/GNN/IVGD/diffusion_model.py:        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GraphSL/GNN/IVGD/main.py:        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GraphSL/GNN/IVGD/main.py:        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GraphSL/GNN/GCNSI/main.py:        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GraphSL/Prescribed.py:        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GraphSL/Prescribed.py:        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GraphSL/Prescribed.py:        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
docs/_build/html/searchindex.js:Search.setIndex({"alltitles": {"Contact": [[5, "contact"]], "Contents:": [[5, null]], "GraphSL": [[6, "graphsl"]], "GraphSL package": [[0, "graphsl-package"]], "GraphSL.GNN package": [[1, "graphsl-gnn-package"]], "GraphSL.GNN.GCNSI package": [[2, "graphsl-gnn-gcnsi-package"]], "GraphSL.GNN.GCNSI.main module": [[2, "module-GraphSL.GNN.GCNSI.main"]], "GraphSL.GNN.GCNSI.model module": [[2, "module-GraphSL.GNN.GCNSI.model"]], "GraphSL.GNN.IVGD package": [[3, "graphsl-gnn-ivgd-package"]], "GraphSL.GNN.IVGD.correction module": [[3, "module-GraphSL.GNN.IVGD.correction"]], "GraphSL.GNN.IVGD.diffusion_model module": [[3, "module-GraphSL.GNN.IVGD.diffusion_model"]], "GraphSL.GNN.IVGD.main module": [[3, "module-GraphSL.GNN.IVGD.main"]], "GraphSL.GNN.IVGD.validity_net module": [[3, "module-GraphSL.GNN.IVGD.validity_net"]], "GraphSL.GNN.SLVAE package": [[4, "graphsl-gnn-slvae-package"]], "GraphSL.GNN.SLVAE.main module": [[4, "module-GraphSL.GNN.SLVAE.main"]], "GraphSL.GNN.SLVAE.model module": [[4, "module-GraphSL.GNN.SLVAE.model"]], "GraphSL.Prescribed module": [[0, "module-GraphSL.Prescribed"]], "GraphSL.utils module": [[0, "module-GraphSL.utils"]], "Indices and tables": [[5, "indices-and-tables"]], "Installation": [[5, "installation"]], "Module contents": [[0, "module-GraphSL"], [1, "module-GraphSL.GNN"], [2, "module-GraphSL.GNN.GCNSI"], [3, "module-GraphSL.GNN.IVGD"], [4, "module-GraphSL.GNN.SLVAE"]], "Quickstart Guide": [[5, "quickstart-guide"]], "Submodules": [[0, "submodules"], [2, "submodules"], [3, "submodules"], [4, "submodules"]], "Subpackages": [[0, "subpackages"], [1, "subpackages"]], "Usage": [[5, "usage"]], "Welcome to GraphSL\u2019s documentation!": [[5, "welcome-to-graphsl-s-documentation"]]}, "docnames": ["GraphSL", "GraphSL.GNN", "GraphSL.GNN.GCNSI", "GraphSL.GNN.IVGD", "GraphSL.GNN.SLVAE", "index", "modules"], "envversion": {"sphinx": 61, "sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.todo": 2, "sphinx.ext.viewcode": 1}, "filenames": ["GraphSL.rst", "GraphSL.GNN.rst", "GraphSL.GNN.GCNSI.rst", "GraphSL.GNN.IVGD.rst", "GraphSL.GNN.SLVAE.rst", "index.rst", "modules.rst"], "indexentries": {"backward() (graphsl.gnn.ivgd.diffusion_model.i_gcn method)": [[3, "GraphSL.GNN.IVGD.diffusion_model.I_GCN.backward", false]], "correction (class in graphsl.gnn.ivgd.correction)": [[3, "GraphSL.GNN.IVGD.correction.correction", false]], "correction() (graphsl.gnn.ivgd.validity_net.validity_net method)": [[3, "GraphSL.GNN.IVGD.validity_net.validity_net.correction", false]], "forward() (graphsl.gnn.ivgd.correction.correction method)": [[3, "GraphSL.GNN.IVGD.correction.correction.forward", false]], "forward() (graphsl.gnn.ivgd.diffusion_model.i_gcn method)": [[3, "GraphSL.GNN.IVGD.diffusion_model.I_GCN.forward", false]], "forward() (graphsl.gnn.ivgd.diffusion_model.i_gcnlayer method)": [[3, "GraphSL.GNN.IVGD.diffusion_model.I_GCNLayer.forward", false]], "forward() (graphsl.gnn.ivgd.main.ivgd_model method)": [[3, "GraphSL.GNN.IVGD.main.IVGD_model.forward", false]], "forward() (graphsl.gnn.ivgd.validity_net.validity_net method)": [[3, "GraphSL.GNN.IVGD.validity_net.validity_net.forward", false]], "graphsl.gnn": [[1, "module-GraphSL.GNN", false]], "graphsl.gnn.ivgd": [[3, "module-GraphSL.GNN.IVGD", false]], "graphsl.gnn.ivgd.correction": [[3, "module-GraphSL.GNN.IVGD.correction", false]], "graphsl.gnn.ivgd.diffusion_model": [[3, "module-GraphSL.GNN.IVGD.diffusion_model", false]], "graphsl.gnn.ivgd.main": [[3, "module-GraphSL.GNN.IVGD.main", false]], "graphsl.gnn.ivgd.validity_net": [[3, "module-GraphSL.GNN.IVGD.validity_net", false]], "i_gcn (class in graphsl.gnn.ivgd.diffusion_model)": [[3, "GraphSL.GNN.IVGD.diffusion_model.I_GCN", false]], "i_gcnlayer (class in graphsl.gnn.ivgd.diffusion_model)": [[3, "GraphSL.GNN.IVGD.diffusion_model.I_GCNLayer", false]], "ivgd (class in graphsl.gnn.ivgd.main)": [[3, "GraphSL.GNN.IVGD.main.IVGD", false]], "ivgd_model (class in graphsl.gnn.ivgd.main)": [[3, "GraphSL.GNN.IVGD.main.IVGD_model", false]], "module": [[1, "module-GraphSL.GNN", false], [3, "module-GraphSL.GNN.IVGD", false], [3, "module-GraphSL.GNN.IVGD.correction", false], [3, "module-GraphSL.GNN.IVGD.diffusion_model", false], [3, "module-GraphSL.GNN.IVGD.main", false], [3, "module-GraphSL.GNN.IVGD.validity_net", false]], "normalize() (graphsl.gnn.ivgd.main.ivgd method)": [[3, "GraphSL.GNN.IVGD.main.IVGD.normalize", false]], "test() (graphsl.gnn.ivgd.main.ivgd method)": [[3, "GraphSL.GNN.IVGD.main.IVGD.test", false]], "train() (graphsl.gnn.ivgd.main.ivgd method)": [[3, "GraphSL.GNN.IVGD.main.IVGD.train", false]], "train_diffusion() (graphsl.gnn.ivgd.main.ivgd method)": [[3, "GraphSL.GNN.IVGD.main.IVGD.train_diffusion", false]], "validity_net (class in graphsl.gnn.ivgd.validity_net)": [[3, "GraphSL.GNN.IVGD.validity_net.validity_net", false]]}, "objects": {"": [[0, 0, 0, "-", "GraphSL"]], "GraphSL": [[1, 0, 0, "-", "GNN"], [0, 0, 0, "-", "Prescribed"], [0, 0, 0, "-", "utils"]], "GraphSL.GNN": [[2, 0, 0, "-", "GCNSI"], [3, 0, 0, "-", "IVGD"], [4, 0, 0, "-", "SLVAE"]], "GraphSL.GNN.GCNSI": [[2, 0, 0, "-", "main"], [2, 0, 0, "-", "model"]], "GraphSL.GNN.GCNSI.main": [[2, 1, 1, "", "GCNSI"]], "GraphSL.GNN.GCNSI.main.GCNSI": [[2, 2, 1, "", "test"], [2, 2, 1, "", "train"]], "GraphSL.GNN.GCNSI.model": [[2, 1, 1, "", "GCNConv"], [2, 1, 1, "", "GCNSI_model"]], "GraphSL.GNN.GCNSI.model.GCNConv": [[2, 2, 1, "", "forward"]], "GraphSL.GNN.GCNSI.model.GCNSI_model": [[2, 2, 1, "", "forward"]], "GraphSL.GNN.IVGD": [[3, 0, 0, "-", "correction"], [3, 0, 0, "-", "diffusion_model"], [3, 0, 0, "-", "main"], [3, 0, 0, "-", "validity_net"]], "GraphSL.GNN.IVGD.correction": [[3, 1, 1, "", "correction"]], "GraphSL.GNN.IVGD.correction.correction": [[3, 2, 1, "", "forward"]], "GraphSL.GNN.IVGD.diffusion_model": [[3, 1, 1, "", "I_GCN"], [3, 1, 1, "", "I_GCNLayer"]], "GraphSL.GNN.IVGD.diffusion_model.I_GCN": [[3, 2, 1, "", "backward"], [3, 2, 1, "", "forward"]], "GraphSL.GNN.IVGD.diffusion_model.I_GCNLayer": [[3, 2, 1, "", "forward"]], "GraphSL.GNN.IVGD.main": [[3, 1, 1, "", "IVGD"], [3, 1, 1, "", "IVGD_model"]], "GraphSL.GNN.IVGD.main.IVGD": [[3, 2, 1, "", "normalize"], [3, 2, 1, "", "test"], [3, 2, 1, "", "train"], [3, 2, 1, "", "train_diffusion"]], "GraphSL.GNN.IVGD.main.IVGD_model": [[3, 2, 1, "", "forward"]], "GraphSL.GNN.IVGD.validity_net": [[3, 1, 1, "", "validity_net"]], "GraphSL.GNN.IVGD.validity_net.validity_net": [[3, 2, 1, "", "correction"], [3, 2, 1, "", "forward"]], "GraphSL.GNN.SLVAE": [[4, 0, 0, "-", "main"], [4, 0, 0, "-", "model"]], "GraphSL.GNN.SLVAE.main": [[4, 1, 1, "", "SLVAE"], [4, 1, 1, "", "SLVAE_model"]], "GraphSL.GNN.SLVAE.main.SLVAE": [[4, 2, 1, "", "infer"], [4, 2, 1, "", "train"]], "GraphSL.GNN.SLVAE.main.SLVAE_model": [[4, 2, 1, "", "forward"], [4, 2, 1, "", "infer_loss"], [4, 2, 1, "", "train_loss"]], "GraphSL.GNN.SLVAE.model": [[4, 1, 1, "", "Decoder"], [4, 1, 1, "", "Encoder"], [4, 1, 1, "", "GCNLayer"], [4, 1, 1, "", "GNN"], [4, 1, 1, "", "VAE"]], "GraphSL.GNN.SLVAE.model.Decoder": [[4, 2, 1, "", "forward"]], "GraphSL.GNN.SLVAE.model.Encoder": [[4, 2, 1, "", "forward"]], "GraphSL.GNN.SLVAE.model.GCNLayer": [[4, 2, 1, "", "forward"]], "GraphSL.GNN.SLVAE.model.GNN": [[4, 2, 1, "", "forward"], [4, 2, 1, "", "loss"]], "GraphSL.GNN.SLVAE.model.VAE": [[4, 2, 1, "", "decode"], [4, 2, 1, "", "encode"], [4, 2, 1, "", "forward"], [4, 2, 1, "", "reparameterization"]], "GraphSL.Prescribed": [[0, 1, 1, "", "LPSI"], [0, 1, 1, "", "NetSleuth"], [0, 1, 1, "", "OJC"]], "GraphSL.Prescribed.LPSI": [[0, 2, 1, "", "predict"], [0, 2, 1, "", "test"], [0, 2, 1, "", "train"]], "GraphSL.Prescribed.NetSleuth": [[0, 2, 1, "", "predict"], [0, 2, 1, "", "test"], [0, 2, 1, "", "train"]], "GraphSL.Prescribed.OJC": [[0, 2, 1, "", "Candidate"], [0, 2, 1, "", "get_K_list"], [0, 2, 1, "", "predict"], [0, 2, 1, "", "test"], [0, 2, 1, "", "train"]], "GraphSL.utils": [[0, 1, 1, "", "Metric"], [0, 3, 1, "", "diffusion_generation"], [0, 3, 1, "", "download_dataset"], [0, 3, 1, "", "generate_seed_vector"], [0, 3, 1, "", "load_dataset"], [0, 3, 1, "", "split_dataset"], [0, 3, 1, "", "visualize_source_prediction"]]}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"], "3": ["py", "function", "Python function"]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method", "3": "py:function"}, "terms": {"": [], "0": [0, 2, 3, 4, 5], "0001": [3, 4], "001": [0, 2], "005": 0, "01": [0, 2], "03724": 5, "1": [0, 2, 3, 4], "10": [0, 2, 3, 4], "100": [0, 2, 3, 4, 5], "1000": [], "1010": 4, "1020": 4, "12": [], "12th": 0, "1e": [], "2": [0, 2, 3, 4, 5], "200": 3, "2012": 0, "2017": 0, "2019": 2, "2022": [3, 4], "2024": 5, "2405": 5, "256": 4, "28th": [2, 4], "3": [0, 3, 4, 5], "30": [], "31": 0, "32": 3, "3f": [0, 3, 4, 5], "400": [], "5": [0, 4], "50": 3, "512": 4, "6": 0, "64": 4, "7": [], "784": 4, "9": [], "A": [0, 2, 4, 5], "For": 5, "If": 5, "It": [2, 3, 4, 5], "No": 0, "Or": 5, "That": 5, "The": [0, 2, 3, 4, 5], "aaai": 0, "acc": [0, 3, 4, 5], "accuraci": [0, 2, 4], "acm": [2, 3, 4], "activ": 3, "actual": [], "add": 5, "aditya": 0, "adj": [0, 2, 3, 4, 5], "adj_mat": [0, 5], "adj_matrix": 4, "adjac": [0, 2, 3, 4, 5], "adjust": [2, 3, 4], "after": [2, 3], "afterward": [], "al": [0, 2, 4], "algorithm": [0, 5], "all": 0, "alpha": [0, 2, 3, 5], "alpha1": 3, "alpha2": 3, "alpha3": 3, "alpha4": 3, "alpha5": 3, "alpha_list": 0, "although": [], "alumni": 5, "an": [3, 5], "ani": 5, "approach": 5, "approxim": 3, "ar": 0, "area": 0, "arg": [0, 2, 3, 4], "argument": 3, "arrai": 5, "articl": 5, "artifici": 0, "arxiv": 5, "atom": 5, "attr_mat": [], "attr_mat_norm": [], "attr_matrix": [], "attribut": [3, 4], "auc": [0, 2, 3, 4, 5], "augment": 2, "author": 5, "autoencod": 4, "awar": 3, "b": 0, "backward": [1, 3], "base": [0, 2, 3, 4, 5], "batch": [], "batch_siz": [], "belong": 5, "benchmark": 5, "best": [], "between": [0, 2, 3], "bia": 4, "binari": 0, "blob": [], "block": [], "bool": 4, "boundari": [], "brief": 5, "bug": 5, "c": 4, "calcul": 4, "call": [], "callabl": [], "can": 5, "candid": [0, 6], "care": [], "cascad": 0, "catch": 0, "check": 5, "chen": 0, "christo": 0, "cite": 5, "class": [0, 2, 3, 4, 5], "clone": 5, "code": 5, "coeff": [], "coeffici": [], "column": [0, 2, 3, 4], "com": 5, "combin": 4, "commit": 5, "comput": [2, 4], "compute_weight": [], "confer": [0, 2, 3, 4], "connect": [2, 4], "consid": 5, "consist": 5, "constraint": 3, "construct": [], "construct_attr_mat": [], "constructor": [], "contain": [0, 2, 3, 4, 5], "content": 6, "contribut": 5, "convolut": [2, 3, 4], "cora_ml": [0, 5], "correct": [0, 1], "cover": 0, "cpu": 4, "criteria": 0, "csr": 5, "csr_matrix": [0, 2, 3, 4], "cuda": 4, "culprit": 0, "curr_dir": [0, 3, 4, 5], "current": [], "curv": 0, "d": [3, 5], "data": [0, 2, 3, 4, 5], "data_dir": [0, 3, 4, 5], "data_nam": [0, 3, 4, 5], "dataload": [], "dataset": [0, 2, 3, 4, 5], "decai": [3, 4], "decod": [1, 4], "deep": [], "default": 0, "defin": [2, 3], "degre": 0, "describ": 5, "desir": 0, "detect": [0, 2], "determin": 5, "devic": 4, "dict": 0, "dictionari": [0, 5], "diff_mat": 0, "diff_typ": [0, 3, 4, 5], "diff_vec": 0, "differ": [], "diffus": [0, 2, 3, 4, 5], "diffusion_gener": [0, 3, 4, 5, 6], "diffusion_model": [1, 5], "diffusionpropag": [], "digg16000": [], "dimens": 4, "dimension": 0, "dirctori": 0, "direct": 5, "directori": 0, "discoveri": 4, "diverg": 4, "doc": [], "dolphin": [0, 5], "dong": 2, "download": [0, 5], "download_dataset": [0, 5, 6], "drop_prob": 4, "dropout": 4, "dure": 4, "each": [0, 3, 4], "earli": [], "early_stop": [], "earlystop": [], "edg": 2, "edge_index": 2, "edu": 5, "eight": [], "either": 0, "element": [], "els": [], "em": 0, "email": 5, "embed": [], "emori": 5, "encod": [1, 4], "entri": 0, "enum": [], "ep": [], "epidem": 0, "epoch": [2, 3, 4], "epoch_num": 3, "epsilon": [], "equat": [], "error": 3, "et": [0, 2, 4], "evalu": [0, 2, 4], "everi": [2, 3, 4], "exampl": [0, 3, 4], "exclud": [], "exclude_idx": [], "exclus": [], "f": [0, 3, 4, 5], "f1": [0, 2, 3, 4, 5], "faloutso": 0, "fals": [], "fea_constructor": [], "featur": [2, 3, 4], "feature_mat": [], "featurecon": [], "feedback": 5, "feel": 5, "fetch": [], "figur": 0, "file": [0, 5], "final": [], "final_pr": [], "first": [0, 2, 3, 4], "fit": [], "fix": 3, "flag": 4, "float": [0, 2, 3, 4], "folder": 5, "follow": 5, "format": 5, "former": [], "forward": [1, 2, 3, 4], "forward_loss": 4, "forwardmodel": [], "fraction": [0, 2], "free": 5, "from": [0, 2, 3, 4, 5], "function": 5, "g": 0, "g_bar": 0, "gamma": [], "garph": [], "gcn": [2, 4], "gcnconv": [1, 2], "gcnlayer": [1, 4], "gcnsi": [0, 1, 5], "gcnsi_model": [0, 1, 2, 5], "gcnsi_source_predict": [0, 5], "gen_se": [], "gen_splits_": [], "gener": [0, 5], "generate_seed_vector": [0, 6], "get": [0, 2], "get_dataload": [], "get_idx_new_se": [], "get_k_list": [0, 6], "get_predict": [], "getcwd": [0, 3, 4, 5], "getpredict": [], "github": 5, "given": [0, 3], "gnn": [0, 5, 6], "gnn_model": [], "gradient": 4, "graph": [0, 2, 3, 4, 5], "ground": 4, "have": 5, "hidden": 4, "hidden_dim": [3, 4], "hiddenunit": 4, "hook": [], "how": 0, "html": [], "http": 5, "i": [0, 2, 3, 4], "i_deepi": [], "i_gcn": [1, 3], "i_gcnlay": [1, 3], "ic": [0, 3, 4, 5], "ideal": [], "ident": [], "identif": [0, 2], "identifi": [0, 2], "idx": [], "idx_exclude_list": [], "idx_split_arg": [], "ieee": 0, "ignor": [], "implement": [0, 2, 3, 4], "import": [0, 3, 4, 5], "impos": 3, "in_channel": 2, "in_featur": [3, 4], "includ": [4, 5], "independ": 0, "index": 5, "indic": [2, 4], "infect": 0, "infect_prob": [0, 3, 4, 5], "infer": [1, 4, 5], "infer_loss": [1, 4], "influ_vector": 3, "influenc": 3, "inform": [0, 2], "initi": [3, 4], "input": [0, 2, 3, 4], "input_dim": 4, "instanc": [], "instead": [], "int": [0, 2, 3, 4], "intellig": 0, "intern": [0, 2], "interv": [], "invers": [3, 4], "invert": 3, "ipynb": 5, "issu": 5, "iter": 3, "its": [0, 2], "ivgd": [0, 1, 5], "ivgd_model": [1, 3, 5], "ivgd_source_predict": 5, "j": 4, "jazz": [0, 5], "jiang": [3, 4], "jill": 0, "jordan": 0, "journal": 5, "junji": 3, "junxiang": [3, 5], "jupyt": 5, "k": [0, 5], "k_list": 0, "kai": 0, "karat": [0, 3, 4, 5], "kei": 5, "kl": 4, "know": 0, "knowledg": [2, 4], "label": [0, 2, 3, 4], "labels_np": [], "lambda": 3, "lamda": 3, "lanczo": [], "lanczos_algo": [], "laplacian": 0, "latent": 4, "latent_dim": 4, "latter": [], "layer": [2, 3, 4], "learn": [2, 3, 4], "learning_r": [], "lei": 0, "liang": [3, 5], "librari": 5, "like": 5, "linear": [0, 4], "ling": 4, "list": [0, 4], "load": [0, 5], "load_dataset": [0, 3, 4, 5, 6], "loader": [], "local": [3, 4, 5], "locat": 0, "log": 4, "log_var": 4, "logvar": 4, "longtensor": [], "loss": [1, 2, 3, 4], "lpsi": [0, 2, 5, 6], "lr": [2, 3, 4], "lt": 0, "made": 4, "main": [0, 1, 5], "make": [], "manag": 2, "mani": 0, "matric": 0, "matrix": [0, 2, 3, 4, 5], "max_epoch": [], "maxim": 0, "me": 5, "mean": [0, 4], "meme8000": [], "messag": 5, "messagepass": 2, "met": [], "method": [4, 5], "metric": [0, 2, 3, 4, 5, 6], "mine": [0, 4], "ming": 2, "mlp": [], "mlptransform": [], "mode": 4, "model": [0, 1, 3], "modul": [5, 6], "multipl": [0, 2], "n_power_iter": [], "name": 0, "ndarrai": [0, 2, 3, 4], "ndim": [], "need": [], "neighbor": [0, 2], "net": 3, "net1": 3, "net2": 3, "net3": 3, "net4": 3, "net5": 3, "netscienc": [0, 5], "netsleuth": [0, 5, 6], "network": [0, 2, 3, 4], "networkx": 0, "neural": [3, 4], "new": 5, "niter": [], "nn": [3, 4], "node": [0, 2, 3, 4], "non": [], "none": [], "normal": [1, 3], "normalize_attribut": [], "notebook": 5, "now": 5, "np": [], "nstop": [], "ntrain": [], "num_class": 4, "num_epoch": [2, 3, 4], "num_lay": 3, "num_nod": [0, 3, 4], "num_sourc": 0, "num_thr": [0, 2, 3, 4], "number": [0, 2, 3, 4], "number_lay": 3, "numpi": [0, 2, 3, 4, 5], "nval": [], "o": [0, 3, 4, 5], "object": [0, 2, 3, 4, 5], "observ": 0, "obtain": 3, "ojc": [0, 5, 6], "one": [], "ones": 0, "open": 5, "opt_alpha": 0, "opt_auc": 0, "opt_f1": [0, 2, 3, 4], "opt_i": 0, "opt_k": 0, "opt_pr": [0, 2], "opt_thr": [0, 2, 3, 4], "optim": [0, 2, 3, 4], "option": [], "order": [], "org": [], "otherwis": [0, 5], "our": 5, "out_channel": 2, "out_dim": 4, "out_featur": [3, 4], "output": [3, 4], "output_dim": 4, "overridden": [], "packag": [5, 6], "page": 5, "paramet": [0, 3, 4], "partial": 0, "pass": [2, 3, 4], "patienc": [], "per": [], "perform": [2, 3], "phase": [], "pickl": [0, 5], "pip": 5, "piter": [], "pleas": 5, "point": 3, "potenti": 0, "power_grid": [0, 5], "pr": [0, 3, 4, 5], "prakash": 0, "precis": [0, 2, 4], "pred": [0, 3, 4, 5], "predict": [0, 2, 3, 4, 5, 6], "preprint": 5, "preprocess": [], "prescrib": [5, 6], "presrib": 5, "print": [0, 2, 3, 4, 5], "print_epoch": [2, 3, 4], "print_interv": [], "prob_matrix": [], "probabl": [0, 4], "problem": 4, "proceed": [0, 2, 3, 4], "prop_pr": [], "propag": 0, "pull": 5, "py": 5, "python": 5, "qualnam": [], "question": 5, "r": 5, "random": [0, 2, 3, 4], "random_epoch": 3, "random_se": [0, 2, 3, 4], "rank": [], "rate": [2, 3, 4], "ratio": 0, "re": [0, 3, 4, 5], "readi": 5, "recal": [0, 2, 4], "recip": [], "reconstruct": 4, "recov": 0, "recover_prob": 0, "recoveri": 0, "refer": [], "reg_param": 4, "regist": [], "regular": [], "rememb": [], "reparameter": [1, 4], "repeat_step": 0, "repetit": 0, "repo": 5, "report": 5, "repres": [0, 2, 3, 4], "represent": 4, "reproduc": 0, "request": 5, "requir": [4, 5], "research": 5, "residu": 3, "respect": [], "result": 5, "return": [0, 2, 3, 4, 5], "rho": 3, "rho1": 3, "rho2": 3, "rho3": 3, "rho4": 3, "rho5": 3, "rumor": 2, "run": [], "runtim": [], "same": [], "sampl": 4, "save": [0, 5], "save_dir": [0, 5], "save_nam": [0, 5], "scipi": [0, 2, 3, 4], "score": [0, 2, 3, 4], "search": 5, "second": [0, 2, 3, 4], "seed": [0, 2, 3, 4], "seed_hat": 4, "seed_idx": [], "seed_num": 0, "seed_ratio": [0, 3, 4, 5], "seed_vae_train": [4, 5], "seed_vec": [0, 4], "seed_vector": [0, 3], "select": [], "set": [0, 2, 3, 4, 5], "shape": [3, 4], "should": [0, 5], "si": 0, "sigkdd": 4, "silent": [], "sim_num": [0, 3, 4, 5], "simul": [0, 2, 3, 4], "sinc": [], "singl": 4, "sir": 0, "six": [], "size": [], "slave": [4, 5], "slvae": [0, 1, 5], "slvae_model": [1, 4, 5], "slvae_source_predict": 5, "sourc": [0, 2, 3, 4, 5], "space": 4, "spars": [0, 2, 3, 4], "specif": [], "specifi": [], "split": [0, 5], "split_dataset": [0, 3, 4, 5, 6], "spot": 0, "sprase": 5, "spread": [], "squar": [], "stand": 0, "start": 5, "step": 0, "stop": [], "stop_varnam": [], "stopping_arg": [], "stopping_idx": [], "stopping_s": [], "stopvari": [], "store": 0, "str": 0, "subclass": [], "subgraph": 0, "subject": 3, "submit": 5, "submodul": [1, 6], "subpackag": 6, "subset": [0, 2, 3, 4], "substitut": [], "sum": 4, "sure": [], "suscept": 0, "system": [], "take": [], "target": 0, "tau": 3, "tau1": 3, "tau2": 3, "tau3": 3, "tau4": 3, "tau5": 3, "temp": 3, "tensor": [0, 2, 3, 4], "term": 4, "test": [0, 1, 2, 3, 4, 5, 6], "test_dataset": [0, 2, 3, 4, 5], "them": [], "thi": 5, "thre": [0, 2, 3, 4, 5], "thres_list": [2, 3, 4], "threshold": [0, 2, 3, 4], "through": 3, "time": [0, 2, 3, 4], "time_step": 0, "titl": 5, "top": 0, "top_nod": 0, "torch": [0, 2, 3, 4], "torch_se": [], "total": 4, "total_loss": 4, "train": [0, 1, 2, 3, 4, 5, 6], "train_auc": [2, 3, 4], "train_dataset": [0, 2, 3, 4, 5], "train_diffus": [1, 3, 5], "train_f1": 0, "train_idx": [], "train_loss": [1, 4], "train_mod": 4, "train_model": [], "train_pr": 4, "train_ratio": 0, "train_siz": [], "tree": [], "trick": 4, "true": 4, "truth": 4, "try": [0, 2, 3, 4], "tupl": [], "tutori": 5, "txt": 5, "type": 0, "typic": 5, "under": [0, 5], "underli": 0, "unit": 4, "updat": [], "update_embed": [], "upload": 5, "url": 0, "us": [0, 2, 3, 4, 5], "util": [2, 3, 4, 5, 6], "vae": [1, 4], "val": [], "val_idx": [], "val_siz": [], "valid": 3, "validity_net": [0, 1], "valu": [0, 2, 3, 4, 5], "var": 4, "variabl": [], "varianc": 4, "variat": 4, "vector": [0, 2, 3, 4], "via": 4, "visual": [0, 5], "visualize_source_predict": [0, 5, 6], "vol": 0, "vreeken": 0, "wang": [0, 3, 4, 5], "wang2024joss": 5, "ware": 3, "we": 5, "web": 3, "weight": [3, 4], "weight_decai": [3, 4], "where": 0, "whether": [4, 5], "which": [0, 4], "while": [], "within": [], "without": 0, "work": 5, "x": [0, 2, 3, 4], "x_hat": 4, "xianggebenben": 5, "y": [0, 4, 5], "y_hat": 4, "y_list": 0, "y_true": 4, "year": 5, "ying": 0, "you": 5, "your": 5, "z": 4, "zhao": [3, 5], "zhen": 0, "zheng": 0, "zhu": 0, "\u03b3": [], "\u03bb": []}, "titles": ["GraphSL package", "GraphSL.GNN package", "GraphSL.GNN.GCNSI package", "GraphSL.GNN.IVGD package", "GraphSL.GNN.SLVAE package", "Welcome to GraphSL\u2019s documentation!", "GraphSL"], "titleterms": {"": 5, "contact": 5, "content": [0, 1, 2, 3, 4, 5], "correct": 3, "diffusion_model": 3, "document": 5, "earlystop": [], "evalu": [], "gcnsi": 2, "gnn": [1, 2, 3, 4], "graphsl": [0, 1, 2, 3, 4, 5, 6], "guid": 5, "i_deepi": [], "indic": 5, "instal": 5, "ivgd": 3, "main": [2, 3, 4], "mlp": [], "model": [2, 4], "modul": [0, 1, 2, 3, 4], "packag": [0, 1, 2, 3, 4], "preprocess": [], "prescrib": 0, "quickstart": 5, "slvae": 4, "submodul": [0, 2, 3, 4], "subpackag": [0, 1], "tabl": 5, "train": [], "usag": 5, "util": 0, "validity_net": 3, "welcom": 5}})
GraphSL.egg-info/PKG-INFO:Version 0.14 uses the num_thres (i.e. number of thresholds to try) instead of specifying the thres_list (i.e. threshold list) for LPSI, GCNSI, IVGD and SLVAE. Moreover, GCNSI, IVGD and SLVAE are improved to run on CUDA if applicable.
GraphSL.egg-info/PKG-INFO:Version 0.15 makes all methods run on CUDA if applicable, replaces the diffusion model of IVGD and the encoder of SLVAE, and revises the generation of diffusion. 
README.md:Version 0.14 uses the num_thres (i.e. number of thresholds to try) instead of specifying the thres_list (i.e. threshold list) for LPSI, GCNSI, IVGD and SLVAE. Moreover, GCNSI, IVGD and SLVAE are improved to run on CUDA if applicable.
README.md:Version 0.15 makes all methods run on CUDA if applicable, replaces the diffusion model of IVGD and the encoder of SLVAE, and revises the generation of diffusion. 
requirements.txt:nvidia-cublas-cu12==12.1.3.1
requirements.txt:nvidia-cuda-cupti-cu12==12.1.105
requirements.txt:nvidia-cuda-nvrtc-cu12==12.1.105
requirements.txt:nvidia-cuda-runtime-cu12==12.1.105
requirements.txt:nvidia-cudnn-cu12==8.9.2.26
requirements.txt:nvidia-cufft-cu12==11.0.2.54
requirements.txt:nvidia-curand-cu12==10.3.2.106
requirements.txt:nvidia-cusolver-cu12==11.4.5.107
requirements.txt:nvidia-cusparse-cu12==12.1.0.106
requirements.txt:nvidia-nccl-cu12==2.19.3
requirements.txt:nvidia-nvjitlink-cu12==12.4.99
requirements.txt:nvidia-nvtx-cu12==12.1.105
build/lib/GraphSL/GNN/SLVAE/model.py:            - device (torch.device): Device to be used for computation, cpu or cuda.
build/lib/GraphSL/GNN/SLVAE/model.py:        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
build/lib/GraphSL/GNN/SLVAE/model.py:        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
build/lib/GraphSL/GNN/SLVAE/main.py:        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
build/lib/GraphSL/GNN/SLVAE/main.py:        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
build/lib/GraphSL/GNN/SLVAE/main.py:        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
build/lib/GraphSL/GNN/IVGD/model/MLP.py:            device='cuda'):
build/lib/GraphSL/GNN/IVGD/diffusion_model.py:        # Determine whether to use GPU or CPU
build/lib/GraphSL/GNN/IVGD/diffusion_model.py:        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
build/lib/GraphSL/GNN/IVGD/diffusion_model.py:        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
build/lib/GraphSL/GNN/IVGD/training.py:                device: str = 'cuda',
build/lib/GraphSL/GNN/IVGD/training.py:    - device (str): Device for training, cpu or cuda.
build/lib/GraphSL/GNN/IVGD/main.py:        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
build/lib/GraphSL/GNN/IVGD/main.py:        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
build/lib/GraphSL/GNN/GCNSI/main.py:        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
build/lib/GraphSL/Prescribed.py:        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
build/lib/GraphSL/Prescribed.py:        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
build/lib/GraphSL/Prescribed.py:        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

```