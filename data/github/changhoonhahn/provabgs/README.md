# https://github.com/changhoonhahn/provabgs

```console
docs/paper/psmf/psmf.bib:  author = {Hahn, ChangHoon and Wilson, Michael J. and {Ruiz-Macias}, Omar and Cole, Shaun and Weinberg, David H. and Moustakas, John and Kremin, Anthony and Tinker, Jeremy L. and Smith, Alex and Wechsler, Risa H. and Ahlen, Steven and Alam, Shadab and Bailey, Stephen and Brooks, David and Cooper, Andrew P. and Davis, Tamara M. and Dawson, Kyle and Dey, Arjun and Dey, Biprateep and Eftekharzadeh, Sarah and Eisenstein, Daniel J. and Fanning, Kevin and {Forero-Romero}, Jaime E. and Frenk, Carlos S. and Gazta{\~n}aga, Enrique and Gontcho, Satya Gontcho A and Guy, Julien and Honscheid, Klaus and Ishak, Mustapha and Juneau, St{\'e}phanie and Kehoe, Robert and Kisner, Theodore and Lan, Ting-Wen and Landriau, Martin and Le Guillou, Laurent and Levi, Michael E. and Magneville, Christophe and Martini, Paul and Meisner, Aaron and Myers, Adam D. and Nie, Jundan and Norberg, Peder and {Palanque-Delabrouille}, Nathalie and Percival, Will J. and Poppett, Claire and Prada, Francisco and Raichoor, Anand and Ross, Ashley J. and Safonova, Sasha and Saulder, Christoph and Schlafly, Eddie and Schlegel, David and {Sierra-Porta}, David and Tarle, Gregory and Weaver, Benjamin A. and Y{\`e}che, Christophe and Zarrouk, Pauline and Zhou, Rongpu and Zhou, Zhimin and Zou, Hu},
bin/tiger/deploy_decoder_train.py:        '#SBATCH --gres=gpu:1', 
bin/tiger/deploy_emu_train.py:        '#SBATCH --gres=gpu:1', 
bin/tiger/deploy_emu_train.py:        "conda activate tf2-gpu", 
src/provabgs/models.py:        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

```
