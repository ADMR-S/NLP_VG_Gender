JUPITER NOTEBOOK FOR ACTION/CHARACTER AND MALE/FEMALE DIALOGUE CLASSIFICATION

Change these variables to modify the datasets used : 

  usedPaths = [hZDPath] #Paths to the datasets used
  gameID = 'Horizon_Zero_Dawn' #To save trained models and other outputs with the game(s) name
  wholeDataset = True #Whether to use the whole dataset or 20% of it to reduce computation time

Change this one if you change the classifier : 
  ClassifierID = 'MLPClassifier' #To save trained models with the classifier name

--------------------------------------------------------
Build dataset from Video Game Dialogue Corpus : 

create virtual env :
python3 -m venv nlp_env

activate it : 
source ./nlp_env/bin/activate

install python packages :
py -m pip install bs4
py -m pip install html5lib
py -m pip install cssutils
py -m pip install lxml
py -m pip install numpy
py -m pip install xlrd==1.2.0
py -m pip install hjson
py -m pip install xlsxwriter
py -m pip install pyyaml
py -m pip install textatistic
py -m pip install igraph

pip install bs4
pip install html5lib
pip install cssutils
pip install lxml
pip install numpy
pip install xlrd==1.2.0
pip install hjson
pip install xlsxwriter
pip install pyyaml
pip install textatistic
pip install igraph