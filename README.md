├── DataSet/  
│   ├── real_vs_fake/  
│   │   ├── real_vs_fake/  
│   │   │   └── test/  
│   │   │       ├── real/  
│   │   │       └── fake/  
│   │   │   └── train/  
│   │   │       ├──real/  
│   │   │       └── fake/  
│   │   │   └── valid/  
│   │   │       ├── real/  
│   │   │       └── fake/  
│   └── History/  
├── MesoNet/  
│   ├── Modeling.ipnb  
│   ├── Modeling.py  
│   ├── Training.ipnb  
│   ├── Training.py  
│   └──  config.json  
├── Models/  

DataSet : Collected from [Keggale](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)  
History : performance of each model at every epoch includes training loss, training accuracy, validation loss and validation accuracy  
MesoNet : Files to train MesoNet models, differ by filter size and/or input sizes  
Modeling.ipnb : Visualize and costumise model architectures  
Modeling.py : After defining model at Modeling.ipnb file copy paste here to use in Traing.ipnb  
Traing.ipnb : Step by step to train models imported forom Modeling.py in browser  
traing.py : Same as Traing.ipnb but in command prompt  
config.json : data of all models current traing status  



