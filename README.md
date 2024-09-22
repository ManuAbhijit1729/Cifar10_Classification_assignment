Change the **root** and **base_path** in train.py 
Finally RUN command **python train.py** in a python compatible terminal

Additionally i have two jupyter notebooks **runs_notebook.ipynb** where i have run two separate instances with two separate models (Pretrained Resnet50 from timm (hugging face) and my custom ConvNet model writtem from scratch in pytorch)
And **inference.ipynb** where i have inferred on the two best models from each run

Evaluation Scores on Test Set (Cifar-10):

**For Resnet50 model :**

**testing accuracy: 0.85800**,
**testing precision: 0.85776**,
**testing recall: 0.85800**,
**testing F1-Score: 0.85743**

**Confusion Matrix :**
![image](https://github.com/user-attachments/assets/b2ed120b-2071-4d34-ae04-6f6400b6c542)

**For Custom ConvNet model :** 

**testing accuracy: 0.76600**,
**testing precision: 0.76442**,
**testing recall: 0.76600**,
**testing F1-Score: 0.76498**

**Confusion Matrix :**
![image](https://github.com/user-attachments/assets/cc550948-ba8a-48c2-a473-a55b100fbe0d)


