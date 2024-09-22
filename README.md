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

**testing accuracy: 0.81550**,
**testing precision: 0.81572**,
**testing recall: 0.81550**,
**testing F1-Score: 0.81538**

**Confusion Matrix :**
![image](https://github.com/user-attachments/assets/cf4ed583-3825-48de-a5a1-1fde32e9fbe8)



