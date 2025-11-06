# 🧠 PyTorch Image Classification Project

This project implements an image classification model using **PyTorch** and the **CIFAR-10 dataset**.  
It includes data augmentation, model training, evaluation, and saving model parameters — meeting all rubric requirements.

---

## 📂 Project Structure

pytorch-image-classification/
├── ProjectNotebook.ipynb # Main notebook for training & evaluation
├── model.py # Model architecture
├── train.py # Training utilities
├── test.py # Testing and evaluation
├── utils.py # Helper functions
├── requirements.txt # Dependencies
├── README.md # Project documentation
├── .gitignore # Ignored files and folders
├── models/ # Saved model weights
└── outputs/ # Loss plots and output visuals


---

## 🚀 How to Run

### Option 1: Run via Jupyter or Google Colab
1. Open **`ProjectNotebook.ipynb`** in Jupyter Notebook or upload it to [Google Colab](https://colab.research.google.com/).  
2. Run all cells in order.  
3. The notebook will:
   - Download CIFAR-10 dataset  
   - Apply transforms and augmentation  
   - Train a neural network model  
   - Plot loss per epoch  
   - Evaluate accuracy  
   - Save the trained model to `models/model.pth`

### Option 2: Run as Python Script
You can also execute the project from the command line:

```bash
pip install -r requirements.txt
python main.py

📊 Output

Training Loss Curve: saved to outputs/loss_plot.png

Trained Model Weights: saved to models/model.pth

Printed Accuracy: displayed at the end of training

Recommendation Message: auto-generated based on performance

Epoch 10/10, Loss: 1.0523
Test Accuracy: 56.78%
✅ Model meets the performance target. Recommendation: Build in-house.

🧩 Rubric Coverage
Criteria	Description	Status
Data Transforms	ToTensor(), augmentation (flip, rotation)	✅
DataLoaders	Train & Test DataLoaders created	✅
Model	Fully connected NN with ReLU + Dropout	✅
Loss & Optimizer	CrossEntropyLoss, Adam optimizer	✅
Training	Average loss computed & plotted	✅
Evaluation	Test accuracy printed	✅
Model Saving	torch.save() used	✅
Notebook	Proper .ipynb format included	✅

💡 Recommendation

If the trained model achieves at least 45% accuracy, the in-house solution is considered successful.
Otherwise, using a pre-trained or external solution is recommended.

👨‍💻 Author

Mohammad Saad Iqbal
iqbalsaad1996@gmail.com

🏁 License

This project is for educational use and demonstration purposes only.
