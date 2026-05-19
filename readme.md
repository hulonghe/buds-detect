\# Tea Bud Image Datasets



This repository provides three datasets — \*\*A\*\*, \*\*B\*\*, and \*\*C\*\* — designed for research on tea bud detection,

classification.  

Each dataset contains high-resolution images of tea buds captured under different lighting and environmental conditions,

suitable for training and evaluating computer vision models.



\## Dataset Overview



\- \*\*Dataset A\*\*: The most complex, consisting of raw images directly captured from real-world environments without special processing, containing a large number of small objects.

\- \*\*Dataset B\*\*: Moderately difficult, with most objects being of medium size and some small objects present.

\- \*\*Dataset C\*\*: The simplest, primarily containing medium to large objects, which are relatively easy to recognize.



\## Access



You can download the datasets from the following link:



🔗 \*\*Download Link1（A，B，C）:\*\* \[Baidu Cloud](https://pan.baidu.com/s/1L-2OfyIftN9h3FcrYUcePw?pwd=n5w3)  

🔑 \*\*Extraction Code:\*\* `n5w3`



🔗 \*\*Download Link2（B）:\*\* \[Roboflow](https://universe.roboflow.com/ycy-9asp0/tearob)



🔗 \*\*Download Link3（C）:\*\* \[Roboflow](https://universe.roboflow.com/project-n7lcw/object-detection-for-tea-bud)



\# Environment and Usage Instructions



This section describes the recommended system environment, software dependencies, and usage workflow for training and

evaluating tea bud recognition models.



\## 1. System Requirements



\- \*\*Operating System:\*\* Windows 10/11, Ubuntu 20.04+

\- \*\*Hardware Configuration:\*\*

&#x20;   - GPU: NVIDIA RTX 5070 or higher (≥12 GB VRAM recommended)

&#x20;   - CPU: Intel i7 / AMD Ryzen 7 or better

&#x20;   - RAM: ≥32 GB

&#x20;   - Storage: ≥100 GB of free disk space



\## 2. Software Environment



\- \*\*Programming Language:\*\* Python 3.10

\- \*\*Deep Learning Framework:\*\* PyTorch = 2.9.0

\- \*\*Essential Libraries:\*\*

&#x20;   - OpenCV = 4.12.0.88

&#x20;   - albumentations ≥ 2.0.8

&#x20;   - NumPy

&#x20;   - Pandas

&#x20;   - Matplotlib

\- \*\*Annotation Tools (optional):\*\* LabelImg, X-AnyLabeling



\## 3. Installation and Setup



```bash

\# Clone the project repository

git clone https://github.com/hulonghe/buds-detect.git

cd buds-detect



\# Create and activate a virtual environment

python -m venv venv

source venv/bin/activate   # (Linux/macOS)

venv\\Scripts\\activate      # (Windows)



\# Install dependencies

pip install -r requirements.txt

```



\## 4. Training and Evaluation Workflow



Download and extract the tea bud datasets (A, B, C). Start training:



```bash

python train\_reg.py

```



Evaluate the trained model:



```bash

python validate\_reg.py

```



Training logs, metrics (accuracy, precision, recall, mAP), and model checkpoints will be saved in the ./runs directory.



\## 5. Recommendations



Adjust batch size and learning rate according to GPU memory limits.

Use mixed precision training (torch.amp) to speed up convergence.



\# Usage



Please cite this dataset appropriately if you use it in your research or publications.  

For academic and non-commercial use only.



\# Acknowledgment and Contact



Thank you for using this dataset and environment guide.  

If you have any questions, suggestions, or collaboration requests, please feel free to contact us:



\- \*\*Author:\*\* TPFD-Net: Transformer-based Path Fusion Fetector Network for teabud detection

\- \*\*Email:\*\* \[hlhcmp@gmail.com]



We appreciate your feedback and contribution to improving tea bud recognition research.

