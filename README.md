
## :hammer_and_wrench:  Requirements and Install 

Basic Dependencies:

* Python == 3.10
* cuda 12.1
<br>

**Installation:**

<br>
1.Install ALO V3 Framework k in your python 3.10 environment.

```bash
pip install mellerikat-alo
```

<br>
2. Clone the oneshot-objectdetection solution repository from GitHub.

```bash
git clone https://github.com/mellerihub/oneshot-objectdetection.git
```

<br>
3. Configure the dataset for fewshot/oneshot embedding into your experiment path.<br>

### directory structure <br>

#### **1.Image and annotation file pair**<br>
:This method uses an annotation file that contains the original image and its bounding box information.<br>

```
Folder hierarchy
data/
├── img_01.jpg
├── img_01_label.txt
├── img_02.jpg
├── img_02_label.txt
└── ...
```
The annotation file (ex: `img_01_label.txt`) follows the following format:
```
Annotation file format
class 0, x, y, w, h
class 1, x, y, w, h
...
```


<br>
4. Local Demo start

```bash
alo run
```
