
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

**1.Image and annotation file pair**<br>
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
**2.Cropped image of class folder structure**<br>
:This method creates a folder for each class and includes the cropped image of the corresponding class in it.

```
Folder hierarchy
data/
├── Copper wire tightening/
│ ├── img01.jpeg
│ ├── img02.jpeg
│ └── ...
├── Ground wire assembled on screw/
│ ├── img01.jpeg
│ ├── img02.jpeg
│ └── ...
└── ...
```

In this structure, the folder name is used as the class name.<br>

If you have both the original and cropped images<br>
If you provide both the original image and the cropped images from it, use the following naming convention:

```
data/
├── Copper wire tightening/
│ ├── origin01.jpeg # Original image
│ ├── origin01_crop01.jpeg # Cropped image from the original
│ ├── origin01_crop02.jpeg
│ ├── origin02.jpeg
│ ├── origin02_crop01.jpeg
│ └── ...
└── ...
```
<br>
4. Local Demo start

```bash
alo run
```
