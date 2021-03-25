## Dataset
There are two newly collected datasets in our paper, i.e., RON and DMON. 

- We provide the labels in this project. 

- The original dataset can be downloaded from ([RON](https://pan.baidu.com/s/1K8G1Lz6cwMMTyFCcllEs2w) (Extraction code: ```ggvk```) and [DMON](https://pan.baidu.com/s/1zBYExyy0gVuIC2D0i-axkw) (Extraction code: ```5xge```)). 


### Note
- As stated in our paper, the DMON dataset is collected from the Daily Mail website. You can expand the dataset according to the given [urls](https://github.com/Gyaya/LD-MAN/blob/main/dataset/DMON/DM.csv).
- Labels for the RON dataset are crawled from the Rappler website. The labels may change as uses click. You can update the emotion labels according to the urls given in the dataset.
- Since we filter out the news without images,  the number of news articles listed in the "labels" files may be inconsistent with the original dataset. The 2nd sheet of "labels" file for the DMON dataset contains the complete annotation. And the complete labels for the RON dataset can be found in "text.txt" of each news article.

### TODO
- [x] Original dataset.
- [ ] Preprocess code.
- [ ] Dataset crawler. 
