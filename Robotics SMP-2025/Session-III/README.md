## FashionMNISTVGGNet11

This repository contains a Jupyter Notebook (`FashionMNISTVGGNet11.ipynb`) that explores the Fashion MNIST dataset and likely implements a VGGNet architecture for classification.

### 1. About the Fashion MNIST Dataset

The Fashion MNIST dataset is a large-scale, publicly available dataset designed as a direct drop-in replacement for the original MNIST dataset, primarily for benchmarking machine learning algorithms. Unlike MNIST, which features images of handwritten digits, Fashion MNIST comprises grayscale images of various clothing items.

#### Key Facts:
* **Dataset Composition**:
    * **Number of Samples**: 70,000 images.
    * **Training Set**: 60,000 images.
    * **Test Set**: 10,000 images.
    * **Image Dimensions**: Each image is 28x28 pixels.
    * **Color Channels**: Grayscale (single channel).
* **Classes**: The dataset includes 10 classes of clothing items:
    1.  T-shirt/top
    2.  Trouser
    3.  Pullover
    4.  Dress
    5.  Coat
    6.  Sandal
    7.  Shirt
    8.  Sneaker
    9.  Bag
    10. Ankle boot
* **Purpose**: Created by Zalando Research and released in 2017, Fashion MNIST aims to provide a more complex and challenging dataset for benchmarking machine learning algorithms, encouraging models that generalize better to real-world data.
* **Format**: The dataset is provided in a similar format to MNIST, ensuring ease of use with existing machine learning frameworks.
* **Usage**: It can be utilized for various machine learning tasks, including classification, clustering, and generative modeling, and is widely adopted in educational settings.

#### Relevant Sources and Links:
* **Original Paper and Dataset**:
    * [Fashion-MNIST GitHub Repository](https://github.com/zalandoresearch/fashion-mnist)
    * [Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms](https://arxiv.org/abs/1708.07747)
* **Dataset Hosting**:
    * [Fashion MNIST on Kaggle](https://www.kaggle.com/zalando-research/fashionmnist)
* **Further Reading**:
    * [Deep Learning with Python by François Chollet](https://www.manning.com/books/deep-learning-with-python-second-edition)
    * [Machine Learning Mastery](https://machinelearningmastery.com/)

### 2. Data Augmentation/Cleaning and Image Folder

The notebook demonstrates how to create training and validation datasets using `ImageFolder` from `torchvision`. Key transformations and data handling strategies include:
* **Using Test Set for Validation**: The test set is directly used as the validation set, providing more data for training.
* **Channel-wise Data Normalization**: Image tensors are normalized by subtracting the mean and dividing by the standard deviation across each channel. This ensures data mean is 0 and standard deviation is 1, preventing disproportionate effects from channels with wider value ranges.
* **Randomized Data Augmentations**: Random transformations are applied during image loading for the training dataset. This includes padding by 4 pixels, random crops of 32x32 pixels, and horizontal flipping with a 50% probability. These augmentations help the model generalize better by exposing it to slightly varied images in each epoch.

The `data_transforms` dictionary defines the transformations applied to the training and test datasets, converting images to tensors and grayscale:

```python
from torchvision.transforms import v2
data_transforms = {
    'train': v2.Compose([
        v2.ToTensor(),
        v2.Grayscale(),
        ]),

    'test': v2.Compose([
        v2.ToTensor(),
        v2.Grayscale(),
        ]),
}
```
The dataset is loaded using `torchvision.datasets.FashionMNIST`, with options to download if not present and apply the defined transforms.

### 3. Technologies Used

The notebook utilizes the following Python libraries:
* `matplotlib.pyplot`
* `torch`
* `torch.optim`
* `torchvision`
* `torchvision.transforms.v2`
* `torch.utils.data.random_split`

The code also checks for CUDA availability to leverage GPU for computations:
```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
```

### 4. Setup and Usage

To run this Jupyter notebook:
1.  **Clone the repository** 
2.  **Install necessary libraries**:
    ```bash
    pip install matplotlib torch torchvision
    pip install -r requirements.txt
    ```
4.  **Run cells sequentially** to download the dataset, apply transformations, and execute the model training and evaluation steps.