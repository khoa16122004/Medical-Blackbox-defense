# Motivation
Brain tumors are aggressive diseases with severe impacts. In 2021, over 84,000 people were diagnosed, and 18,600 died from brain cancer. MRI is crucial for detection. Early diagnosis and treatment improve outcomes. 

This repository uses a Convolutional Neural Network (CNN) for brain tumor classification.

# Data Preprocessing
This idea is based on [this research paper](https://github.com/SartajBhuvaji/Brain-Tumor-Classification-Using-Deep-Learning-Algorithms/blob/master/Research%20Paper.pdf).

MRIs often include a background around the brain image, which is irrelevant for classification and inefficient for neural networks. We apply computer vision techniques to detect and crop the brain area:

- Identify and mark the largest contour in the image.
- Determine extreme points of this contour.
- Crop the image using these points to remove unnecessary background and noise.
- All images are resized to 224x224x3 to preserve relevant information while reducing file size.

<p align="center">
    <img src=croping.png width=80%>
</p>

# Data Augmentation
Due to limited data, we augment our dataset approximately tenfold using transformations such as:

- Random horizontal flip with a probability of 0.5.
- Random vertical flip with a probability of 0.5.
- Random rotation within ±45 degrees.
- Random adjustments to brightness, contrast, saturation, and hue.


<p align="center">
    <img src=argument.png width=60%>
</p>

# Experimental Results

I apologize for the confusion. Markdown itself doesn't support centering of tables directly. However, you can achieve centering in Markdown-rendered documents by embedding the Markdown table within HTML <div> tags styled with CSS for centering. Here's how you can do it:

<div align="center">

| Model                   | Accuracy (%) |
|-------------------------|--------------|
| VGG19 (no augmentation) | 0.76         |
| VGG19 (augmentation)    | 0.78         |
| Resnet50 (augmentation) | 0.81         |
| Mobilenet               |              |
| Unet                    |              |

</div>
