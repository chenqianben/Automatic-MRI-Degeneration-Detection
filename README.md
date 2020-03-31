# Title
Development of Automated-analysis Tool for Quantitative MRI Images in the Assessment of Lumbar Disc Degeneration in Sheep

# Context
Discogenic low back pain, named discogenic lombalgia, is a major public health concern that is
frequently associated with lumbar intervertebral disc degenerative disease (DDD). Numerous tools
and animal models have been used to help us improve our understanding of the DDD physiopathology
and develop imaging methods to detect it as early as possible. The sheep has been shown to be a
valuable large animal model thanks to the similarities (gross anatomy, mechanical properties) of its
lumbar intervertebral disc (IVD) with those of human lumbar IVD. In parallel, quantitative magnetic
resonance imaging (MRI) seems to be a clinically-relevant tool to explore early DDD, in particular
T2* relaxation time measurements. During preliminary study, we compared three MRI methods of
quantitative time measurements evaluation of lumbar ovine IVD (T1, T2 and T2*) using different
manually drawn region-of-interest (ROI). Interestingly, while T2 and T2* mapping are well described
to characterize the DDD in various species, our preliminary data strongly suggest that T1 mapping
using the variable flip-angle could be a promising tool to specifically assess the early events of
DDD in an ovine model and maybe in human patient.

![image1](https://github.com/chenqianben/Project-MRI-Segmentation/raw/master/images/src_tgt_images.PNG)

# Objectives
Regarding the time-consuming analysis of data and opportunity offered with the deep learning
approach, the objective of this project is to develop an automated-analysis program of MRI
images based on sheep image database acquired during manually-preliminary study. 

Given a complete source image (T1 SAG or T2 SAG) that is technically clear without much noise, the program
aims to locate the three ROIs and make the registration to find the corresponding ROIs in the target image(T1, T2 or T2*) where there is much noise but the values can refer to the spinal degenerative level.

![image2](https://github.com/chenqianben/Project-MRI-Segmentation/raw/master/images/three_ROIs.PNG)

# Kerwords
Intervertebral disc, disc degeneration, MRI sequences, MRI mapping, automated-analysis, image
segmentation, machine learning.

# Syllabus
- [__Part1__] __Detection__
  - Input: Complete source images (T1 SAG, T2 SAG)
  - Output: Patchs where the ROIs are located
  - Tool: CNN for traversal

- [__Part2__] __Segmentation__
  - Input: Patchs where the ROIs are located
  - Output: The three ROIs
  - Tool: U-net for segmentation
  
- [__Part2__] __Registration__
  - Input: Source image ROIs location 
  - Output: Target image ROIs location
  - Tool: Spatial Tranformation Networks/ Manual ROI location(semi-automatic)

# Notes
- The train process and the pipiline are in Juppyter Notebook for educative demonstration. 
- The **[PIPILINE.ipynb](https://github.com/chenqianben/Project-MRI-Segmentation/blob/master/PIPELINE.ipynb)** collects all the parts.
