# Happiness-Detector

Sup ! Back again with another computer vision challenge. Let's detect if you are happy or not by leveraging OpenCV powerful routines, including the Viola-Jones algorithm.

## Background

Object recognition or various shapes, nature and colour is in high demand in the industry, from automatic tagging whenever you post a picture on Facebook, security software for CCTV devices or to unlock your new shining iPhone X.

The approaches used to detect objects in the surroundings have mutated a lot in the past 15 years. The first proper algorithm to work efficiently for real world tasks was the method proposed by **Paul Viola** and **Michael Jones** and their **Haar Feature**,  **Adaboosted** and  **Cascaded** approach in 2001. Nowadays, thanks to more powerful GPUs and a data-driven methodology we are able to detect multiple objects in a scene astonishingly fast by using Machine Learning techniques.

## Task & Result

The rather simple task in this occasion was to use a pre-trained cascade classifier to detect smiles in a face. The result does not meet a lot of the criteria for a day-to-day use given the inaccuracies of the approach when the face is not in a frontal position.

The following pictures show the result before the tuning the parameters of the **detectMultiScale** routine and after.

![alt text](https://github.com/itaouil/Happiness-Detector/blob/master/weak_parameters.png)

![alt text](https://github.com/itaouil/Happiness-Detector/blob/master/strong_parameters.png)
