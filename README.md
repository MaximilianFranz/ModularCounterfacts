# GradientGrow - Adaption and Extension 

This repository contains adaptions of the original *GradientGrow* source code supervised by Nadia el Bekri at Fraunhofer IOSB, Karlsruhe. 

The `evaluation.py` is the executable file which implements a comparison of different methods to explain instances of a classfication performed with a RandomForestClassifier. The compared methods are: 
 - LIME 
 - LocalSurrogate
 - GradientGrow 



# To Try yourself
Clone the repository into some directory, create a virtual environment with 
```bash
virtualenv pyvenv
```

activate the environment with 

```bash 
source pyvenv/bin/activate
```
and install the required packages with `pip`: 
```bash
pip install -r requirements.txt
```
from inside the source code directory you can then try a test run on the UCI Credit Dataset with: 
```bash
python evaluation.py
```
