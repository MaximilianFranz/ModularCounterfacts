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

# Documentation
We're still working on the documentation of the source code. To do this consistently we adhere to the Google Style Python Docstrings like explained [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

# Code Refactoring
Along with adding consistent docs we ought to improve code formatting and the
general architecture to be more API-like, modular and accessible.
A great example for a good Python DataScience project in this regard is [LIME](https://github.com/marcotcr/lime), which is largely object-oriented.

Adhere to the [PEP8 Style Guide](https://www.python.org/dev/peps/pep-0008/) as much as possible.

Generally:
 - Look out for consistent variable naming and expressive names
 - remove legacy code and commented-out code
 - Make stuff simpler
 - Write explanatory comments inside the code to explain *magic* steps.
