# Modular Counterfactuals 

This repository contains the preliminary code of the modular counterfactual explanation framework *MCE*, supervised by Nadia el Bekri at Fraunhofer IOSB, Karlsruhe.


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
from inside the source code directory you can then try a test run using the `experiments.py`
```bash
python experiments.py --ex ids
```

using the `--ex ids/kdd` flag, you can choose between running the experimental evaluation on KDD or IDS datasets, if provided correctly. 

# Experiment Environment - Sacred
To keep track of our experiments, we use `sacred`. How to use sacred for other experiments as well can be seen in the `experiments.py`. 
We use `sacred` with the [MongoObserver](https://sacred.readthedocs.io/en/latest/observers.html#mongo-observer) and display results using [Omniboard](https://vivekratnavel.github.io/omniboard/#/quick-start).

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
