# AIFeel

AIFeel is a simple positive/negative sentiment classifier for Python.

No other documentation for now. Just install and use it.

## Installation

From source:

```sh
$ pip install -e . # for minimal dependencies
$ pip install -e .[full] # for full dependencies
```

WARNING: Installing the `requirements.txt` directly is not recommended, and will install the same requirements as `.[full]`.
Use the above method to install instead, as the requirements file is only meant to be for streamlit.

## Usage

```sh
$ python -m aifeel
```

Importing as a library is also possible (but not supported yet).

```py
import aifeel
```
