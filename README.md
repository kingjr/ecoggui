Ecog GUI
========

Ecog GUI is a Python module to localize intracranial electrodes from MRI.

It interactively fits a priori defined grids, strips or rigid array of
electrodes to the MRI with a Graphical User Interface.

Its main advantages as compared to current alternatives are:
* it provides the best fits in real time, and can thus help you adjust your
locations interactively.
* it can fit curved surfaces, including non-convex ones, but remains realistic
thanks to a rigidity constrain.


This is currently an alpha version.

![demo](docs/ecoggui_animation.gif)

Online Tutorials
================

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/kingjr/ecoggui) to run the online examples in your browser. Then go to notebooks and select the tutorials:

* `examples/model_displacement.pynb` how the 2D grid/strips are fitted with a
rotation and translation to match the 3D locations.
* `examples/model_surface.pynb`: how the 2D grid/strips are fitted with a
2nd order polynomial with a constrain on the electrodes local distances, as we
know the grid can be flexible but is not elastic.
* `examples/gui_mri_ecog.pynb`: try to fit the grid this patient brain. It should take no more than five minutes.

Installation
============

Clone this repository and install using setup.py:

```python setup.py develop --user```


Dependencies
============

The required dependencies to use the software are:

* Python >= 2.7
* setuptools
* Numpy >= 1.6.1
* SciPy >= 0.9
* Scikit-learn >= 0.17
* Nibabel >= 1.1.0
* Matplotlib >= 1.5.1
