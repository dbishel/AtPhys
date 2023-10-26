# AtPhys

## Description
This project is a home for various atomic physics related scripts.

Included are 
- Screened Hydrogenic model
- Saha-Boltzmann model
- FAC (Flexible Atomic Code) helper functions
- FAC examples


## Installation
The set of required packages is fairly minimal:
- numpy
- matplotlib
- pandas
- scipy
- jupyter (to run example notebook)

A working environment is included in the examples folder. To install this environment, install Anaconda and run from the terminal

`conda create -n ENVIRONMENT_NAME -f ENVIRONMENT.yml`

Replace ENVIRONMENT_NAME with what you want the environment to be named, and ENVIRONMENT.yml with the full name of the .yml file. Switch to the new environment with

`conda activate ENVIRONMENT_NAME`

## Citing this project
For any work that uses or is based on this project, please cite the associated DOI with the project (if on GitHub) and cite the article D. T. Bishel _et al._, "Toward constraint of ionization-potential depression models in a convergent geometry", HEDP (2023). See https://www.sciencedirect.com/journal/high-energy-density-physics for the full citation.

## Examples
Multiple examples are included at the bottom of the main scripts `ScHyd.py` and `ScHyd_AtomicData.py`. These can be run by either copying to a new scritp, or by setting `if 0:` to `if 1:` at the top of whichever example you want to run and running the file.

For more detail on the equations and algorithms used, see ScreenedHydrogenic_description.pdf.

## Support
For help using the code, email dbishel@ur.rochester.edu.

## Authors and acknowledgment
David Bishel (University of Rochester)

## License
For open source projects, say how it is licensed.

## Project status
This project is being developed on an as-needed basis in a private Gitlab project. It has been ported to a public GitHub project and released to enable attachment of a DOI. If interested in the current version, email David Bishel (dbishel@ur.rochester.edu).

