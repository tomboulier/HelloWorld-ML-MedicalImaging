[![main script](https://github.com/tomboulier/HelloWorld-ML-MedicalImaging/actions/workflows/python-app.yml/badge.svg)](https://github.com/tomboulier/HelloWorld-ML-MedicalImaging/actions/workflows/python-app.yml)
# Hello World for Machine Learning in Medical Imaging

The idea is based on this [article](https://link.springer.com/article/10.1007/s10278-018-0079-6), the code and data come from this [repo](https://github.com/paras42/Hello_World_Deep_Learning).

I want to start from this state of code and makes it "more MLOps", *i.e.*
- version control (Git and DVC),
- main code put outside notebook, to a proper Python package,
- testing (Pytest) and CI/CD (*e.g.* GitHub Actions),
- logging,
- configuration file for global variables (*e.g.* with dynaconf),
- Object-Oriented Programming with best practices, such as Design Patterns.

# Installation and usage

Clone the repository:
```
git clone https://github.com/tomboulier/HelloWorld-ML-MedicalImaging
cd HelloWorld-ML-MedicalImaging
```

Get data from the original repository:
```
curl -o data.zip -k https://raw.githubusercontent.com/paras42/Hello_World_Deep_Learning/master/Open_I_abd_vs_CXRs.zip
unzip data.zip
```
Install required dependencies:
```
pip install -r requirements.txt
```

Run the main script:
```
python main_script.py
```
