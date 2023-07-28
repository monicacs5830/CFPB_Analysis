# CFPB_Complaints_Analysis
## Understanding Financial Complaint Outcomes: A Data-Driven Approach

This project aims to analyze financial product complaints, predict their outcomes, and uncover insights into consumer behavior and possible improvements for financial services.

Table of Contents

1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Usage](#usage)
4. [Contributing](#contributing)

# Installation

To run this project, you will need to have Python installed on your machine and GCP setup.

### Python

You can download and install Python from the official website: https://www.python.org/downloads/

Then install all requiremnts using the following:

- !pip install requirements.txt

### Java (good to have)

Java installation can be achieved via SDKMAN!. Install SDKMAN! by following the instructions on their website: https://sdkman.io/install

After SDKMAN! is installed, a new Java version can be installed by running:

- sdk install java 11.0.3.hs-adpt
  
Then, the installed version can be set as default with:

- sdk use java 11.0.3.hs-adpt

Check the installed version with:

- java -version


#### Note versions that I am using are:
- **openjdk version "11.0.20" 2023-07-18**
- **OpenJDK Runtime Environment Homebrew (build 11.0.20+0)**
- **OpenJDK 64-Bit Server VM Homebrew (build 11.0.20+0, mixed mode)**
- **Python 3.9.13**

# Dataset
We use the Consumer Financial Protection Bureau (CFPB) Complaint Database, hosted on Google's BigQuery public datasets. This 2.15GB dataset consists of over 3.4 million rows of data on consumer complaints related to financial products and services reported to the CFPB from 2011 to 2023.

# Usage
After you've installed all the necessary prerequisites, you can run the application with the following command:
python app.py

# Contributing
Contributions are welcome. Please submit a Pull Request with any improvements or bug fixes.
