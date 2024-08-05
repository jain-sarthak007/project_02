{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "URmAxWVIRiR7"
      },
      "source": [
        "Importing the Dependencies"
      ]
    },
    {
      "source": [
        "!pip install streamlit\n",
        "import streamlit as st\n",
        "st.title('Medical Insurance Cost Prediction')"
      ],
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "collapsed": true,
        "id": "ddQbSaZ3XODZ",
        "outputId": "9969919b-9f6f-4536-f8dc-758af7b541cd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting streamlit\n",
            "  Downloading streamlit-1.37.0-py2.py3-none-any.whl.metadata (8.5 kB)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.2.2)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/lib/python3/dist-packages (from streamlit) (1.4)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.4.0)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.1.7)\n",
            "Requirement already satisfied: numpy<3,>=1.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.26.4)\n",
            "Requirement already satisfied: packaging<25,>=20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (24.1)\n",
            "Requirement already satisfied: pandas<3,>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.1.4)\n",
            "Requirement already satisfied: pillow<11,>=7.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (9.4.0)\n",
            "Requirement already satisfied: protobuf<6,>=3.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.20.3)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (14.0.2)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.31.0)\n",
            "Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (13.7.1)\n",
            "Collecting tenacity<9,>=8.1.0 (from streamlit)\n",
            "  Downloading tenacity-8.5.0-py3-none-any.whl.metadata (1.2 kB)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.12.2)\n",
            "Collecting gitpython!=3.1.19,<4,>=3.0.7 (from streamlit)\n",
            "  Downloading GitPython-3.1.43-py3-none-any.whl.metadata (13 kB)\n",
            "Collecting pydeck<1,>=0.8.0b4 (from streamlit)\n",
            "  Downloading pydeck-0.9.1-py2.py3-none-any.whl.metadata (4.1 kB)\n",
            "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.3.3)\n",
            "Collecting watchdog<5,>=2.1.5 (from streamlit)\n",
            "  Downloading watchdog-4.0.1-py3-none-manylinux2014_x86_64.whl.metadata (37 kB)\n",
            "Requirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (4.23.0)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.12.1)\n",
            "Collecting gitdb<5,>=4.0.1 (from gitpython!=3.1.19,<4,>=3.0.7->streamlit)\n",
            "  Downloading gitdb-4.0.11-py3-none-any.whl.metadata (1.2 kB)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2024.1)\n",
            "Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2024.1)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.7)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2024.7.4)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (2.16.1)\n",
            "Collecting smmap<6,>=3.0.1 (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit)\n",
            "  Downloading smmap-5.0.1-py3-none-any.whl.metadata (4.3 kB)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.5)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.2.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.12.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.35.1)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.19.1)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.3.0->streamlit) (1.16.0)\n",
            "Downloading streamlit-1.37.0-py2.py3-none-any.whl (8.7 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m8.7/8.7 MB\u001b[0m \u001b[31m62.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading GitPython-3.1.43-py3-none-any.whl (207 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m207.3/207.3 kB\u001b[0m \u001b[31m14.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading pydeck-0.9.1-py2.py3-none-any.whl (6.9 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m6.9/6.9 MB\u001b[0m \u001b[31m61.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading tenacity-8.5.0-py3-none-any.whl (28 kB)\n",
            "Downloading watchdog-4.0.1-py3-none-manylinux2014_x86_64.whl (83 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m83.0/83.0 kB\u001b[0m \u001b[31m6.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading gitdb-4.0.11-py3-none-any.whl (62 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m62.7/62.7 kB\u001b[0m \u001b[31m4.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading smmap-5.0.1-py3-none-any.whl (24 kB)\n",
            "Installing collected packages: watchdog, tenacity, smmap, pydeck, gitdb, gitpython, streamlit\n",
            "  Attempting uninstall: tenacity\n",
            "    Found existing installation: tenacity 9.0.0\n",
            "    Uninstalling tenacity-9.0.0:\n",
            "      Successfully uninstalled tenacity-9.0.0\n",
            "Successfully installed gitdb-4.0.11 gitpython-3.1.43 pydeck-0.9.1 smmap-5.0.1 streamlit-1.37.0 tenacity-8.5.0 watchdog-4.0.1\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2024-08-05 11:47:03.812 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "DeltaGenerator()"
            ]
          },
          "metadata": {},
          "execution_count": 2
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dCei9HxoNdG5"
      },
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn import metrics"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "k4ydeF1FSK2n"
      },
      "source": [
        "Data Collection & Analysis"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "HvkkGhfRSGdn"
      },
      "source": [
        "# loading the data from csv file to a Pandas DataFrame\n",
        "insurance_dataset = pd.read_csv('/content/medical_insurance.csv')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "jKHJM3wUSfhe",
        "outputId": "4e5712e4-2f4a-4bbb-97b6-a47fa0cc1ac6"
      },
      "source": [
        "# first 5 rows of the dataframe\n",
        "insurance_dataset.head()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   age     sex     bmi  children smoker     region      charges\n",
              "0   19  female  27.900         0    yes  southwest  16884.92400\n",
              "1   18    male  33.770         1     no  southeast   1725.55230\n",
              "2   28    male  33.000         3     no  southeast   4449.46200\n",
              "3   33    male  22.705         0     no  northwest  21984.47061\n",
              "4   32    male  28.880         0     no  northwest   3866.85520"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-1bbe23d1-7393-4f9a-8ae1-c56cebc92786\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>age</th>\n",
              "      <th>sex</th>\n",
              "      <th>bmi</th>\n",
              "      <th>children</th>\n",
              "      <th>smoker</th>\n",
              "      <th>region</th>\n",
              "      <th>charges</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>19</td>\n",
              "      <td>female</td>\n",
              "      <td>27.900</td>\n",
              "      <td>0</td>\n",
              "      <td>yes</td>\n",
              "      <td>southwest</td>\n",
              "      <td>16884.92400</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>18</td>\n",
              "      <td>male</td>\n",
              "      <td>33.770</td>\n",
              "      <td>1</td>\n",
              "      <td>no</td>\n",
              "      <td>southeast</td>\n",
              "      <td>1725.55230</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>28</td>\n",
              "      <td>male</td>\n",
              "      <td>33.000</td>\n",
              "      <td>3</td>\n",
              "      <td>no</td>\n",
              "      <td>southeast</td>\n",
              "      <td>4449.46200</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>33</td>\n",
              "      <td>male</td>\n",
              "      <td>22.705</td>\n",
              "      <td>0</td>\n",
              "      <td>no</td>\n",
              "      <td>northwest</td>\n",
              "      <td>21984.47061</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>32</td>\n",
              "      <td>male</td>\n",
              "      <td>28.880</td>\n",
              "      <td>0</td>\n",
              "      <td>no</td>\n",
              "      <td>northwest</td>\n",
              "      <td>3866.85520</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-1bbe23d1-7393-4f9a-8ae1-c56cebc92786')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-1bbe23d1-7393-4f9a-8ae1-c56cebc92786 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-1bbe23d1-7393-4f9a-8ae1-c56cebc92786');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-30e9233c-d482-4af7-b718-499c964eccb6\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-30e9233c-d482-4af7-b718-499c964eccb6')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-30e9233c-d482-4af7-b718-499c964eccb6 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "insurance_dataset",
              "summary": "{\n  \"name\": \"insurance_dataset\",\n  \"rows\": 2772,\n  \"fields\": [\n    {\n      \"column\": \"age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 14,\n        \"min\": 18,\n        \"max\": 64,\n        \"num_unique_values\": 47,\n        \"samples\": [\n          21,\n          45,\n          36\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"sex\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"male\",\n          \"female\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"bmi\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 6.1294486949652205,\n        \"min\": 15.96,\n        \"max\": 53.13,\n        \"num_unique_values\": 548,\n        \"samples\": [\n          23.18,\n          26.885\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"children\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1,\n        \"min\": 0,\n        \"max\": 5,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"smoker\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"no\",\n          \"yes\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"region\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 4,\n        \"samples\": [\n          \"southeast\",\n          \"northeast\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"charges\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 12151.768945168045,\n        \"min\": 1121.8739,\n        \"max\": 63770.42801,\n        \"num_unique_values\": 1337,\n        \"samples\": [\n          8688.85885,\n          5708.867\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qB730MywSmwM",
        "outputId": "5a379896-085e-4e17-d751-577d6bcf168b"
      },
      "source": [
        "# number of rows and columns\n",
        "insurance_dataset.shape"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(2772, 7)"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zbBK33o-S_Q7",
        "outputId": "5ea2f9e2-f5b3-4e20-ee02-3ffa4a3e6b6f"
      },
      "source": [
        "# getting some informations about the dataset\n",
        "insurance_dataset.info()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 2772 entries, 0 to 2771\n",
            "Data columns (total 7 columns):\n",
            " #   Column    Non-Null Count  Dtype  \n",
            "---  ------    --------------  -----  \n",
            " 0   age       2772 non-null   int64  \n",
            " 1   sex       2772 non-null   object \n",
            " 2   bmi       2772 non-null   float64\n",
            " 3   children  2772 non-null   int64  \n",
            " 4   smoker    2772 non-null   object \n",
            " 5   region    2772 non-null   object \n",
            " 6   charges   2772 non-null   float64\n",
            "dtypes: float64(2), int64(2), object(3)\n",
            "memory usage: 151.7+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yjwwR0fnTg2H"
      },
      "source": [
        "Categorical Features:\n",
        "- Sex\n",
        "- Smoker\n",
        "- Region"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 303
        },
        "id": "-DsX_XZUTOw8",
        "outputId": "f874f962-5d1d-4d96-8b45-361f6d1dcf76"
      },
      "source": [
        "# checking for missing values\n",
        "insurance_dataset.isnull().sum()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "age         0\n",
              "sex         0\n",
              "bmi         0\n",
              "children    0\n",
              "smoker      0\n",
              "region      0\n",
              "charges     0\n",
              "dtype: int64"
            ],
            "text/html": [
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>age</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>sex</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>bmi</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>children</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>smoker</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>region</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>charges</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table><br><label><b>dtype:</b> int64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "39CD23eNUBpW"
      },
      "source": [
        "Data Analysis"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        },
        "id": "H3JJHLVgT4kV",
        "outputId": "6a7eab1c-014a-48b7-9270-28e9d9f46919"
      },
      "source": [
        "# statistical Measures of the dataset\n",
        "insurance_dataset.describe()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "               age          bmi     children       charges\n",
              "count  2772.000000  2772.000000  2772.000000   2772.000000\n",
              "mean     39.109668    30.701349     1.101732  13261.369959\n",
              "std      14.081459     6.129449     1.214806  12151.768945\n",
              "min      18.000000    15.960000     0.000000   1121.873900\n",
              "25%      26.000000    26.220000     0.000000   4687.797000\n",
              "50%      39.000000    30.447500     1.000000   9333.014350\n",
              "75%      51.000000    34.770000     2.000000  16577.779500\n",
              "max      64.000000    53.130000     5.000000  63770.428010"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-4e564d9d-2e87-4057-924b-a9e06978157f\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>age</th>\n",
              "      <th>bmi</th>\n",
              "      <th>children</th>\n",
              "      <th>charges</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>2772.000000</td>\n",
              "      <td>2772.000000</td>\n",
              "      <td>2772.000000</td>\n",
              "      <td>2772.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>39.109668</td>\n",
              "      <td>30.701349</td>\n",
              "      <td>1.101732</td>\n",
              "      <td>13261.369959</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>14.081459</td>\n",
              "      <td>6.129449</td>\n",
              "      <td>1.214806</td>\n",
              "      <td>12151.768945</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>18.000000</td>\n",
              "      <td>15.960000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1121.873900</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>26.000000</td>\n",
              "      <td>26.220000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>4687.797000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>39.000000</td>\n",
              "      <td>30.447500</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>9333.014350</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>51.000000</td>\n",
              "      <td>34.770000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>16577.779500</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>64.000000</td>\n",
              "      <td>53.130000</td>\n",
              "      <td>5.000000</td>\n",
              "      <td>63770.428010</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-4e564d9d-2e87-4057-924b-a9e06978157f')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-4e564d9d-2e87-4057-924b-a9e06978157f button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-4e564d9d-2e87-4057-924b-a9e06978157f');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-9b7c9beb-4a77-4b8f-a657-4cba136743c7\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-9b7c9beb-4a77-4b8f-a657-4cba136743c7')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-9b7c9beb-4a77-4b8f-a657-4cba136743c7 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "summary": "{\n  \"name\": \"insurance_dataset\",\n  \"rows\": 8,\n  \"fields\": [\n    {\n      \"column\": \"age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 967.505576170101,\n        \"min\": 14.081459420836477,\n        \"max\": 2772.0,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          39.10966810966811,\n          39.0,\n          2772.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"bmi\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 970.1788543044319,\n        \"min\": 6.1294486949652205,\n        \"max\": 2772.0,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          30.70134920634921,\n          30.447499999999998,\n          2772.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"children\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 979.5302099405272,\n        \"min\": 0.0,\n        \"max\": 2772.0,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          2772.0,\n          1.1017316017316017,\n          2.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"charges\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 20252.240669716044,\n        \"min\": 1121.8739,\n        \"max\": 63770.42801,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          13261.369959046897,\n          9333.014350000001,\n          2772.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 541
        },
        "id": "thRxLvZwUJNW",
        "outputId": "babf4c46-d952-4d9d-a8b9-b9507bf0a79c"
      },
      "source": [
        "# distribution of age value\n",
        "sns.set()\n",
        "plt.figure(figsize=(6,6))\n",
        "sns.displot(insurance_dataset['age'])\n",
        "plt.title('Age Distribution')\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 600x600 with 0 Axes>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 500x500 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAeQAAAH6CAYAAADWcj8SAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABHeElEQVR4nO3de1xUdeL/8ffMcPECg5f1sgkq2EpeILALmkhqpoKWdnEzVy0ry1Jb7eemmZpuptXaqpm63rtuF7NaXcnMNF2N1eqr+bVy09A0/aLmhQFEgZn5/cEyOYI6N+AAr+fj4QPmnM85n8/5cMb3nM+5jMnpdDoFAAAqlbmyGwAAAAhkAAAMgUAGAMAACGQAAAyAQAYAwAAIZAAADIBABgDAAAhkAAAMgEAGAMAACGQAkqSff/5ZsbGx+uCDD8q9rg8++ECxsbH6+eefXdO6d++uRx55pNzrlqTt27crNjZW27dvr5D6AE8EVXYDgOrkrbfe0p///GfFx8dr5cqVldqW2NhY1+8Wi0VhYWGKjIxUhw4dNHDgQF199dUBqeett95S7dq1deeddwZkfYFk5LYBFzPxLGsgcAYOHKjjx4/ryJEjWr9+vVq0aFFpbYmNjVXnzp3Vr18/OZ1O5ebmau/evVq3bp3y8/M1btw4DRs2zFXe6XSqoKBAQUFBslgsHtfTt29f1a9fX2+88YbHy9jtdhUVFSkkJEQmk0lS8RHy7373Oy1atMjzjfSxbQ6HQ4WFhQoODpbZzEAhjIE9EQiQw4cPa+fOnXrqqafUoEEDrVmzprKbpJYtW6pfv37q37+/Bg8erOnTp+vTTz9VXFycnn/+eW3evNlV1mQyKTQ01Ksw9tbZs2clFR+xh4aGusK4opnNZoWGhhLGMBT2RiBA1qxZo4iICN18883q1avXJQP59OnT+tOf/qQOHTro+uuv1/jx47V3794yz9/++OOPevzxx3XjjTcqLi5Od955pz777DO/2lm/fn399a9/VVBQkBYuXOiaXtY55BMnTuipp55SSkqK2rdvr+TkZD366KOuc7/du3fXvn37tGPHDsXGxio2NlZDhgyR9Ot54h07dmjq1Knq1KmTbr75Zrd5F55DLrF161b169dPcXFxSktL0/r1693mz5s3z204vsTF67xc2y51Dvnjjz/WnXfeqfj4eCUlJWncuHE6duyYW5kJEyYoMTFRx44d02OPPabExER17NhRL7zwgux2u2d/BKAMnEMGAmTNmjW69dZbFRISor59++rtt9/W7t27FR8f7yrjcDj06KOPavfu3br33nsVExOjzz77TOPHjy+1vn379unee+9VkyZNNHz4cNWpU0cff/yxRo4cqXnz5unWW2/1ua1XXXWVbrjhBm3fvl25ubkKCwsrs9zo0aO1f/9+DR48WM2aNdOpU6e0bds2/d///Z8iIyM1ceJEPfvss6pTp45GjBghSfrNb37jto5p06apQYMGGjlypOsI+VIOHjyosWPHauDAgbrjjju0atUq/fGPf9TSpUvVuXNnr7bRk7Zd6IMPPtBTTz2luLg4PfHEEzp58qRef/11/c///I8++ugjWa1WV1m73a4HH3xQ8fHxevLJJ5WRkaHly5crKipKgwYN8qqdQAkCGQiAPXv2KDMzU5MnT5YkXXfddWratKnWrFnjFsgbNmzQzp07NXHiRN13332SpHvvvdftXG6J5557Tr/97W+1atUqhYSESJIGDRqke++9V7NmzfIrkCXpd7/7nTIyMvTzzz/rmmuuKTXfZrNp586devLJJ/Xggw+6pl94JXSPHj00Z84c1a9fX/369SuznoiICL366qseDYUfPHhQ8+bNU8+ePSVJd999t3r37q1Zs2Z5HcietK1EYWGhZs2apdatW+utt95SaGiopOK/4yOPPKJXX31Vjz/+uKv8+fPnlZqaqpEjR0oq/hvecccdev/99wlk+IwhayAA1qxZo9/85jdKSkqSVHw+Ni0tTenp6W7DmP/6178UHBys3//+965pZrNZf/jDH9zWd+bMGf373/9WamqqcnNzderUKZ06dUqnT59WcnKyDh48WGoo1Vt16tSRJOXl5ZU5v1atWgoODtaOHTuUnZ3tcz2///3vPT4v3bhxY7cPGmFhYerfv7++++47nThxwuc2XMmePXt08uRJ3Xvvva4wlqSuXbsqJiZGn3/+eall7r33XrfX1113XZlD8ICnOEIG/GS327V27VolJSW5/YccHx+v5cuXKyMjQ8nJyZKko0ePqlGjRqpdu7bbOpo3b+72+tChQ3I6nZo7d67mzp1bZr0nT55UkyZNfG53yfBx3bp1y5wfEhKicePG6YUXXlDnzp117bXXqmvXrurfv78aNWrkcT2RkZEel23RokWpC71atmwpSTpy5IhX9Xrj6NGjkqTo6OhS82JiYvT111+7TQsNDVWDBg3cpkVERPj1wQUgkAE//fvf/9aJEye0du1arV27ttT8NWvWuALZUw6HQ5L0wAMPqEuXLmWWuTjEvbVv3z5ZLJbLBub999+v7t27a8OGDdq6davmzp2rxYsX67XXXlPbtm09qufCI85AuNSV2RV5QVV5XomOmotABvy0Zs0aNWzYUFOmTCk179NPP9Wnn36qadOmqVatWrrqqqu0fft25efnux0lHzp0yG25qKgoSVJwcLBuuummgLf56NGj+vLLL5WQkHDJC7pKNG/eXA888IAeeOABHTx4UP3799fy5cs1a9YsSZcOSF/89NNPcjqdbus8ePCgJKlZs2aS5Lq4ymazuV1oVXKUeyFP23bVVVdJkg4cOKBOnTq5zTtw4IBrPlCeOIcM+OHcuXNav369unbtqt69e5f694c//EF5eXnauHGjJCk5OVmFhYV67733XOtwOBx666233NbbsGFD3XjjjXr33Xd1/PjxUvWeOnXK5zafOXNGTzzxhOx2u+vq47Lk5+fr/PnzbtOaN2+uunXrqqCgwDWtdu3astlsPrfnQsePH9enn37qep2bm6uPPvpIbdq0cQ1Xl4wMfPnll65yZ8+e1UcffVRqfZ62rX379mrYsKHeeecdt23bvHmzfvzxR3Xt2tXHLQI8xxEy4IeNGzcqLy9P3bt3L3N+QkKCGjRooNWrVystLU09evRQfHy8XnjhBR06dEgxMTHauHGj69zjhUd0zzzzjAYNGqTbbrtNv//97xUVFaVffvlFu3btUlZWllavXn3F9h08eFD/+Mc/5HQ6lZeX53pS19mzZzVhwgSlpKRcdtn7779fvXv31tVXXy2LxaINGzbol19+UZ8+fVzl2rVrp7ffflsLFixQixYt1KBBg1JHmZ5q2bKlnn76af3v//6vGjZsqFWrVunkyZOaOXOmq0znzp111VVX6emnn1ZmZqYsFotWrVql+vXrlzpK9rRtwcHBGjdunJ566ikNHjxYffr0cd321KxZM91///0+bQ/gDQIZ8MPq1asVGhp6yVtyzGazunbtqjVr1uj06dOqX7++Fi1apOeee04ffvihzGazbr31Vo0cObLUFb5XX321Vq1apVdeeUUffvihzpw5owYNGqht27au222uZNu2bdq2bZvM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VvPnz1f37t0VGRmpQ4cOqbCwsNRXcLZq1UqSSn1lZ02zbt06/fDDDxo5cmSpeex3l9e3b1+1adNGt9xyixYtWuQacqXfLu2bb75R69attWDBAnXq1Ent27fXwIED9c0330hSjXu/cg65Cvjqq6+Unp6u8ePHSyr+JitJslqtbuVKXpfMr8m6deumY8eOSZK6dOmil156SRJ9dzn5+fl6/vnnNXbsWIWFhZWaT9+VrVGjRho9erSuvfZamUwmbdy4UXPmzNGxY8c0ZcoU+u0yTpw4oT179uiHH37QM888o9q1a+tvf/ubHnjgAa1fv77G9R2BbHBZWVkaO3askpKSNHTo0MpuTpWxePFi5efna//+/Vq4cKFGjBihFStWVHazDG3hwoVq2LCh7rrrrspuSpXSpUsXdenSxfU6OTlZoaGheu211zRixIhKbJnxOZ1OnT17VnPnztU111wjSbr22mvVvXt3vfnmm0pOTq7kFlYshqwNzGazafjw4apXr57mzZsns7n4zxURESFJpW7lsdlsbvNrsmuuuUaJiYkaMGCAFixYoO3bt+vTTz+l7y7hyJEjWr58uR5//HHl5OTIZrPp7NmzkqSzZ88qLy+PvvNCamqq7Ha7vv/+e/rtMqxWq+rVq+cKY6n4mo+2bdtq//79Na7vCGSDOnfunB555BHl5OSUup2i5HzKxedPMjMzFRwcrKioqAptq9HFxsYqODhYhw4dUvPmzRUcHFxm30kqda6qpvj5559VWFiohx9+WDfccINuuOEG19Hd0KFDNWzYMPY7H9Fvl3b11Vdfct758+dr3PuVQDagoqIijRkzRpmZmVq6dKnb/YySFBUVpZYtW2rdunVu09PT09WpUyeFhIRUZHMN75tvvlFhYaEiIyMVEhKipKQkffLJJ25l0tPT1apVK0VGRlZSKytXmzZt9Prrr7v9e+qppyRJ06ZN0zPPPMN+54X09HRZLBa1bduWfruMbt266cyZM/r+++9d006fPq1vv/1W7dq1q3HvV84hG9C0adO0adMmTZgwQbm5udq1a5drXtu2bRUSEqLRo0dr3Lhxat68uZKSkpSenq7du3frzTffrLyGG8CoUaPUvn17xcbGqlatWtq7d6+WLVum2NhY9ejRQ5L06KOPaujQoZo6dapSU1O1fft2/fOf/9Ts2bMrufWVx2q1Kikpqcx57dq1U7t27SSJ/a4MDz74oJKSkhQbGytJ+uyzz/Tee+9p6NChatSokST67VJ69OihuLg4Pf744xo7dqxCQ0O1ePFihYSEaNCgQZJq1vvV5HRe9DxGVLru3bvryJEjZc777LPPXJ8KV65cqSVLlujo0aOKjo7WE088oW7dulVkUw1n8eLFSk9P16FDh+R0OtWsWTPdeuutevDBB92uHP7ss880Z84cHThwQFdddZUefvhh3X333ZXYcuPZvn27hg4dqvfff19xcXGu6ex37qZPn65//etfysrKksPhUMuWLTVgwAANGTJEJpPJVY5+K9upU6c0c+ZMbdq0SYWFhbr++uv11FNPuQ1n15T3K4EMAIABcA4ZAAADIJABADAAAhkAAAMgkAEAMAACGQAAAyCQAQAwAAIZAAADIJABADAAAhkAAAMgkAEAMAACGQAAAyCQAbgcOXJEU6dOVa9evRQfH6+kpCQ9/vjj+vnnn0uV3bt3rwYPHqz4+HilpKRowYIFWrVqlWJjY0uV37x5swYNGqSEhAQlJibq4Ycf1r59+ypqs4AqgS+XAOCybt06LVy4ULfccouaNm2qI0eO6O2331ZYWJjWrl2r2rVrS5KOHTum22+/XZI0ZMgQ1alTRytXrlRISIj27t3r9q1kH330kSZMmKDk5GR17dpV+fn5evvtt5WTk6MPP/yw2n2nLeArAhmAy7lz51SrVi23abt27dI999yjF154Qf3795dU/JWDb775pj788EO1adNGknTmzBn16tVLZ86ccQVyXl6eunbtqt69e+vZZ591rfOXX35R7969lZqa6jYdqMkYsgbgcmEYFxYW6vTp02revLmsVqu+++4717x//etfSkhIcIWxJNWrV0+33Xab2/q++OIL2Ww29enTR6dOnXL9M5vNuvbaa7V9+/by3yigigiq7AYAMI5z585p0aJF+uCDD3Ts2DFdOICWk5Pj+v3IkSNKSEgotXzz5s3dXh88eFCSdN9995VZX1hYmP+NBqoJAhmAy7PPPqsPPvhA9913nxISEhQeHi6TyaSxY8fKl7NbJcu8+OKLatSoUan5FovF7zYD1QWBDMDlk08+Uf/+/TVhwgTXtPPnz7sdHUtSs2bN9NNPP5Va/tChQ26vo6KiJEkNGzbUTTfdVA4tBqoPziEDcCnriPWNN96Q3W53m5acnKxdu3bp+++/d007c+aM1qxZ41auS5cuCgsL06JFi1RYWFhq3adOnQpQy4GqjyNkAC5du3bVP/7xD4WFhenqq6/Wrl279MUXX6hevXpu5R566CGtXr1aw4YN0+DBg123Pf32t7/VmTNnZDKZJBWfI546daqefPJJ3XnnnUpLS1ODBg109OhRbd68WR06dNCUKVMqYUsB4yGQAbg8/fTTMpvNWrNmjc6fP68OHTpoxYoVeuihh9zK/fa3v9Xrr7+u6dOna9GiRWrQoIH+8Ic/qHbt2po+fbpCQ0NdZW+77TY1btxYixcv1rJly1RQUKAmTZro+uuv15133lnRmwgYFvchAwiY5557Tu+++6527tzJBVuAlziHDMAn586dc3t9+vRprV69Wtdddx1hDPiAIWsAPrnnnnt04403qlWrVvrll1+0atUq5ebm6rHHHqvspgFVEkPWAHzy17/+VZ988omysrJkMpnUtm1bjRo1itubAB8RyAAAGADnkAEAMAACGQAAAyCQAQAwAAIZAAADIJABADAAAhkAAAMgkAEAMAACGQAAAyCQAQAwgP8P6qlrFBJYJ4YAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 573
        },
        "id": "sJaFjdJJU644",
        "outputId": "9db51d89-5eca-492b-a222-1e99ed4e2dcb"
      },
      "source": [
        "# Gender column\n",
        "plt.figure(figsize=(6,6))\n",
        "sns.countplot(x='sex', data=insurance_dataset)\n",
        "plt.title('Sex Distribution')\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 600x600 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjAAAAIsCAYAAADs5ZOPAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABAJ0lEQVR4nO3de1zUVf7H8fcA4qIxKC5a3gGDtEQoVyRYystqoKVZZpraeu3mBddSY9VFM3Uz08S70papaZlblqyrWUkSa1taZpaaoCGr2XphUDBu8/vDB/NzFjXk4nDi9Xw8fOic7/l+v5+j8x3fnO+ZGYvdbrcLAADAIG6uLgAAAOBaEWAAAIBxCDAAAMA4BBgAAGAcAgwAADAOAQYAABiHAAMAAIxDgAEAAMYhwAAAAOMQYAAYpXPnzpo0aVKVn+fYsWMKDg7Wxo0bHW2TJk1SWFhYlZ+7RHBwsBITE6/b+QCTeLi6AADX14EDB7Ro0SJ9/fXX+u9//6t69eqpVatW6ty5swYNGnRdaxk0aJA+++wzSZLFYlGdOnXk5+enkJAQ9e7dW5GRkZVynh07dmjv3r0aPXp0pRyvMlXn2oDqjAAD1CC7d+/W4MGD1bhxY/Xt21d+fn46fvy4vvrqK61ateq6BxhJuvHGG/WnP/1JkpSXl6ejR49q27Zt2rRpk2JiYjRnzhzVqlXL0X/Lli2yWCzXdI4dO3ZozZo11xQSmjRpor1798rDo2pfJq9W2969e+Xu7l6l5wdMRYABapClS5fK29tbGzZskNVqddp26tQpl9Tk7e2tXr16ObU9/fTTmjFjhtauXasmTZromWeecWzz9PSs0noKCwtVXFwsT09P1a5du0rP9UtcfX6gOmMNDFCD/PDDD2rVqlWp8CJJDRo0KNX27rvvqk+fPgoJCVGHDh00btw4HT9+3LH97bffVnBwsDZs2OC039KlSxUcHKwdO3aUq053d3dNnjxZrVq10po1a5STk+PY9r9rYAoKCrRw4UJ169ZNbdu2VXh4uPr376/U1FRJF9etrFmzRtLFNSUlv6T/X+eSlJSkV199VV27dlXbtm11+PDhy66BKZGZmalhw4YpNDRUUVFRWrhwoex2u2P7rl27FBwcrF27djnt97/HvFptJW3/uwZm//79Gj58uG6//XaFhYXp0Ucf1ZdffunUZ+PGjQoODtYXX3yhWbNmqWPHjgoNDdVTTz2l06dPl+0fAajmmIEBapAmTZpoz549OnjwoIKCgq7ad8mSJXr55ZcVExOjBx98UKdPn9bq1av1yCOP6J133pHVatUDDzygbdu2afbs2YqMjNRNN92kAwcOaOHChXrwwQd11113lbtWd3d39ejRQy+//LK++OIL3X333Zftt3DhQi1btkx9+/ZVSEiIzp07p3379umbb75RZGSk+vXrp5MnTyo1NVUvvPDCZY+xceNG/fzzz3rooYfk6ekpHx8fFRcXX7ZvUVGRhg8frnbt2umZZ57RJ598osTERBUVFWns2LHXNMay1HapQ4cO6ZFHHlHdunU1fPhweXh4aP369Ro0aJBWr16tdu3aOfWfMWOGrFarRo0apaysLL322muaPn265s+ff011AtURAQaoQYYOHaoRI0aod+/eCgkJ0R133KGIiAiFh4c7rTPJyspSYmKi4uLi9Pjjjzvau3Xrpvvvv19r1651tD/33HPq2bOn/vznP2vp0qWaNGmS/Pz89Oyzz1a43pKQ9cMPP1yxz8cff6y77rpLzz333GW3h4WFqWXLlkpNTS11q6rEiRMntG3bNvn6+jrajh07dtm+P//8s37/+99r8uTJkqQBAwbo8ccf14oVKzRo0CCnY/ySstR2qfnz56ugoEBvvPGGmjVrJknq3bu37rnnHs2ZM0erV6926l+vXj298sorjjVDxcXFev3115WTkyNvb+8y1wlUR9xCAmqQyMhIrVu3Tp07d9Z3332nlStXatiwYYqOjtb27dsd/bZt26bi4mLFxMTo9OnTjl+//e1v1aJFC6dbI35+fpo6dapSU1P1yCOP6Ntvv9XMmTN1ww03VLjeOnXqSJLOnz9/xT5Wq1WHDh3SkSNHyn2ebt26XVPweOSRRxx/tlgseuSRR1RQUKC0tLRy1/BLioqKlJqaqq5duzrCiyQ1bNhQPXv21BdffKFz58457fPQQw85LXhu3769ioqKlJWVVWV1AtcLMzBADRMSEqKFCxcqPz9f3333nT744AO9+uqrGjt2rN555x21atVKR44ckd1uV7du3S57jP99Z06PHj20adMmffzxx+rXr58iIiIqpdbc3FxJUt26da/YZ8yYMXryySfVvXt3BQUFKSoqSr169dItt9xS5vM0bdq0zH3d3NycAoQk+fv7S1KVBoPTp08r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oeOHhoaqTp06mjZtmkaOHKkff/xRiYmJvxhKqrouoKYICAjQjTfeqLlz56q4uFi5ublasGCBGjZseNX9IiMj1alTJw0fPlzDhw9XcHCw8vLy9P333+vo0aPMjhqIAINfFU9PT61atUqJiYlaunSpfvrpJ9WrV09t2rTRgAEDKnz83/72t3r55Zf1wgsv6Mknn1TLli01bdo0rVy50qV1ATWFp6enEhMTNX36dI0dO1Y33XSTnnjiCf3rX//Svn37rrrvggULtHz5cr3xxhvKysqSt7e3br75ZvXp0+c6VY/KZLGXLN8GAAAwBJ/ECwAAjEOAAQAAxiHAAAAA4xBgAACAcQgwAADAOAQYAABgHAIMAAAwDgEGAAAYhwADAACMQ4ABAADGIcAAAADjEGAAGO3cuXN6/vnn1blzZ912222KiIjQkCFD9M033zj6fPXVVxo2bJjuuOMOtWvXTgMHDtQXX3zh2H748GGFhIRowoQJTsf+/PPP1bp1a82ZM+e6jQdA2fBljgCMNn78eP3zn//UwIEDFRgYqLNnz+qLL75QbGys7rvvPqWlpWnEiBG67bbb1L17d1ksFm3cuFHp6elau3atQkJCJElJSUl64YUXtHjxYnXp0kW5ubnq1auXPD099fe//12enp4uHimASxFgABitffv2uu+++zR16tRS2+x2u+655x41bdpUK1eulMVikSRduHBBPXr0UIsWLfTKK69IkoqLizVw4EAdPXpU77//vhITE7V+/XqtW7dObdu2va5jAvDLuIUEwGhWq1VfffWVfvzxx1Lbvv32Wx05ckT33nuvzpw5o9OnT+v06dPKzc1VRESE/v3vf6u4uFiS5ObmptmzZys3N1cjRozQ2rVrNXLkSMILUE0xAwPAaMnJyZo0aZIKCgp066236q677lLv3r3VrFkzJScna9y4cVfd/7PPPpOPj4/jccmtpKCgIG3cuFG1atWq6iEAKAcPVxcAABURGxur9u3ba9u2bUpNTVVSUpJWrFihxMRElfx8NmHCBLVu3fqy+9epU8fpcWpqqiTp5MmTOnv2rPz8/Kp2AADKhRkYAL8qp06d0v33368mTZro2WefVd++fTV9+nT169fvF/d94403lJCQoHHjxmnZsmXq2LGjlixZch2qBnCtWAMDwFhFRUXKyclxamvQoIEaNmyo/Px83XbbbWrevLleeeUVnT9/vtT+p0+fdvw5MzNTL7zwgrp3767HH39cEydO1Icffqh33nmnqocBoByYgQFgLJvNprvuukvdu3fXLbfcojp16ujTTz/VP/7xD02aNElDhgzRrl27NGLECDVo0EB9+vRRo0aN9OOPP2rXrl264YYbtHTpUtntdg0ePFjff/+9Nm/eLF9fX0nS0KFD9fXXX+v9999Xo0aNXDxaAJciwAAwVn5+vubPn6/U1FRlZmbKbrerefPm6tevnwYMGODo9+2332rx4sX67LPPlJubKz8/P4WEhKhfv36KiIjQqlWr9PzzzysxMVHdunVz7Hf8+HH17NlTd9xxh5YvX+6KIQK4AgIMAAAwDmtgAACAcQgwAADAOAQYAABgHAIMAAAwDgEGAAAYhwADAACMQ4ABAADGIcAAAADjEGAAAIBxCDAAAMA4BBgAAGAcAgwAADAOAQYAABjn/wBUA9kKx3RAigAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 178
        },
        "id": "zV5Jx_ytVmIy",
        "outputId": "1e1eb869-6bf5-4d62-edf3-01af9c23fff8"
      },
      "source": [
        "insurance_dataset['sex'].value_counts()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "sex\n",
              "male      1406\n",
              "female    1366\n",
              "Name: count, dtype: int64"
            ],
            "text/html": [
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>count</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>sex</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>male</th>\n",
              "      <td>1406</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>female</th>\n",
              "      <td>1366</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table><br><label><b>dtype:</b> int64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 541
        },
        "id": "bqUZ1u_9Vyg_",
        "outputId": "dbe7e754-aee7-43eb-bb01-513e9cccfec8"
      },
      "source": [
        "# bmi distribution\n",
        "plt.figure(figsize=(6,6))\n",
        "sns.displot(insurance_dataset['bmi'])\n",
        "plt.title('BMI Distribution')\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 600x600 with 0 Axes>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 500x500 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAeQAAAH6CAYAAADWcj8SAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABBA0lEQVR4nO3deXgUhf3H8c/s5pAENgFFVMKRpBJRAwmKIQIR40FBhHpE8AhoEaEWBNQWfggIHsBTbUFRqUJExQso2iIGRARBkIIWhaJWsUHkeFAeMNkcHNnj90ealTWB7CbZ7Ozu+/U8PGFmvpn5biazn+ychtvtdgsAAASVJdgNAAAAAhkAAFMgkAEAMAECGQAAEyCQAQAwAQIZAAATIJABADABAhkAABMgkAEAMAECGYAmTpyo3NzcJllWbm6uJk6c6Bl+6623lJaWpn//+99Nsvz8/Hzl5+c3ybIAfxDIQANVB8rJ/7Kzs5Wfn6/169fXqK+ueeihh2qd3+zZsz01R44c8YyfOHGiMjMz6+xn7ty5Xr107dpVffr00ahRo7Rs2TKdOHGi/i/2JN9++63mzp2rffv2Ncr8GpOZewNOJSrYDQDh4r777lNSUpLcbrcOHz6st99+W/fcc4/++te/6sorr/SqjY2N1erVq/Xwww8rJibGa9qKFSsUGxur48ePN6ifadOmKS4uTidOnNAPP/ygjRs3atKkSXr55Zf1/PPP69xzz/XUPvroo/L3tvbffvutnnnmGV122WVKSkry+ftWrVolwzD8Wpa/TtdbQUFBQJcN1BeBDDSSnJwcpaene4Zvvvlm9ezZUytWrKgRyL1799batWu1YcMGXX311Z7x27Zt0759+9S3b1+99957Deqnb9++atWqlWd49OjRWr58uSZMmKCxY8dqyZIlnmnR0dENWlZd3G63jh8/rjPOOKPGHyBNLdjLB06FXdZAgNhsNsXGxioqqubfvW3atNGll16qFStWeI1/55131KlTJ51//vkB6WngwIHKy8vT9u3btWnTJs/42o4hv/vuu7rxxhuVmZmpbt266frrr9fLL78sqWo3/dixYyVJQ4cO9ewe37Jli6Sq48QjR47URx99pBtvvFFdunTRm2++6Zl28jHkaseOHdPUqVOVlZWlbt266Y9//KNKSkq8atLS0jR37twa33vyPOvqrbZjyIcPH9akSZN0+eWXKz09XQMHDtTbb7/tVbNv3z6lpaWpoKBAixcv1tVXX62LL75YN910k3bs2HG6HzvgEz4hA42krKzMc8z38OHDWrRokSoqKjRw4MBa66+//no9/vjjKi8vV3x8vBwOh1atWqW77rqrwburT2fgwIFavHixNm7cqJ49e9Zas2nTJt1///3Kzs7Wgw8+KEkqKirStm3bNGzYMHXv3l35+flatGiRRo0apZSUFElSamqqZx67d+/WAw88oMGDB+uWW25RcnLyaft65JFHZLPZNHr0aO3evVtvvPGGDhw4oEWLFvm1i9uX3k527Ngx5efn6/vvv9ftt9+upKQkrVq1ShMnTpTdbtewYcO86lesWKHy8nINHjxYhmFowYIFGjNmjNasWRPwPQ0IbwQy0EjuvPNOr+GYmBjNmDHjlKHXt29fPfLII1qzZo0GDRqkTZs26aefftJ1112nt956K2B9durUSZK0d+/eU9Z8+OGHat68uQoKCmS1WmtMb9eunS699FItWrRIl19+ubKysmrU7NmzRwsWLFDv3r196is6OlovvfSSJ9TOO+88PfHEE1q7dq2uuuoqn+bha28nW7x4sf773//qiSee8PzxNGTIEOXn52vOnDm66aab1Lx5c0/9gQMHtHr1aiUkJEiSkpOTde+992rjxo01Dk0A/mCXNdBIpk6dqoULF2rhwoV64oknlJWVpcmTJ2v16tW11ickJKh379569913JVXtrs7MzFTbtm0D2mdcXJwkqby8/JQ1NptNR48e9dqt7a+kpCSfw1iSBg8e7PUJ89Zbb1VUVFStZ6o3pg0bNqh169YaMGCAZ1x0dLTy8/NVUVGhTz75xKu+f//+njCWpEsvvVTS6f/AAXxBIAONpEuXLrr88st1+eWXa+DAgXrhhReUmpqqRx555JSXGl1//fX6+OOPdeDAAX3wwQdeoRAoFRUVkqT4+PhT1tx2223q2LGjRowYoZycHP3f//2fNmzY4Ndy/DnzWpI6dOjgNRwfH6/WrVtr//79fs3HX/v371eHDh1ksXi/HVbv4j5w4IDX+JPPTpfkCWe73R7ALhEJCGQgQCwWi7KysnTo0CHt2bOn1prc3FxFR0drwoQJOnHihPr16xfwvr755htJUvv27U9Zc+aZZ+rvf/+75s2bp9zcXG3ZskUjRozQhAkTfF7OGWec0eBe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AIDTMtUu66NHj2r9+vWSpP3796usrEyrVq2SJF122WUqKirSggULdM0116ht27b68ccftXDhQh06dEhPPfWUZz6ZmZnq1auXJk2apAkTJig2NlazZ89WWlqarr322qC8NgAATsdUgXz48GGNHTvWa1z18CuvvKJzzjlHlZWVmj17toqLi9WsWTNlZmZq+vTp6tKli9f3zZkzRzNnztTUqVPlcDjUq1cvTZ48mbt0AQBMyVTplJSUpK+//vq0NQUFBT7Nq0WLFpoxY4ZmzJjRGK0BABBQIX8MGQCAcEAgAwBgAgQyAAAmYKpjyABCQ/UNPxqrDgCBDMAPsTFWOV2uiLifO9DUCGQAPouJsspqseipN7eptPx4nfXnntVcwwel11kHgEAGUA+l5cdVUnaizroWcXXXAKjCSV0AAJgAgQwAgAkQyAAAmADHkAGYij+XSlU9ZzmAzQBNiEAGYAr1uaTK4XCpuLicUEZYIJABmIK/l1S1iI/V2CHdZBiG3CQywgCBDMBUfL2kCgg3BDJCni/HHLmFIwCzI5ARsriNI4BwQiAjZPlzzJFbOAIwOwIZIc+XY47cwhGA2XFjEAAATIBABgDABAhkAABMgEAGAMAECGQAAEyAQAYAwAQIZAAATIBABgDABAhkAABMgEAGAMAECGQAAEyAQAYAwAQIZAAATIBABgDABAhkAABMwFTPQ96zZ48KCgq0fft27dq1SykpKVqxYoVnellZmRYuXKj169fru+++U0xMjLp06aLx48crLS3NU7dv3z5dddVVNebftWtXLVmypEleCwAA/jBVIO/atUvr169X165d5XK55Ha7vaYfOHBAixcv1k033aRx48bp+PHjevHFFzV48GAtW7ZMqampXvX333+/srKyPMPx8fFN8joAAPCXqQI5NzdXV199tSRp4sSJ2rlzp9f0pKQkvf/++2rWrJlnXI8ePZSbm6vXX39dU6ZM8arv0KGDMjIyAt43AAANZapAtlhOf0g7Li6uxrj4+Hi1b99eP/74Y6DaAgAg4EL+pC673e453vxL06ZNU+fOnZWdna3JkyeruLi46RsEAMAHpvqEXB9PPPGEDMPQrbfe6hkXExOjW2+9Vb169ZLNZtP27dv117/+VTt37tTSpUsVHR3doGVGRYX83zGnZLVavL42JcMwPF+r/99Y9YGcd0B6MX7+GvRe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qbS0tMb4kpISJSQkBKEjAABOLWwDOSUlRUVFRV7jSktLdejQIaWkpASpKwAAahe2gZyTk6OPP/5YdrvdM27VqlWyWCzq2bNnEDsDAKCmsD3LuqSkRNddd52Sk5M1cuRIz41Brr/+em4MAgAwnbANZKnq1pmPPvqo160zx48fz60zAQCmE9aBDABAqAjbY8gAAIQSAhkAABMgkAEAMAECGQAAEyCQAQAwAQIZAAATIJAj0MqVK/W73/1OOTk5ysjI0KBBg/S3v/2txvNEly5dqr59+yo9PV0DBw7UunXrgtRx+Fm/fr3uuOMO9ejRQxdffLGuuuoqzZw5s8b919euXauBAwcqPT1dffv21bJly4LUcfgrLy9XTk6O0tLS9O9//9trGttC4Lz11ltKS0ur8e/JJ5/0qouEdRC2j1/Eqb300ktq27atJk6cqJYtW+rjjz/WlClTdPDgQY0ePVqS9O6772rKlCkaNWqUevToocLCQo0ePVqvvfaaMjIygvsCwkBxcbG6dOmi/Px8JSYmateuXZo7d6527dqlF198UZL06aefavTo0br55ps1adIk/fOf/9RDDz2k+Ph4/frXvw7yKwg/zz33nJxOZ43xbAtNY8GCBWrRooVnuE2bNp7/R8w6cCPiHD58uMa4yZMnu7t16+Z2Op1ut9vtvvbaa93333+/V83gwYPdd999d5P0GIkWL17s7tSpk/vgwYNut9vt/u1vf+sePHiwV83999/v7tevXzDaC2vffvutOyMjw/3GG2+4O3Xq5N6xY4dnGttCYC1btszdqVOnWt+XqkXKOmCXdQRq1apVjXGdO3dWWVmZKioqtHfvXn333Xfq16+fV03//v21efNmnThxoqlajSiJiYmSpMrKSp04cUJbtmyp8Um4f//++u9//6t9+/YFocPw9dhjj2nIkCFKTk72Gs+2EHyRtA4IZEiS/vWvf6lNmzZq3ry557GVv3xzSk1NVWVlpfbu3RuMFsOS0+nU8ePH9cUXX+jZZ59Vbm6ukpKS9P3336uysrLGo0JTU1MlqcajRVF/q1at0jfffKPf//73NaaxLTSdAQMGqHPnzrrqqqv0/PPPew4fRNI64Bgy9Omnn6qwsFATJkyQVPWkLEmy2WxeddXD1dPRcFdeeaV++OEHSVLv3r315z//WRLroKkcPXpUs2bN0vjx49W8efMa01kPgde6dWuNGTNGXbt2lWEYWrt2rebMmaMffvhBU6dOjah1QCBHuIMHD2r8+PHKysrS0KFDg91OxHnhhRd09OhRffvtt5o3b55GjRqlhQsXBrutiDFv3jydeeaZuummm4LdSsTq3bu3evfu7Rnu1auXYmNj9fLLL2vUqFFB7Kzpscs6gtntdo0YMUKJiYmaO3euLJaqX4eEhARJqnEJjt1u95qOhrvggguUmZmpvLw8Pffcc9qyZYvef/991kET2L9/v1588UXdd999Ki0tld1uV0VFhSSpoqJC5eXlrIcg6devn5xOp7766quIWgcEcoQ6duyYRo4cqdLS0hqXG1Qft/zlccqioiJFR0erXbt2TdprpEhLS1N0dLS+//57tW/fXtHR0bWuA0k1ji3Df/v27VNlZaXuuecede/eXd27d/d8Ihs6dKjuuusutgUTiKR1wC7rCORwODRu3DgVFRXptdde87reT5LatWunjh07atWqVbr66qs94wsLC5Wdna2YmJimbjkibN++XZWVlUpKSlJMTIyysrL03nvvadiwYZ6awsJCpaamKikpKYidhofOnTvrlVde8Rr31VdfaebMmZo+fbrS09PZFoKksLBQVqtVF154oVq3bh0x64BAjkDTp0/XunXrNHHiRJWVlenzzz/3TLvwwgsVExOjMWPG6MEHH1T79u2VlZWlwsJC7dixQ6+++mrwGg8jo0eP1sUXX6y0tDSdccYZ+s9//qOCggKlpaV53nR+97vfaejQoZo2bZr69eunLVu2aMWKFZo9e3aQuw8PNptNWVlZtU676KKLdNFFF0kS20KADR8+XFlZWUpLS5MkffDBB1qyZImGDh2q1q1bS4qcdWC43b+4XyLCXm5urvbv31/rtA8++MDz6Wvp0qWaP3++Dhw4oOTkZN1///268sorm7LVsPXCCy+osLBQ33//vdxut9q2batrrrlGw4cP9zrb94MPPtCcOXO0e/dunXfeebrnnnt08803B7Hz8LZlyxYNHTpUf/vb35Senu4Zz7YQOI899pg++ugjHTx4UC6XSx07dlReXp7y8/NlGIanLhLWAYEMAIAJcFIXAAAmQCADAGACBDIAACZAIAMAYAIEMgAAJkAgAwBgAgQyAAAmQCAD8DJ37lylpaXpyJEjAV3OxIkTlZubG9BlAKGEQAYAwAS4lzWAoHj00UfFjQKBnxHIAIIiOjo62C0ApsIuawC1+umnnzR27Fh169ZNWVlZeuyxx3T8+HHP9LS0ND3yyCNauXKl+vfvry5dumjw4MH6+uuvJUlvvvmmrrnmGqWnpys/P1/79u3zmj/HkAFvfEIGUKtx48apbdu2euCBB/T5559r0aJFstvt+tOf/uSp+fTTT7V27VrddtttkqqeYjVq1Cjdfffdev3113XbbbeppKRECxYs0KRJk2o8fxjAzwhkALVKSkrSvHnzJEm33367mjdvrtdff12//e1vdcEFF0iSdu/erZUrV3oe2ZmQkKCpU6dq3rx5WrVqledRki6XS88//7z27dvnqQXgjV3WAGp1++23ew3fcccdkqQNGzZ4xmVnZ3sFbNeuXSVJ1157rddznbt06SJJ2rt3b8D6BUIdgQygVh06dPAabt++vSwWi9ex4HPPPderpjqEzznnHK/xLVq0kCTZ7fZAtAqEBQIZgE8Mw6gxzmq11lp7qvFc5gScGoEMoFZ79uypMexyuTgGDAQIgQygVq+99prX8KuvvipJysnJCUY7QNjjLGsAtdq3b59GjRql3r176/PPP9fy5cs1YMAAzxnWABoXn5AB1GrOnDmKiYnRn//8Z61fv1533HGHZsyYEey2gLBluDnLAgCAoOMTMgAAJkAgAwBgAgQyAAAmQCADAGACBDIAACZAIAMAYAIEMgAAJkAgAwBgAgQyAAAmQCADAGACBDIAACZAIAMAYAIEMgAAJvD/suEmlt6HUXQAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0wAqPe_vWJ23"
      },
      "source": [
        "Normal BMI Range --> 18.5 to 24.9"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 573
        },
        "id": "xNwyB3IzWAsU",
        "outputId": "34290bc5-3b1e-4a01-edec-fdc8410cc433"
      },
      "source": [
        "# children column\n",
        "plt.figure(figsize=(6,6))\n",
        "sns.countplot(x='children', data=insurance_dataset)\n",
        "plt.title('Children')\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 600x600 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjAAAAIsCAYAAADs5ZOPAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA4kUlEQVR4nO3deXSU5aHH8d9MQiAEJiEciLJJAiUCRhMPlxAT0gJWmoByq4KABVoporKmegvmsioFalF2UBYXwAVcWrXEFKSUSJrSq4IWQVkSIMRCtJCZhIRmmbl/cJjjdAImk5DJQ76fczw67/vMO8+8DeTbd56ZsbhcLpcAAAAMYvX3BAAAAGqLgAEAAMYhYAAAgHEIGAAAYBwCBgAAGIeAAQAAxiFgAACAcQgYAABgHAIGAAAYh4AB0GhFR0frqaee+t5x77zzjqKjo3X69Gn3tjFjxmjMmDHfe999+/YpOjpa+/btq9NcATSsQH9PAEDTdOrUKW3YsEHZ2dkqLCxUs2bN1KNHD6WkpOiBBx5QixYt/D1FAI0YAQOgwf3lL3/RtGnTFBQUpGHDhqlHjx6qqKjQJ598ot/97nc6duyYnn766Rofb9iwYRoyZIiCgoKu4awBNCYEDIAGlZ+fr7S0NHXo0EGvvPKK2rdv79734IMP6uTJk/rLX/5Sq2MGBAQoICCgnmfqqaysTMHBwdf0MQDUHGtgADSoDRs2qLS0VL/5zW884uWym266SePGjfPY9uGHH2ro0KG65ZZbNGTIEGVlZXnsr24NTHXOnDmjxx57TLGxsUpISNDChQtVXl7uNW7MmDEaOnSoDh48qAcffFC33XabnnvuOUlSeXm5VqxYoR//+Me65ZZb9MMf/lDPPPOM13Eur9/5vrkD8A1XYAA0qN27d6tz5866/fbbazT+k08+0Y4dOzR69GiFhIRo8+bNmjp1qnbv3q02bdrU+HEvXryocePG6Z///KfGjBmj9u3b691339Xf/va3ascXFRVpwoQJGjJkiO655x61bdtWTqdTjz76qD755BONGDFC3bp105EjR/TKK6/oxIkTWrNmzTWZOwBvBAyABlNSUqKzZ89q0KBBNb7P8ePHlZGRoS5dukiS4uPjNWzYMG3fvl0/+9nPanycrVu36sSJE1q2bJlSUlIkSSNGjNCwYcOqHf/NN99o/vz5GjlypHvbu+++q7/+9a/avHmz+vTp497+gx/8QHPnztWnn37qEWb1NXcA3ngJCUCDKSkpkSSFhITU+D533HGHOwAk6eabb1arVq2Un59fq8fOyspSu3bt9JOf/MS9LTg4WCNGjKh2fFBQkO69916PbZmZmerWrZuioqJ07tw59z/9+vWTJK+3YtfX3AF44woMgAbTqlUrSdKFCxdqfJ8bb7zRa1toaKgcDketHrugoEA33XSTLBaLx/bIyMhqx0dERHi9q+nkyZM6fvy4EhISqr3Pv/71L4/b9TV3AN4IGAANplWrVmrfvr2OHj1a4/tc6d1FLpervqZVreo+h8bpdKpHjx568sknq73PDTfc4HHbX3MHmgICBkCDGjBggLZu3ar9+/crLi6uwR63Y8eOOnLkiFwul8dVmLy8vBofo0uXLvryyy+VkJDgdSUHQMNiDQyABvXLX/5SLVu21KxZs/Ttt9967T916pReeeWVen/c5ORkFRYWKjMz072trKxM27Ztq/ExUlJSdPbs2Wrvc/HiRZWWltbLXAF8P67AAGhQXbp00ZIlS5SWlqbU1FT3J/GWl5dr//79yszM9Fo8Wx9GjBihV199VTNmzNAXX3yhdu3a6d13363VVxYMGzZMH3zwgebOnat9+/bp9ttvV1VVlXJzc5WZmakNGzYoJiam3ucOwBsBA6DBDRo0SO+99542btyoXbt26fXXX1dQUJCio6M1c+bMK74zqC6Cg4P18ssv6+mnn9aWLVvUokUL3X333UpOTtYvf/nLGh3DarVq9erVevnll/Xuu+9q586dCg4OVqdOnTRmzJgrLggGUP8sLlaTAQAAw7AGBgAAGIeAAQAAxiFgAACAcQgYAABgHAIGAAAYh4ABAADGIWAAAIBx+CC7a8Dlcsnp5ON1AACoDavVUuPvGSNgrgGn06Vz5y74exoAABglPDxEAQE1CxheQgIAAMYhYAAAgHEIGAAAYBwCBgAAGIeAAQAAxiFgAACAcQgYAABgHAIGAAAYh4ABAADGaVQBc/LkSc2ZM0fDhg1Tr169NHToUI/9JSUlWrlype6//3716dNHd9xxhx555BF99dVXXscqLi5Wenq6+vbtq7i4OE2dOlWFhYVe4z799FM98MADuvXWWzVgwACtW7dOLhdfAwAAQGPWqALm6NGj2rNnj2666SZ169bNa//XX3+trVu3KjEx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zp079f7779fnfAAAAL6Xz2tgpk+frsrKSv3617/WvHnzdMMNN8hq9ewhi8Wi9957r86TBAAA+C6fAyYsLExhYWG66aab6nM+AAAA38vngNm8eXN9zgMAAKDGmt5X+wIAAOP5fAXm//7v/2o07r/+6798fQgAAIBq+RwwY8aMkcVi+d5xhw8f9vUhAAAAqlWnT+L9T1VVVSooKNC2bdvkdDr1+OOP12lyAAAA1bkm30Z97733avTo0fr73/+uhIQEXx8CAACgWtdkEa/VatWQIUP05ptvXovDAwCAJu6avQvJbreruLj4Wh0eAAA0YT6/hPT1119Xu93hcOjjjz/Wxo0b1adPH58n9n1+//vf65VXXtHx48fVsmVLxcTEaNWqVWrRooUk6c9//rOWLVumvLw8dejQQQ8//LDuu+8+j2OUl5dr6dKleu+993ThwgXFxcVp9uzZioqKumbzBgAAdedzwAwcOPCK70JyuVyKjY3V/PnzfZ7Y1axdu1br16/XI488otjYWJ0/f145OTmqqqqSJH388ceaPHmy7r//fqWnp+tvf/ub/vd//1chISH6yU9+4j7OggULlJGRoZkzZyoiIkLPP/+8fv7zn2v79u1q3br1NZk7AACoO58DZuHChV4BY7FYZLPZ1KVLF3Xv3r3Ok6tObm6uVq1apTVr1uiHP/yhe/vgwYPd/7127VrdeuuteuqppyRJ/fr1U35+vlasWOEOmDNnzuitt97S3Llzdf/990u69A3bAwYM0BtvvKEJEyZck/kDAIC68zlg7r333vqcR42988476tSpk0e8fFd5ebn27dunJ554wmN7amqq/vjHP+r06dPq1KmT9u7dK6fT6XFFJiwsTImJicrKyiJgAABoxOplEe+xY8e0Z88e7dmzR8eOHauPQ17RZ599ph49emjNmjVKSEjQLbfcopEjR+qzzz6TJJ06dUoVFRVe61i6desm6dIVnMv/btu2rUJDQ73GXR4DAAAaJ5+vwEjShx9+qMWLF6ugoMBje6dOnTRz5kwNGjSoTpOrzjfffKODBw/qyJEjmjt3roKDg/X888/roYce0o4dO2S32yVJNpvN436Xb1/e73A4ql3nYrPZ3GPqIjCw+jYMCGjaXz/l6/PnvDXt5w8A/8nngNmzZ4+mTp2qDh06KC0tzX2F4/jx49q2bZumTJmi559/XsnJ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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 303
        },
        "id": "4TMelPK-Wx5x",
        "outputId": "054029f4-8901-4552-b741-b6d258b4a6b5"
      },
      "source": [
        "insurance_dataset['children'].value_counts()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "children\n",
              "0    1186\n",
              "1     672\n",
              "2     496\n",
              "3     324\n",
              "4      52\n",
              "5      42\n",
              "Name: count, dtype: int64"
            ],
            "text/html": [
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>count</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>children</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1186</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>672</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>496</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>324</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>52</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>42</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table><br><label><b>dtype:</b> int64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 573
        },
        "id": "poIPFFSBW9n4",
        "outputId": "ddcad76a-b205-4e2e-8ed4-30bb55ff832b"
      },
      "source": [
        "# smoker column\n",
        "plt.figure(figsize=(6,6))\n",
        "sns.countplot(x='smoker', data=insurance_dataset)\n",
        "plt.title('smoker')\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 600x600 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjAAAAIsCAYAAADs5ZOPAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAxqUlEQVR4nO3de1RU9eL//9cMiKEyCH5E/aokYOAlFLRUloQlXhIozY9WCuSnzNNVw7SjsspLp6Oeyi6WHRPplGmlRcuyiDRPqamni2JWZpqgR02xMmdQLISZ3x8u59cElI7g8JbnYy1Xzt7vvee9EfTZ3ntmLC6XyyUAAACDWH09AQAAgHNFwAAAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAgKQ333xTMTEx+vLLL309FQBngYABAADGIWAAAIBxCBgAuACcTqd+/fVXX08DuGgQMAB86vjx4/r73/+u/v376/LLL1dCQoJuvfVWff3115KkzMxMpaWlaefOncrIyFD37t01cOBAFRQUSJI+/fRTjRw5Ut26ddPgwYO1adOmKs+xY8cO3X777erRo4fi4+M1ZswYbdu27U/nZrfbNWLECCUlJamoqEiSVF5ervnz52vgwIG6/PLL1a9fPz366KMqLy/32DYmJkYPP/yw3n77baWmpio2NlYbNmw4z68WgDP8fT0BAA3bjBkz9P777ysjI0NRUVE6duyYtmzZoj179qhr166STofEnXfeqZSUFF177bV69dVXdf/998vpdGr27Nm6+eablZaWptzcXE2YMEEfffSRmjVrJknavXu30tPT1bRpU91+++3y9/fX8uXLlZmZqaVLl6p79+7Vzuvo0aO67bbbZLfbtXTpUoWHh8vpdOquu+7Sli1bdOONNyoqKkq7du3SSy+9pL179+q5557z2Md//vMfvffee0pPT1dISIjatm1bt19MoCFxAYAP9ezZ0zVr1qwa12dkZLiio6Ndq1atci/bs2ePKzo62tWpUyfXtm3b3Ms3bNjgio6OduXl5bmX3X333a6uXbu6/vvf/7qXlZSUuOLj413p6enuZXl5ea7o6GjX9u3bXUeOHHGlpqa6kpOTXQcOHHCPWblypatTp06uzz77zGOOr776qis6Otq1ZcsW97Iz89u9e/c5fkUAnA0uIQHwKZvNpi+++EIlJSU1jmnSpIlSU1PdjyMjI2Wz2RQVFeVxBuXM7/fv3y9Jqqys1MaNGzVgwAC1b9/ePS4sLExpaWnasmWLjh8/7vFcJSUlysjI0KlTp7Rs2TKPsyYFBQWKiopSZGSkjh496v7Vp08fSdInn3zisa8rr7xSHTt2PNcvCYCzwCUkAD41efJkTZ06VVdffbW6du2qfv36adiwYR7B0bp1a1ksFo/tgoKC1Lp16yrLJMnhcEg6fRno5MmTioiIqPK8UVFRcjqdOnTokC677DL38gceeED+/v7Kz89Xy5YtPbbZt2+f9uzZo4SEhGqP5aeffvJ43K5duz87fABeImAA+FRKSoquuOIKrVmzRhs3blRubq5ycnL0zDPPqF+/fpIkPz+/aretabnL5fJ6PoMGDdLKlSu1ZMkSTZo0yWOd0+lUdHS0pk2bVu22vw+qSy65xOt5APhjBAwAnwsLC1N6errS09P1008/6YYbbtDChQvdAeOt0NBQBQYGqri4uMq6oqIiWa1WtWnTxmN5RkaGwsPDNX/+fAUFBekvf/mLe114eLh27typhISEKmeEAFxY3AMDwGcqKytVWlrqsaxFixYKCwur8rJkb/j5+alv375au3atDhw44F7+448/6p133lHPnj3dr1b6rXvuuUe33Xab5s2bp1deecW9fMiQISopKdGKFSuqbPPLL7+orKzsvOcM4OxwBgaAz5w4cUL9+vXT4MGD1alTJzVp0kSbNm3Sl19+qalTp9bKc2RlZWnTpk0aPXq0Ro8eLT8/Py1fvlzl5eV64IEHatxuypQpOn78uB5++GE1bdpUQ4cO1dChQ/Xee+9pxowZ+uSTT9SjRw9VVlaqqKhIBQUFWrx4sWJjY2tl3gD+GAEDwGcuueQSjRo1Shs3btTq1avlcrkUHh6uGTNmaPTo0bXyHJdddpmWLVumefPm6fnnn5fL5VK3bt302GOP1fgeMGfMmjVLZWVlys7OVtOmTTVgwAAtWLBAL774ot566y2tWbNGgYGBateunTIzM6u9WRhA3bC4zuduNwAAAB/gHhgAAGAcAgYAABiHgAEAAMYhYAAAgHEIGAAAYBwCBgAAGIeAAQAAxuGN7OqAy+WS08nb6wAAcC6sVstZf84YAVMHnE6Xjh494etpAABglNDQpvLzO7uA4RISAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOP4+3oCAHAxsVotslotvp4GUKecTpecTpdP50DAAEAtsVotat68ifz8OLmNi1tlpVPHjpX5NGIIGACoJVarRX5+Vi14daMOHrH7ejpAnWgbFqx7RvWV1WohYADgYnLwiF17D/7s62kAFzXOcwIAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADj1KuAee+993TXXXcpKSlJcXFxGjp0qN544w25XC6Pca+//roGDx6s2NhYXX/99frwww+r7Ku0tFTZ2dnq1auX4uPjNWHCBB05cqTKuK1bt+qmm25St27ddM0112jRokVVng8AANQv9SpgXnzxRQUGBmrq1Kn65z//qaSkJD300ENasGCBe8y7776rhx56SEOGDFFOTo7i4uJ07733atu2bR77ysrK0saNGzVz5kw9/vjjKi4u1rhx41RRUeEes2/fPo0dO1YtW7bU888/rzFjxmj+/Pl64YUXLtQhAwAAL/j7egK/9c9//lOhoaHuxwkJCTp27Jj+9a9/6e6775bVatX8+fOVmpqqrKwsSVKfPn20a9cuLViwQDk5OZKkwsJCffzxx8rNzVViYqIkKSIiQikpKVq9erVSUlIkSbm5uQoJCdETTzyhgIAAJSQk6OjRo1q4cKEyMzMVEBBwYb8AAADgrNSrMzC/jZczOnfurOPHj6usrEz79+/X3r17NWTIEI8xKSkp2rx5s8rLyyVJ69evl81mU9++fd1jIiMj1blzZ61fv969bP369UpOTvYIlZSUFDkcDhUWFtb24QEAgFpSrwKmOlu2bFGrVq3UrFkzFRUVSTp9NuW3oqKidOrUKe3fv1+SVFRUpIiICFksFo9xkZGR7n2UlZXp0KFDioyMrDLGYrG4xwEAgPqnXl1C+r3PP/9c+fn5mjJliiTJbrdLkmw2m8e4M4/PrHc4HAoKCqqyv+DgYH311VeSTt/kW92+AgICFBgY6N6Xt/z9630bAqhlfn783KPh8PX3e70NmMOHD2vixInq3bu3brnlFl9P55xYrRaFhDT19TQAAKgzNlugT5+/XgaMw+HQuHHj1Lx5cz3zzDOyWk9XXnBwsKTTZ09atmzpMf636202mw4fPlxlv3a73T3mzBmaM2dizigvL9fJkyfd47zhdLrkcJR5vT0AM/n5WX3+lzpwoTgcJ1VZ6azVfdpsgWd9ZqfeBcwvv/yiO+64Q6WlpVq+fLnHpaAz96sUFRV53LtSVFSkRo0aqX379u5xmzdvlsvl8rgPpri4WNHR0ZKkJk2aqE2bNlXudSkuLpbL5apyb8y5qqio3T9UAADqk8pKp0//ratXF2wrKiqUlZWloqIiLV68WK1atfJY3759e3Xo0EEFBQUey/Pz85WQkOB+NVFSUpLsdrs2b97sHlNcXKwdO3YoKSnJvSwpKUlr167VqVOnPPZls9kUHx9fF4cIAABqQb06AzNr1ix9+OGHmjp1qo4fP+7x5nRdunRRQECAxo8fr8mTJys8PFy9e/dWfn6+tm/frqVLl7rHxsfHKzExUdnZ2ZoyZYoaN26sJ598UjExMRo0aJB73NixY7Vq1SpNmjRJo0aN0q5du5Sbm6uJEyfyHjAAANRjFlc9et/8/v376+DBg9WuW7t2rdq1ayfp9EcJ5OTk6Pvvv1dERITuv/9+XXPNNR7jS0tLNWfOHK1Zs0YVFRVKTEzUgw8+WOWsztatWzV37lx98803Cg0NVXp6usaNG1flJdjnorLSqaNHT3i9PQAz+ftbFRLSVNlP52vvwZ99PR2gTnRoG6LZ96Xo559P1PolpNDQpmd9D0y9CpiLBQEDNEwEDBqC+hIw9eoeGAAAgLNBwAAAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOPUq4DZt2+fpk+frqFDh6pLly5KS0urMiYzM1MxMTFVfu3Zs8djXGlpqbKzs9WrVy/Fx8drwoQJOnLkSJX9bd26VTfddJO6deuma665RosWLZLL5aqzYwQAAOfP39cT+K3du3dr3bp16t69u5xOZ40h0aNHD02ZMsVjWbt27TweZ2Vl6bvvvtPMmTPVuHFjPfXUUxo3bpzy8vLk73/6sPft26exY8eqb9++ysrK0rfffqvHH39cfn5+Gjt2bN0cJAAAOG/1KmD69++vAQMGSJKmTp2qr776qtpxNptNcXFxNe6nsLBQH3/8sXJzc5WYmChJioiIUEpKilavXq2UlBRJUm5urkJCQvTEE08oICBACQkJOnr0qBYuXKjMzEwFBATU7gECAIBaUa8uIVmttTOd9evXy2azqW/fvu5lkZGR6ty5s9avX+8xLjk52SNUUlJS5HA4VFhYWCtzAQAAta9eBczZ+vTTTxUXF6fY2FhlZGTos88+81hfVFSkiIgIWSwWj+WRkZEqKiqSJJWVlenQoUOKjIysMsZisbjHAQCA+qdeXUI6G1deeaWGDh2qDh066MiRI8rNzdWtt96ql19+WfHx8ZIkh8OhoKCgKtsGBwe7L0uVlpZKOn056rcCAgIUGBgou91+XvP09zeyDQGcBz8/fu7RcPj6+924gJkwYYLH46uvvlppaWl67rnnlJOT46NZebJaLQoJaerraQAAUGdstkCfPr9xAfN7TZo0Ub9+/fT++++7l9lsNh0+fLjKWLvdruDgYElyn6E5cybmjPLycp08edI9zhtOp0sOR5nX2wMwk5+f1ed/qQMXisNxUpWVzlrdp80WeNZndowPmOpERkZq8+bNcrlcHvfBFBcXKzo6WtLp8GnTpk2Ve12Ki4vlcrmq3BtzrioqavcPFQCA+qSy0unTf+uMv2BbVlamjz76SLGxse5lSUlJstvt2rx5s3tZcXGxduzYoaSkJI9xa9eu1alTp9zL8vPzZbPZ3PfTAACA+qdenYE5efKk1q1bJ0k6ePCgjh8/roKCAklSr169VFRUpMWLF2vgwIFq27atjhw5on/961/64Ycf9PTTT7v3Ex8fr8TERGVnZ2vKlClq3LixnnzyScXExGjQoEHucWPHjtWqVas0adIkjRo1Srt27VJubq4mTpzIe8AAAFCPWVz16H3zDxw4oOTk5GrXLVmyRK1bt9bDDz+sb7/9VseOHVNgYKDi4+N17733qlu3bh7jS0tLNWfOHK1Zs0YVFRVKTEzUgw8+qFatWnmM27p1q+bOnatvvvlGoaGhSk9P17hx46q8BPtcVFY6dfToCa+3B2Amf3+rQkKaKvvpfO09+LOvpwPUiQ5tQzT7vhT9/POJWr+EFBra9KzvgalXAXOxIGCAhomAQUNQXwLG+HtgAABAw0PAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwjtcBs3LlSh04cKDG9QcOHNDKlSu93T0AAECNvA6YadOmqbCwsMb127dv17Rp07zdPQAAQI28DhiXy/WH68vKyuTn5+ft7gEAAGrkfy6Dd+7cqZ07d7off/7556qsrKwyzuFw6LXXXlNERMT5zxAAAOB3zilgPvjgAz377LOSJIvFouXLl2v58uXVjrXZbPrHP/5x/jMEAAD4nXMKmBtvvFFXX321XC6XRo4cqQkTJigpKcljjMViUWBgoMLDw+Xvf067BwAAOCvnVBhhYWEKCwuTJC1ZskRRUVFq0aJFnUwMAACgJl6fIunVq1dtzgMAAOCsndc1ng0bNuiNN97Q/v375XA4qrwyyWKx6IMPPjivCQIAAPye1wGzePFizZs3Ty1atFC3bt0UExNTm/MCAACokdcBs2TJEvXp00eLFi1So0aNanNOAAAAf8jrN7JzOBwaPHgw8QIAAC44rwMmNjZWxcXFtTkXAACAs+J1wMycOVNr1qzRqlWranM+AAAAf8rre2CysrJUUVGhv/71r5o5c6Zat24tq9WzhywWi95+++3zniQAAMBveR0wzZs3V/PmzXXppZfW5nwAAAD+lNcB8/LLL9fmPAAAAM6a1/fAAAAA+IrXZ2A+++yzsxp35ZVXevsUAAAA1fI6YDIzM2WxWP503DfffOPtUwAAAFTrvN6J9/cqKyt18OBBrVixQk6nU5MmTTqvyQEAAFSnTj6Nevjw4Ro9erQ+/fRTJSQkePsUAAAA1aqTm3itVqtSU1P1+uuv18XuAQBAA1dnr0Ky2+0qLS2tq90DAIAGzOtLSN9//321yx0Ohz7//HPl5ubqiiuu8HpiAAAANfE6YPr371/jq5BcLpfi4uI0a9YsrycGAABQE68DZvbs2VUCxmKxyGazKTw8XB07djzvyQEAAFTH64AZPnx4bc4DAADgrHkdML/13Xff6eDBg5Kktm3bcvYFAADUqfMKmA8++EBz5851x8sZ7dq109SpU5WcnHxekwMAAKiO1wGzbt06TZgwQf/v//0/TZw4UVFRUZKkPXv2aMWKFRo/frwWLlyopKSkWpssAACAdB4B89xzzykmJkbLli1TkyZN3MuTk5OVkZGh0aNHa8GCBQQMAACodV6/kd23336rYcOGecTLGU2aNNENN9ygb7/99rwmBwAAUB2vA6Zx48ay2+01rrfb7WrcuLG3uwcAAKiR1wHTu3dvLVmyRIWFhVXWffHFF3r55Zf5IEcAAFAnvL4H5oEHHtDNN9+s0aNHq1u3boqIiJAkFRcXa/v27WrRooUmT55caxMFAAA4w+szMO3bt9fbb7+tzMxM2e125efnKz8/X3a7XbfccoveeusttWvXrjbnCgAAIOk8zsBUVFSocePGys7OVnZ2dpX1x48fV0VFhfz9a+W98gAAANy8PgPzyCOP6Oabb65x/ahRozR37lxvdw8AAFAjrwNmw4YNGjx4cI3rBw8erPXr13u7ewAAgBp5HTBHjhxRq1atalwfFhamkpISb3cPAABQI68Dpnnz5iouLq5x/Z49e9SsWTNvdw8AAFAjrwPmqquu0muvvaYdO3ZUWff1119rxYoVfIwAAACoE16/ROi+++7Thg0bNHLkSPXv318dO3aUJO3evVsffvihQkNDdd9999XaRAEAAM7wOmBatWqlvLw8zZs3T2vXrtWaNWskSc2aNdN1112niRMn/uE9MgAAAN46rzdpCQsL0z/+8Q+5XC4dPXpUkhQaGiqLxVIrkwMAAKhOrbzLnMViUYsWLWpjVwAAAH/K65t468K+ffs0ffp0DR06VF26dFFaWlq1415//XUNHjxYsbGxuv766/Xhhx9WGVNaWqrs7Gz16tVL8fHxmjBhgo4cOVJl3NatW3XTTTepW7duuuaaa7Ro0SK5XK5aPzYAAFB76lXA7N69W+vWrdOll16qqKioase8++67euihhzRkyBDl5OQoLi5O9957r7Zt2+YxLisrSxs3btTMmTP1+OOPq7i4WOPGjVNFRYV7zL59+zR27Fi1bNlSzz//vMaMGaP58+frhRdeqMvDBAAA56lefVBR//79NWDAAEnS1KlT9dVXX1UZM3/+fKWmpiorK0uS1KdPH+3atUsLFixQTk6OJKmwsFAff/yxcnNzlZiYKEmKiIhQSkqKVq9erZSUFElSbm6uQkJC9MQTTyggIEAJCQk6evSoFi5cqMzMTAUEBFyAowYAAOeqXp2BsVr/eDr79+/X3r17NWTIEI/lKSkp2rx5s8rLyyVJ69evl81mU9++fd1jIiMj1blzZ4+PN1i/fr2Sk5M9QiUlJUUOh0OFhYW1cUgAAKAO1KuA+TNFRUWSTp9N+a2oqCidOnVK+/fvd4+LiIio8mqoyMhI9z7Kysp06NAhRUZGVhljsVjc4wAAQP1Try4h/Rm73S5JstlsHsvPPD6z3uFwKCgoqMr2wcHB7stSpaWl1e4rICBAgYGB7n15y9/fqDYEUAv8/Pi5R8Ph6+93owLGFFarRSEhTX09DQAA6ozNFujT5zcqYIKDgyWdPnvSsmVL93KHw+Gx3maz6fDhw1W2t9vt7jFnztCcORNzRnl5uU6ePOke5w2n0yWHo8zr7QGYyc/P6vO/1IELxeE4qcpKZ63u02YLPOszO0YFzJn7VYqKijzuXSkqKlKjRo3Uvn1797jNmzfL5XJ53AdTXFys6OhoSVKTJk3Upk2bKve6FBcXy+VyVbk35lxVVNTuHyoAAPVJZaXTp//WGXXBtn379urQoYMKCgo8lufn5yshIcH9aqKkpCTZ7XZt3rzZPaa4uFg7duzw+ITspKQkrV27VqdOnfLYl81mU3x8fB0fDQAA8Fa9OgNz8uRJrVu3TpJ08OBBHT9+3B0rvXr1UmhoqMaPH6/JkycrPDxcvXv3Vn5+vrZv366lS5e69xMfH6/ExERlZ2drypQpaty4sZ588knFxMRo0KBB7nFjx47VqlWrNGnSJI0aNUq7du1Sbm6uJk6cyHvAAABQj1lc9eh98w8cOKDk5ORq1y1ZskS9e/eWdPqjBHJycvT9998rIiJC999/v6655hqP8aWlpZozZ47WrFmjiooKJSYm6sEHH6zyCdlbt27V3Llz9c033yg0NFTp6ekaN27ceX0gZWWlU0ePnvB6ewBm8ve3KiSkqbKfztfegz/7ejpAnejQNkSz70vRzz+fqPVLSKGhTc/6Hph6FTAXCwIGaJgIGDQE9SVgjLoHBgAAQCJgAACAgQgYAABgHAIGAAAYh4ABAADGIWAAAIBxCBgAAGAcAgYAABiHgAEAAMYhYAAAgHEIGAAAYBwCBgAAGIeAAQAAxiFgAACAcQgYAABgHAIGAAAYh4ABAADGIWAAAIBxCBgAAGAcAgYAABiHgAEAAMYhYAAAgHEIGAAAYBwCBgAAGIeAAQAAxiFgAACAcQgYAABgHAIGAAAYh4ABAADGIWAAAIBxCBgAAGAcAgYAABiHgAEAAMYhYAAAgHEIGAAAYBwCBgAAGIeAAQAAxiFgAACAcQgYAABgHAIGAAAYh4ABAADGIWAAAIBxCBgAAGAcAgYAABiHgAEAAMYhYAAAgHEIGAAAYBwCBgAAGIeAAQAAxiFgAACAcQgYAABgHAIGAAAYh4ABAADGIWAAAIBxCBgAAGAcAgYAABiHgAEAAMYhYAAAgHEIGAAAYBwCBgAAGIeAAQAAxiFgAACAcQgYAABgHAIGAAAYh4ABAADGIWAAAIBxCBgAAGAcAgYAABiHgAEAAMYhYAAAgHEIGAAAYBwCBgAAGIeAAQAAxiFgAACAcQgYAABgHAIGAAAYh4ABAADGIWAAAIBxCBgAAGAcf19PAOfOarXIarX4ehpAnXI6XXI6Xb6eBoB6ioAxjNVqUfPmTeTnx8kzXNwqK506dqyMiAFQLQLGMFarRX5+Vi14daMOHrH7ejpAnWgbFqx7RvWV1WohYABUi4Ax1MEjdu09+LOvpwEAgE9wHQIAABiHgAEAAMYhYAAAgHEIGAAAYBzjAubNN99UTExMlV+PP/64x7jXX39dgwcPVmxsrK6//np9+OGHVfZVWlqq7Oxs9erVS/Hx8ZowYYKOHDlyoQ4FAAB4ydhXIS1evFhBQUHux61atXL//t1339VDDz2kO++8U3369FF+fr7uvfdeLVu2THFxce5xWVlZ+u677zRz5kw1btxYTz31lMaNG6e8vDz5+xv7pQEA4KJn7L/SXbt2VWhoaLXr5s+fr9TUVGVlZUmS+vTpo127dmnBggXKycmRJBUWFurjjz9Wbm6uEhMTJUkRERFKSUnR6tWrlZKSckGOAwAAnDvjLiH9mf3792vv3r0aMmSIx/KUlBRt3rxZ5eXlkqT169fLZrOpb9++7jGRkZHq3Lmz1q9ff0HnDAAAzo2xAZOWlqbOnTsrOTlZzz//vCorKyVJRUVFkk6fTfmtqKgonTp1Svv373ePi4iIkMXi+ZlCkZGR7n0AAID6ybhLSC1bttT48ePVvXt3WSwW/fvf/9ZTTz2lkpISTZ8+XXb76bfXt9lsHtudeXxmvcPh8LiH5ozg4GB99dVX5z1Pf/+6aUM+AwkNiWnf76bNFzgfvv5+Ny5grrrqKl111VXux4mJiWrcuLFeeukl3XnnnT6c2f/ParUoJKSpr6cBGM9mC/T1FADUwNc/n8YFTHWGDBmiF154Qd98842Cg4MlnX6JdMuWLd1jHA6HJLnX22w2HT58uMq+7Ha7e4y3nE6XHI6y89pHTfz8rD7/pgEuFIfjpCornb6exlnj5xMNSV38fNpsgWd9ZueiCJjfioyMlHT6Hpczvz/zuFGjRmrfvr173ObNm+VyuTzugykuLlZ0dPR5z6Oiwpy/dIH6qrLSyc8SUE/5+ufzorhgm5+fLz8/P3Xp0kXt27dXhw4dVFBQUGVMQkKCAgICJElJSUmy2+3avHmze0xxcbF27NihpKSkCzp/AABwbow7AzN27Fj17t1bMTExkqS1a9dqxYoVuuWWW9yXjMaPH6/JkycrPDxcvXv3Vn5+vrZv366lS5e69xMfH6/ExERlZ2drypQpaty4sZ588knFxMRo0KBBPjk2AABwdowLmIiICOXl5enw4cNyOp3q0KGDsrOzlZmZ6R6TlpamkydPKicnR4sWLVJERISeffZZxcfHe+zrqaee0pw5czR9+nRVVFQoMTFRDz74IO/CCwBAPWfcv9QPPvjgWY0bOXKkRo4c+YdjgoKCNHv2bM2ePbs2pgYAAC6Qi+IeGAAA0LAQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOMQMAAAwDgEDAAAMA4BAwAAjEPAAAAA4xAwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOMQMAAAwDgNPmD27NmjW2+9VXFxcerbt68effRRlZeX+3paAADgD/j7egK+ZLfbNWbMGHXo0EHPPPOMSkpKNHfuXP3yyy+aPn26r6cHAABq0KAD5rXXXtOJEyf07LPPqnnz5pKkyspKzZo1S3fccYdatWrl2wkCAIBqNehLSOvXr1dCQoI7XiRpyJAhcjqd2rhxo+8mBgAA/lCDDpiioiJFRkZ6LLPZbGrZsqWKiop8NCsAAPBnGvQlJIfDIZvNVmV5cHCw7Ha71/u1Wi0KDW16PlOrkcVy+r9TxvZXZaWzTp4D8DU/v9P/bxUcHCiXy8eTOQf8fKIhqMufT6vVctZjG3TA1BWLxSI/v7P/Q/BGcLNL6nT/QH1gtZp5kpifTzQEvv75NPNvh1pis9lUWlpaZbndbldwcLAPZgQAAM5Ggw6YyMjIKve6lJaW6ocffqhybwwAAKg/GnTAJCUladOmTXI4HO5lBQUFslqt6tu3rw9nBgAA/ojF5TLpFrnaZbfblZqaqoiICN1xxx3uN7K77rrreCM7AADqsQYdMNLpjxL429/+psLCQjVt2lRDhw7VxIkTFRAQ4OupAQCAGjT4gAEAAOZp0PfAAAAAMxEwAADAOAQMAAAwDgEDAACMQ8AAAADjEDAAAMA4BAwAADAOAQMAAIxDwAAAAOMQMAAAwDgEDBqcf//734qJidHevXs9ltvtdnXr1k3Lli2TJBUWFuqWW25RXFycevbsqUmTJumnn37y2GbRokUaOHCgYmNj1adPH/3f//2f9u/ff6EOBWgQpk6dqrS0NH3yyScaNmyY4uLiNGLECH311VfuMb/++qvmzJmjxMRExcbGaujQoVqzZo0PZ426RsCgwenXr59atWqlvLw8j+XvvPOOJOm6665TYWGhMjMzFRQUpCeffFJ/+9vf9OWXX+ruu+92j1+5cqWefvppjRgxQosXL9Yjjzyizp0768SJExf0eICG4IcfftAjjzyisWPH6qmnntKvv/6qe++9V6dOnZIkTZ48WcuXL9ftt9+uBQsWqGPHjho/frzWrl3r45mjrvj7egLAhebn56fhw4crLy9PWVlZ8vPzkyTl5eVp4MCBstlsmjdvni6//HI9++yzslgskqTo6GilpaVp3bp16tevn7Zv366YmBjdcccd7n0PGDDAJ8cEXOzsdruWLl2qyy67TJIUGBioW265RV988YWaNWum1atXa9asWbr55pslSUlJSTp48KAWLFig5ORkX04ddYQzMGiQRowYoR9++EEbNmyQJO3cuVNff/21RowYoZMnT2rr1q269tprVVlZqYqKClVUVKhDhw5q06aNvvzyS0lSly5dtGPHDs2ZM0eff/65+/8EAdS+sLAwd7xIUseOHSVJJSUl2rJliyTp2muv9dhmyJAh2rFjh8rKyi7cRHHBcAYGDVK7du3Ut29fvfHGG7r66quVl5endu3aqU+fPjpy5IgqKys1Z84czZkzp8q2hw4dkiQNHz5cJ06c0IoVK/Tiiy8qKChIw4YN0+TJk3XJJZdc6EMCLmo2m83jcaNGjSSdvvfFbrerUaNGat68uceY//mf/5HL5VJpaamaNGlyoaaKC4SAQYM1cuRITZ48WSUlJVq1apUyMzNlsVgUFBQki8WiO+64o9pLQiEhIZIkq9WqMWPGaMyYMSopKdG7776refPmKSQkRPfcc8+FPhygwQoODtapU6dkt9sVHBzsXv7jjz+6f6Zx8SFg0GAlJyfLZrNp0qRJstvtGj58uCSpSZMmiouLU1FRkWJjY89qX61atdJtt92md955R0VFRXU5bQC/07NnT0lSQUGBbrrpJvfygoICdenShbMvFykCBg1Wo0aNNGzYMOXm5ioxMVFt2rRxr/vrX/+qMWPGKCsrS6mpqbLZbDp8+LA2bdqk4cOHq3fv3po+fbpsNpvi4uJks9m0detW7dy5U6NGjfLhUQENT6dOnTRo0CDNnTtXv/zyiyIiIvT222+rsLBQzz33nK+nhzpCwKBBGzhwoHJzc/W///u/Hst79OihV155Rc8884ymTZumU6dOqXXr1urTp48uvfRSSVJ8fLxWrFih119/XSdPnlT79u01bdo0jRw50heHAjRojz32mJ544gnl5OTo2LFjioyM1Pz589W/f39fTw11xOJyuVy+ngTgK08//bReeeUVbdiwQQEBAb6eDgDgLHEGBg1SUVGRiouLtXTpUo0ePZp4AQDDcAYGDVJmZqa2bdumq666So8//jg3+QGAYQgYAABgHN6JFwAAGIeAAQAAxiFgAACAcQgYAABgHAIGACS9+eabiomJcX/aOID6jYABAADGIWAAAIBxCBgAuACcTqd+/fVXX08DuGgQMAB86vjx4/r73/+u/v376/LLL1dCQoJuvfVWff3115JOv2tyWlqadu7cqYyMDHXv3l0DBw5UQUGBJOnTTz/VyJEj1a1bNw0ePFibNm2q8hw7duzQ7bffrh49eig+Pl5jxozRtm3b/nRudrtdI0aMUFJSkoqKiiRJ5eXlmj9/vgYOHKjLL79c/fr106OPPqry8nKPbWNiYvTwww/r7bffVmpqqmJjY7Vhw4bz/GoBOIPPQgLgUzNmzND777+vjIwMRUVF6dixY9qyZYv27Nmjrl27SjodEnfeeadSUlJ07bXX6tVXX9X9998vp9Op2bNn6+abb1ZaWppyc3M1YcIEffTRR2rWrJkkaffu3UpPT1fTpk11++23y9/fX8uXL1dmZqaWLl2q7t27Vzuvo0eP6rbbbpPdbtfSpUsVHh4up9Opu+66S1u2bNGNN96oqKgo7dq1Sy+99JL27t2r5557zmMf//nPf/Tee+8pPT1dISEhatu2bd1+MYGGxAUAPtSzZ0/XrFmzalyfkZHhio6Odq1atcq9bM+ePa7o6GhXp06dXNu2bXMv37Bhgys6OtqVl5fnXnb33Xe7unbt6vrvf//rXlZSUuKKj493paenu5fl5eW5oqOjXdu3b3cdOXLElZqa6kpOTnYdOHDAPWblypWuTp06uT777DOPOb766quu6Oho15YtW9zLzsxv9+7d5/gVAXA2uIQEwKdsNpu++OILlZSU1DimSZMmSk1NdT+OjIyUzWZTVFSUxxmUM7/fv3+/JKmyslIbN27UgAED1L59e/e4sLAwpaWlacuWLTp+/LjHc5WUlCgjI0OnTp3SsmXLPM6aFBQUKCoqSpGRkTp69Kj7V58+fSRJn3zyice+rrzySnXs2PFcvyQAzgKXkAD41OTJkzV16lRdffXV6tq1q/r166dhw4Z5BEfr1q1lsVg8tgsKClLr1q2rLJMkh8Mh6fRloJMnTyoiIqLK80ZFRcnpdOrQoUO67LLL3MsfeOAB+fv7Kz8/Xy1btvTYZt++fdqzZ48SEhKqPZaffvrJ43G7du3+7PABeImAAeBTKSkpuuKKK7RmzRpt3LhRubm5ysnJ0TPPPKN+/fpJkvz8/KrdtqblLpfL6/kMGjRIK1eu1JIlSzRp0iSPdU6nU9HR0Zo2bVq12/4+qC655BKv5wHgjxEwAHwuLCxM6enpSk9P108//aQbbrhBCxcudAeMt0JDQxUYGKji4uIq64qKimS1WtWmTRuP5RkZGQoPD9f8+fMVFBSkv/zlL+514eHh2rlzpxISEqqcEQJwYXEPDACfqaysVGlpqceyFi1aKCwsrMrLkr3h5+envn37au3atTpw4IB7+Y8//qh33nlHPXv2dL9a6bfuuece3XbbbZo3b55eeeUV9/IhQ4aopKREK1asqLLNL7/8orKysvOeM4CzwxkYAD5z4sQJ9evXT4MHD1anTp3UpEkTbdq0SV9++aWmTp1aK8+RlZWlTZs2afTo0Ro9erT8/Py0fPlylZeX64EHHqhxuylTpuj48eN6+OGH1bRpUw0dOlRDhw7Ve++9pxkzZuiTTz5Rjx49VFlZqaKiIhUUFGjx4sWKjY2tlXkD+GMEDACfueSSSzRq1Cht3LhRq1evlsvlUnh4uGbMmKHRo0fXynNcdtllWrZsmebNm6fnn39eLpdL3bp102OPPVbje8CcMWvWLJWVlSk7O1tNmzbVgAEDtGDBAr344ot66623tGbNGgUGBqpdu3bKzMys9mZhAHXD4jqfu90AAAB8gHtgAACAcQgYAABgHAIGAAAYh4ABAADGIWAAAIBxCBgAAGAcAgYAABiHgAEAAMYhYAAAgHEIGAAAYBwCBgAAGIeAAQAAxiFgAACAcf4/o/i2MSu7cjoAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 178
        },
        "id": "2OHFYb59XI2h",
        "outputId": "1f604895-79a0-4160-bfe0-446f5a59f7be"
      },
      "source": [
        "insurance_dataset['smoker'].value_counts()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "smoker\n",
              "no     2208\n",
              "yes     564\n",
              "Name: count, dtype: int64"
            ],
            "text/html": [
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>count</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>smoker</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>no</th>\n",
              "      <td>2208</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>yes</th>\n",
              "      <td>564</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table><br><label><b>dtype:</b> int64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 573
        },
        "id": "5972T2X4XRkv",
        "outputId": "bfc6473c-879a-469f-b373-94c9af57e9ff"
      },
      "source": [
        "# region column\n",
        "plt.figure(figsize=(6,6))\n",
        "sns.countplot(x='region', data=insurance_dataset)\n",
        "plt.title('region')\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 600x600 with 1 Axes>"
            ],
            "image/png": "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HnroId10002KjIzU6NGjdejQIbfHXb16tfr27atOnTopNjZWd911l3bt2iVJstlskqRnn33WNe5t27Zd2gn2kCuhrtKZP6LLli1Tnz591LFjR/Xs2VOvvvqqW5sDBw5o4sSJ6tatmzp37qzk5GT9+c9/Vnl5uVu7JUuW6NZbb1VERITi4uJ0//33Kz8/X4cPH1bPnj0lSRMmTHCN9fDhw1U7ubWs4rLY+c7lzz//rPT0dCUkJCgiIkIDBw7Uxx9/fM5+Nm/erAEDBigiIkKffPKJRowYIUkaPHiw69yczeFwaNKkSYqKilL37t3dnkc7duyQzWZz+70aM2aMbDabvv76a9e2Rx99VKNHj3bdLi4u1gsvvKDu3burY8eOSkpKqjSVf77n8MU890xncl0r7Ny5UyNGjFBkZKRuuukmTZo0ST/++KNbm+eff179+/dXVFSUbrnlFj366KOVZmn++c9/6p577tFNN92kqKgo9e/fX++8844kafjw4frHP/6hv/3tb65xLliwoOonthowY+JhY8aM0bXXXqs//vGPatiwoQ4dOqQjR45IkjZt2qTx48erb9++mjRpknJzczVv3jx9++23ysjIuOjHePjhh3Xs2DHl5ubq+eeflyQ1btzYtf/777/XzJkzNXr0aAUEBGju3LkaO3asPv74Y/n6+qpLly764IMP9PPPP8vf3187duyQn5+f9u7dqxMnTqhhw4bauXOnSktL1aVLF0lSfn6+hg0bpl//+teaPXu2LBaL/vSnP+n+++/X+vXr5efnp+3bt+sPf/iDfv/736tbt246ffq0du3apePHj0uSVq5cqTvvvFPDhw93rZe4/vrrq+W817Qroa6S9Mc//lGrVq3SmDFj1LlzZ33xxRd6/vnn5e/vr7vuukuSdPToUYWEhKh///5q0KCB9u7dqwULFujUqVMaO3asJOndd9/Viy++qPHjxysyMlLHjx/XP//5T508eVKhoaFauHChxo4dq0cffVSxsbGSpGbNml3+ia4lFzqXkydP1meffaaUlBSFhobqvffe07hx47Ro0SJXKJPOnMuZM2fqoYce0q9+9Ss1atRI06dP19NPP6309HSFhoZWeuwnn3xSAwcO1KJFi7Rx40Y9//zzstlsSkxMVKdOneTv76/t27crODhY5eXl+uc//+na9utf/1qStH37dg0fPtzV54QJE/TFF1/okUceUVhYmDZv3qwpU6bIarWqW7duks7/HL7Qc+9KYWpdpTOhZPjw4erWrZvmzZunoqIizZ8/Xw8//LBWrlzp6ufHH3/Ugw8+qGbNmunYsWN65ZVXNHz4cK1bt04+Pj46ceKEHnzwQd1000164YUX5Ofnp/3798vhcLjGMWXKFF1zzTVKTU2VJLVo0aImT/uFOeExP/74ozM8PNy5adOmc+6//fbbnXfeeafbtrfeessZHh7u/Oqrr5xOp9O5Zs0aZ3h4uPPHH390azdgwABnamqq63Zqaqqzb9++lR4jNTXVabPZnPv27XNt+/vf/+4MDw93bt++3el0Op2HDh1yhoeHO7dt2+Z0Op3OmTNnOh999FFnTEyMc/PmzU6n0+l84YUXnL1793b18dhjjzl79uzpPH36tNvxRkZGOpcvX+50Op3OpUuXOmNiYs57jsLDw51Lly49bxvTXCl1/eabb5w2m8351ltvud33ueeec8bHxzvLysoq9VteXu4sKSlxLl682BkfH+/a/tRTTzkHDRp0zuN1Op3O/Px8Z3h4uPOjjz76xTamutC53Lt3rzM8PNz55ptvut3vzjvvdDsnqampzvDwcOeXX37p1q6ir127dp1z+5w5c1zbysvLnd27d3empaW5tt1zzz3OqVOnOp1Op3PPnj3ODh06OJ944glnSkqK0+l0Og8ePOgMDw93fvHFF06n0+nMyclxhoeHOz/77DO3x0tJSXH+9re/dTqdF34OVxzPuZ57V4oroa533nmns7y83LXt66+/dtpsNuff/va3cx5TaWmp88iRI2713bVrl9vflnO59957naNHj/7F/bWNSzke1KhRI7Vs2VIvvPCC3nnnHde/RqQzl0b27t2rPn36uN0nOTlZ0pmpuerSrFkz17+spP83K1Gxmrx169Zq0aKFtm/fLunM9HFMTIyio6PdtlXMlkjS1q1b1aNHD3l7e6u0tFSlpaWyWq1q3769a6q0ffv2Kiws1NSpU7V161YVFRVV2zF50pVS188//1yS1Lt3b1eNSktLdfPNN+v777/Xt99+K+nMdHZGRobrMk2HDh00b948ff/99zp58qSkM7Xcs2eP0tPTtWPHjjp3jf1857KiZrfddpvbfZKSkrRnzx6dOnXKtS0oKEidO3e+pMdOSEhw/WyxWBQWFub2nDr793D79u3q2LGjEhMT3bbVq1dPHTt2lHTmdzMoKEhxcXGV6r53716VlZWd9zlcl5ha16KiIn3xxRe67bbbVFZW5qpR27Zt9atf/crtHT2bN2/WsGHDdNNNN6l9+/auGZeDBw9KkoKDg9WwYUPNmDFDWVlZlS4Pm4hLOR5ksVi0bNkyzZs3T08//bROnTqlDh066PHHH1fr1q3ldDrVpEkTt/sEBATIz89Pdru92sZhtVrdbldM85+95qFLly7asWOHTpw4oa+++krR0dEqKirS+vXrVVxcrF27dmnIkCGu9j/99JNee+01vfbaa5Uer6L/rl276tlnn9Xrr7+ukSNHyt/fX3369FFaWpqCgoKq7fhq25VS159++klOp1NxcXHnvP+3336rli1b6rnnntOqVav0yCOPqGPHjgoICNCmTZu0ePFi/fzzz2rQoIHuuOMOnTx5Um+//bZeffVVBQQE6Pbbb9fkyZN1zTXXVNsxecr5zqXdbpevr2+l5+y1114rp9Op48ePq379+q5tlyogIKDSY1dc7pSkmJgYLV68WN9995127Nih6OhoRUdH64cfftDBgwe1Y8cOde7c2TXmn376SYWFherQocM5H+/7779XixYtfvE5fPY/QK50ptbV4XCorKxM6enpSk9Pr3Tfin807Nq1Sw8//LB69uypUaNGqUmTJrJYLBo6dKjr9zwwMFCvvPKKMjIy9Nhjj6msrEzR0dGaNm1apXUvpiCYeFhISIgyMjJUUlKinTt36oUXXtCYMWOUnZ0ti8VSKd0eP35cxcXFCgwMlCT5+/tLUqV/oVZcP6wuXbp00ezZs7Vt2zY1atRIYWFhKioq0vPPP6+///3vKi4uVnR0tKt9YGCgunXr5rZotkKDBg1cPw8cOFADBw7UsWPHtGnTJqWnp8vHx0ezZs2q1vHXtiuhroGBgbJYLHrjjTdcf5D/9xgkaf369brzzjvdFk9u3rzZra2Xl5fuu+8+3Xffffruu++0bt06zZ07V40aNdIjjzxSbWM2UWBgoEpKSmS32131k6QffvhBFovF7QXIYrFU++NHRkbK19dX27dv144dO/Tb3/5WQUFB+vWvf63t27dr+/btuv32293G27hxYy1ZsuSc/VWsFTnfc/js3+G6ypN1DQgIkMVi0YMPPqhevXpV2t+oUSNJZ96y3bBhQ82fP19eXmcugBQUFFRq36lTJy1dulSnT5/Wtm3bNGfOHD3yyCPauHFjtY67unApxxC+vr6KiYnR6NGjdeLECR09elTt2rXT+vXr3dpVvOvlpptukiQ1b95ckpSbm+tqc+DAAVeiPrv/s2dALlV0dLROnTqlV1991RVA2rVrJ39/f2VmZupXv/qVWrVq5WrftWtXff3112rfvr0iIiLc/jvXQrDGjRtryJAhio+PdzuWyx23p5lc165du0qSCgsLK9UoIiJCDRs2lHTmX49nB5eysjKtW7fuF/tt3ry5fv/738tms7nGf65ZuLqiomb/W9P169erffv2rn9V/5LLPTf169dX+/bttXLlShUWFrrG06VLF73//vs6fPiw2z8abr75Zh07dky+vr7nrLufn1+l8f3vc7hie12sZwVP1rV+/fqKjIxUbm7uOWtU8bf29OnT8vX1dQtG5/ugtGuuuUbdunXTXXfdpcOHD7vGZlotmTHxoK+++kpz5sxRcnKyWrdurRMnTujll19Wy5YtFRwcrLFjx+qRRx7R5MmTNWDAAOXl5WnevHnq06ePawquc+fO+tWvfqVZs2Zp0qRJOnHihJYsWVJp+jEsLExr1qzRhx9+qDZt2qhRo0ZuQeJCwsLC1KRJE/3jH//QtGnTJEne3t668cYblZ2drf79+7u1Hz9+vAYPHqyRI0dq6NChuvbaa/XDDz/oH//4h6Kjo9WvXz9lZGSosLBQMTExatKkifbt26fPPvtM999/v6uf0NBQbdq0SdHR0apXr55CQkJcL5imulLqGhISonvuuUePPfaYRo4cqc6dO6ukpEQHDx7Utm3b9NJLL0k680K2atUqXX/99WrUqJHeeOMNFRcXu/U1ffp0Wa1WRUZGymq16osvvtBXX33lemdP06ZNZbVatW7dOrVq1Up+fn6y2WyVXgSvRDfccIN69+6t2bNn6/Tp0woJCdH777+vnTt3us7h+bRt21be3t5as2aNfHx85O3trYiIiEsaQ3R0tJYtW6YOHTq4fj+io6O1YsUK+fr6KioqytU2Pj5e3bt31wMPPKAHHnhANptNRUVF2r9/v7755hv98Y9/vOBzWLr8vymm83RdH3vsMd13331KSUlR3759ZbVadeTIEX3++ee64447FBsbq/j4eL322mt65plndOutt2rnzp1677333Pr529/+ptWrV6tXr1667rrr9MMPP2j58uW68cYbXTOzoaGhevfdd/XJJ5+oadOmatasmesfR55AMPGgpk2b6tprr9XLL7+s7777TgEBAYqOjtZzzz0nb29v9ezZUy+++KIWLVqkhx9+WEFBQRo6dKgmTZrk6sPX11cLFy7UjBkzNGHCBAUHBystLU2zZ892e6zBgwdr165deuaZZ1RYWKhBgwZVanMh0dHR+utf/+p2jblLly7Kzs6udN25TZs2WrVqlebPn6+nnnpKp06dUtOmTdWlSxfXi29ERIRee+01ffTRRzpx4oRatGihkSNH6qGHHnL1M336dM2aNUujRo3S6dOn9frrr7vebmqqK6mu06ZNU0hIiFauXKlFixapQYMGCgkJcVvw98QTT+jJJ5/UM888o3r16mnQoEG69dZbXQFVkqKiovT2229r1apVKioqUuvWrfX444+71h15eXkpPT1dL7zwgu6//34VFxdr06ZNdeaF7LnnntMLL7ygzMxMFRYWKjQ0VBkZGerRo8cF79u4cWNNnz5dS5cu1fvvv6/S0lL95z//uaTHj4mJ0bJly9xmRip+Jzt27FhpnU9GRoaWLFmiN998UwUFBQoICNCvf/1r3XHHHZIu/ByWqudviuk8Wdcbb7xRb7zxhhYsWKDHH39cJSUlatGiheLi4tSmTRtJUrdu3TR58mQtX75ca9eu1Y033qiXX37ZbXF9cHCwvLy8NH/+fP34448KCgpSQkKCHn30UVebUaNG6dChQ0pNTZXD4dDYsWM1bty4SzhT1cvidDqdHnt0AACAs7DGBAAAGINgAgAAjEEwAQAAxiCYAAAAYxBMAACAMQgmAADAGAQTAABgDIIJgDrn8OHDstlsWrt2raeHAuASEUwAAIAx+ORXAHWO0+lUcXGx6ztKAFw5mDEB4FGnTp2q9j4tFov8/f0JJcAViGACoNYsWLBANptN+/fv16RJk9SlSxfdfffdkqT33ntPd9xxhzp16qSYmBhNnDhR3377baU+VqxYoZ49e6pTp04aPHiwduzYoeHDh2v48OGuNr+0xiQnJ0d33323IiMjFR0drYceekgHDhw45xi/+eYbTZ06VdHR0brpppv0+OOPq6ioqAbOCoCzEUwA1LoJEyaoqKhIEydO1JAhQ7R48WKlpqaqTZs2mjp1qkaMGKGcnBzdc889cjgcrvu98cYbevrpp9WiRQtNmTJF0dHReuSRR3TkyJELPubnn3+uBx54QD/++KPGjh2r+++/Xzt37tRdd92lw4cPV2qfkpKikydP6tFHH1VSUpLWrl2rhQsXVut5AFCZj6cHAODqc8MNN2ju3LmSpIKCAt16661KSUnRmDFjXG169+6tQYMG6Y033tCYMWNUXFysF198UREREXrttdfk43Pmz5fNZtPUqVPVokWL8z7ms88+q8DAQK1cuVJBQUGSpF69emnQoEFasGCB5syZ49a+Xbt2mjVrlut2YWGhVq9erSlTplTHKQDwC5gxAVDrhg0b5vr5448/Vnl5uZKSknTs2DHXf9dee63atGmjbdu2SZJ2796twsJCDR061BVKJKl///4KDAw87+MdPXpUe/fu1aBBg1yhRDoTkG6++WZt3rz5vGOUpOjoaBUWFurEiRNVOWQAF4kZEwC1rlWrVq6fDx48KKfTqd69e5+zbUUI+e9//ytJCg4OrrS/ZcuW5328ivuGhIRU2hcWFqYtW7bo1KlTql+/vmv7dddd59bOarVKkux2uxo2bHjexwNQdQQTALXO39/f9XN5ebksFosyMzPP+S6as8NCbfLyOveEMp+wANQsggkAjwoODpbT6VSrVq3OOaNRoWIG49ChQ4qLi3NtLy0tVUFBgWw22wXvm5eXV2lfbm6uGjVq5LEABMAda0wAeFTv3r3l7e2thQsXVpqNcDqd+umnnyRJHTt2VFBQkN5++22Vlpa62nzwwQey2+3nfYxmzZqpXbt2evfdd93e5bNv3z5t3bpV3bp1q8YjAnA5mDEB4FHBwcFKSUnR3LlzVVBQoF69eqlBgwY6fPiwNm7cqKFDh2rkyJHy8/PTuHHj9Mwzz+i+++5TUlKSCgoKtHbt2krrTs7lscce06hRo3TnnXdq8ODBOn36tJYvX66AgACNHTu2Fo4UwMUgmADwuNGjR6tt27Z69dVXtWjRIklSixYtFB8frx49erja3XvvvXI6nXrllVc0Z84c3XDDDVq8eLFmzpzptm7lXG6++WYtXbpUGRkZysjIkI+Pj7p06aIpU6aodevWNXp8AC4e35UD4IpWXl6url276tZbb9XMmTM9PRwAl4k1JgCuGD///HOldSjvvvuuCgsLFRMT46FRAahOXMoBcMX48ssvlZ6erttuu01BQUHas2ePVq9erfDwcN12222eHh6AakAwAXDFaNmypVq0aKG//OUvstvtCgwM1MCBAzV58mT5+fl5engAqgFrTAAAgDFYYwIAAIxBMAEAAMYgmAAAAGMQTAAAgDEIJgAAwBgEEwAAYAyCCQAAMAbBBAAAGINgAgAAjPH/AckjNiqr2LwfAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 241
        },
        "id": "rkckkdYVXYnV",
        "outputId": "ab4b8605-c676-4156-a3d8-4e5abd7ac01d"
      },
      "source": [
        "insurance_dataset['region'].value_counts()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "region\n",
              "southeast    766\n",
              "southwest    684\n",
              "northwest    664\n",
              "northeast    658\n",
              "Name: count, dtype: int64"
            ],
            "text/html": [
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>count</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>region</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>southeast</th>\n",
              "      <td>766</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>southwest</th>\n",
              "      <td>684</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>northwest</th>\n",
              "      <td>664</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>northeast</th>\n",
              "      <td>658</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table><br><label><b>dtype:</b> int64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 21
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 541
        },
        "id": "vV_nE8lNXgji",
        "outputId": "7f6f4e0d-4059-4925-c5a8-16c2c71181f9"
      },
      "source": [
        "# distribution of charges value\n",
        "plt.figure(figsize=(6,6))\n",
        "sns.displot(insurance_dataset['charges'])\n",
        "plt.title('Charges Distribution')\n",
        "plt.show()"
      ],
      "execution_count": 40,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 600x600 with 0 Axes>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 500x500 with 1 Axes>"
            ],
            "image/png": "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OdasWVPu29frJWg0j4qss7FRQqVygEaTBZ1Ob3K/i0sV5ObqihxHTo3cunIZS/EkjHN1OiCfc/DkbvPhw0xZdeZS1PNpKdin5bCGHoGK0afcDxwVLpCfVaVKFbRt2xZ79uwxLFOpVLhz545JbVpaGpydnUu9zdxc+U+6TqcvsL6gs8iLW2POsQy7qaX815O7zeI8puZU2PNpSdin5bCGHgHL6LPi73TPh6enJ65du2YSBteuXYOnp6eZZkVERFSwCh/Ijx49wuHDh+Hn52dYFhYWhrS0NBw7dsyw7Nq1a7h48SLCwsLMMU0iIqJCCbXLOisrC0eOHAEA3L59GxkZGYiLiwMAtGjRAsnJyVi7di06d+6MunXr4u7du1i/fj3u3buHRYsWGcYJDAxEaGgoJkyYgHHjxqFSpUpYsGAB1Go1unTpYpbeiIiICiNUIN+/fx8ffPCB0bK821988QVq164NrVaLBQsWIDU1FQ4ODggMDMS0adPg7+9vtN7ChQsxa9YsTJ48Gbm5uQgNDcXEiRN5lS4iIhKSUOnk5uaGP//8s9CadevWyRrLyckJM2fOxMyZM8tialSOcnV6vPCCY6E1OVodNGlZz2lGRETPn1CBTNbJ1kaJsYsLv4La3JE89k9Elq3Cn9RFRERkCRjIREREAmAgExERCYCBTEREJAAGMhERkQAYyERERAJgIBMREQmAgUxERCQABjIREZEAGMhEREQCYCATEREJgIFMREQkAAYyERGRABjIREREAmAgExERCYCBTEREJAAGMhERkQAYyERERAJgIBMREQmAgUxERCQABjIREZEAGMhEREQCYCATEREJgIFMREQkAAYyERGRABjIREREAmAgExERCYCBTEREJAAGMhERkQAYyERERAJgIBMREQmAgUxERCQABjIREZEAGMhEREQCYCATEREJgIFMREQkAAYyERGRABjIREREAmAgExERCUCoQL5x4wYmT56MqKgoNGrUCJGRkfnWbd26FV27doWfnx+6d++OQ4cOmdSkp6djwoQJaNGiBQIDAzFy5EjcvXu3vFsgIiIqEaECOSkpCUeOHEH9+vXh5eWVb82uXbswadIkhIeHY82aNQgICMDw4cPx22+/GdWNGjUKCQkJmDp1KubNm4dr164hJiYGubm5z6ETIiKi4rE19wSe1qFDB3Tq1AkAMH78eJw/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KJVK6PV65PcIK5UKPNA8LnSM6qrKRdbIrSvPsRRQQIJpkxVl/nLr8utThHmV9VjP9inKvMp6rJL0KdL85dSI0KOc/1+W9v+HT/cpZ5tytid3LDlsbORd8sNiA7lx48b44IMP8O677xotj4yMRGBgIKZPn26mmREREZnilbqIiIgEYLGBrFKpkJ6ebrI8LS0Nzs7OZpgRERFRwSw2kD09PZGcnGy0LD09Hffu3YOnp6eZZkVERJQ/iw3ksLAwJCYmQqPRGJbFxcVBqVSidevWZpwZERGRKYs9qSstLQ3dunWDh4cH3nvvPcOFQV599VVeGISIiIRjsYEMPLl05vTp040unTl69GheOpOIiIRj0YFMRERUUVjsMWQiIqKKhIFMREQkAAYyERGRABjIREREAmAgExERCYCBTEREJAAGcjm4evUqBg8ejICAALRu3Rpz585FTk6OuaeFGzduYPLkyYiKikKjRo0QGRmZb93WrVvRtWtX+Pn5oXv37jh06JBJTXp6OiZMmIAWLVogMDAQI0eOxN27d03qfv31V/Tu3Rv+/v5o3749Vq9ejWe/aSdJElavXo127drB398fvXv3xm+//VaiHnfv3o33338fYWFhCAgIQFRUFLZt22ayzYrcIwAcOXIE/fv3R8uWLeHr64uOHTti1qxZJtdvP3jwILp37w4/Pz907doV27dvNxkrJycHc+bMQevWrREQEIDBgwebXHYWkP+6lvPYlkRmZibCwsKgVqvx+++/F3uboj6fO3bsgFqtNvkzb948i+nxad999x1ee+01+Pn5ITg4GO+88w4eP/7/n0K0pNdssUlUplJTU6XWrVtLb731lhQfHy9t3bpVatasmTRt2jRzT03at2+fFBYWJo0YMUKKjIyUunXrZlLz008/SWq1WlqwYIF07NgxadKkSVKjRo2kM2fOGNUNGTJECgsLk3bt2iXt379fioyMlLp37y5ptVpDzfXr16WAgABp2LBhUmJiorR+/XqpcePG0tq1a43GWrVqldS4cWNp/fr1UmJiojRs2DApMDBQunnzZrF77NWrlzR69Ghp165dUmJiojRv3jypYcOG0pIlSyymR0mSpO+//16aM2eOFBcXJx0/flzatGmT1KJFC2nw4MGGmlOnTkk+Pj7SpEmTpGPHjkkLFiyQ1Gq1tHv3bqOxJk2aJDVr1kzaunWr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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gKYZDvzqX3iR"
      },
      "source": [
        "Data Pre-Processing"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "pr9nJq15YFsY"
      },
      "source": [
        "Encoding the categorical features"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QVXYBAt_XwPO"
      },
      "source": [
        "# encoding sex column\n",
        "insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)\n",
        "\n",
        "3 # encoding 'smoker' column\n",
        "insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)\n",
        "\n",
        "# encoding 'region' column\n",
        "insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "EL0T11t-ZdNF"
      },
      "source": [
        "Splitting the Features and Target"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Z5UK60q_ZMgr"
      },
      "source": [
        "X = insurance_dataset.drop(columns='charges', axis=1)\n",
        "Y = insurance_dataset['charges']"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ACFgPgoUZvzP",
        "outputId": "c0159e71-af21-4cdd-f6db-11982884f819"
      },
      "source": [
        "print(X)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "      age  sex     bmi  children  smoker  region\n",
            "0      19    1  27.900         0       0       1\n",
            "1      18    0  33.770         1       1       0\n",
            "2      28    0  33.000         3       1       0\n",
            "3      33    0  22.705         0       1       3\n",
            "4      32    0  28.880         0       1       3\n",
            "...   ...  ...     ...       ...     ...     ...\n",
            "2767   47    1  45.320         1       1       0\n",
            "2768   21    1  34.600         0       1       1\n",
            "2769   19    0  26.030         1       0       3\n",
            "2770   23    0  18.715         0       1       3\n",
            "2771   54    0  31.600         0       1       1\n",
            "\n",
            "[2772 rows x 6 columns]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7zcw-kgaZxvI",
        "outputId": "19fff35b-c9eb-4d98-efbe-c3b0529b1876"
      },
      "source": [
        "print(Y)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0       16884.92400\n",
            "1        1725.55230\n",
            "2        4449.46200\n",
            "3       21984.47061\n",
            "4        3866.85520\n",
            "           ...     \n",
            "2767     8569.86180\n",
            "2768     2020.17700\n",
            "2769    16450.89470\n",
            "2770    21595.38229\n",
            "2771     9850.43200\n",
            "Name: charges, Length: 2772, dtype: float64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "N56oAuY3bQSF"
      },
      "source": [
        "Splitting the data into Training data & Testing Data"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8HQEpONYbL0-"
      },
      "source": [
        "X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GKh0p46hb3iD",
        "outputId": "d0b45bf1-a8de-4963-f0cc-bc4d3581d497"
      },
      "source": [
        "print(X.shape, X_train.shape, X_test.shape)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(2772, 6) (2079, 6) (693, 6)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DQoIaogAcCF2"
      },
      "source": [
        "Model Training"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-1RWRMnncEJj"
      },
      "source": [
        "Linear Regression"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "18wYy3bub9ej"
      },
      "source": [
        "# loading the Linear Regression model\n",
        "regressor = LinearRegression()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 74
        },
        "id": "yRHiG9dqcxxN",
        "outputId": "9d6d6130-f3df-447c-c015-0a397df6352f"
      },
      "source": [
        "regressor.fit(X_train, Y_train)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LinearRegression()"
            ],
            "text/html": [
              "<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearRegression</label><div class=\"sk-toggleable__content\"><pre>LinearRegression()</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 29
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "PGM93AzWc-VJ"
      },
      "source": [
        "Model Evaluation"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "NhRL9x_wc6-p"
      },
      "source": [
        "# prediction on training data\n",
        "training_data_prediction =regressor.predict(X_train)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bQ8gnyDMdMSb",
        "outputId": "b1feaf86-eb76-409b-d238-4c74b1ed2cac"
      },
      "source": [
        "# R squared value\n",
        "r2_train = metrics.r2_score(Y_train, training_data_prediction)\n",
        "print('R squared vale : ', r2_train)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "R squared vale :  0.754033366262729\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pHQSjvekdsgO"
      },
      "source": [
        "# prediction on test data\n",
        "test_data_prediction =regressor.predict(X_test)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YWPIzD8ud5KA",
        "outputId": "c27e76e0-d9c8-4153-8a06-161d1f3cd8fa"
      },
      "source": [
        "# R squared value\n",
        "r2_test = metrics.r2_score(Y_test, test_data_prediction)\n",
        "print('R squared vale : ', r2_test)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "R squared vale :  0.7398782467495058\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def get_input():\n",
        "    try:\n",
        "        x = int(input(\"Enter age: \"))\n",
        "        if x < 18:\n",
        "            print(\"Age should be 18 or above\")\n",
        "            return False\n",
        "        else:\n",
        "            print(\"Age is valid\")\n",
        "\n",
        "        y = int(input(\"Enter gender (male-0, female-1): \"))\n",
        "        if y not in [0, 1]:\n",
        "            print(\"Gender should be either 0 (male) or 1 (female)\")\n",
        "            return False\n",
        "\n",
        "        z = float(input(\"Enter BMI: \"))\n",
        "        if z == 0:\n",
        "            print(\"BMI cannot be zero\")\n",
        "            return False\n",
        "\n",
        "        n = int(input(\"Enter the number of children: \"))\n",
        "\n",
        "        s = int(input(\"Enter smoker status (smoker-0, non-smoker-1): \"))\n",
        "        if s not in [0, 1]:\n",
        "            print(\"Smoker status should be either 0 (smoker) or 1 (non-smoker)\")\n",
        "            return False\n",
        "\n",
        "        r = int(input(\"Enter region (southeast:0, southwest:1, northeast:2, northwest:3): \"))\n",
        "        if r not in [0, 1, 2, 3]:\n",
        "            print(\"Region should be either 0, 1, 2, or 3\")\n",
        "            return False\n",
        "\n",
        "        return (x, y, z, n, s, r)\n",
        "\n",
        "    except ValueError:\n",
        "        print(\"Invalid input. Please enter the correct data type.\")\n",
        "        return False\n",
        "\n",
        "input_data = None\n",
        "while not input_data:\n",
        "    input_data = get_input()\n",
        "    if input_data:\n",
        "        print(\"All inputs are valid.\")\n",
        "        print(\"Input data:\", input_data)\n",
        "    else:\n",
        "        print(\"Invalid input. Please try again.\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QSdub0yHk7up",
        "outputId": "116df9dd-424d-49b9-cd81-d3d118efa6b9"
      },
      "execution_count": 53,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Enter age: 19\n",
            "Age is valid\n",
            "Enter gender (male-0, female-1): 1\n",
            "Enter BMI: 3\n",
            "Enter the number of children: 1\n",
            "Enter smoker status (smoker-0, non-smoker-1): 0\n",
            "Enter region (southeast:0, southwest:1, northeast:2, northwest:3): 0\n",
            "All inputs are valid.\n",
            "Input data: (19, 1, 3.0, 1, 0, 0)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_RhHS3AkeOVA"
      },
      "source": [
        "Building a Predictive System"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "H9bGdPbBd_Xd",
        "outputId": "12fe885b-7824-46ff-97f5-008eb7cdcd6b",
        "collapsed": true
      },
      "source": [
        "# changing input_data to a numpy array\n",
        "input_data_as_numpy_array = np.asarray(input_data)\n",
        "\n",
        "# reshape the array\n",
        "input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)\n",
        "\n",
        "prediction = regressor.predict(input_data_reshaped)\n",
        "print(prediction)\n",
        "\n",
        "print('The insurance cost is $', prediction[0])"
      ],
      "execution_count": 55,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[17754.78032225]\n",
            "The insurance cost is $ 17754.780322253504\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/sklearn/base.py:465: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names\n",
            "  warnings.warn(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "iSihL4MUmf5b"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}