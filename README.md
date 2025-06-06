# THE TEAM

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

---

## 🔧 Setup Instructions for THE\_TEAM-AISD (Kedro + Spark Project)

### 🚨 Prerequisites

* Ensure **Docker Desktop** is installed and running on your machine:
  👉 [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)

---

### 🛠️ Build and Start the Project with Docker Compose

#### 1. Clone the repository

```bash
git clone https://github.com/myriosMin/THE_TEAM-AISD
cd THE_TEAM-AISD
```

#### 2. Build the containers

```bash
docker-compose build
```

> ⏳ This may take several minutes (\~15 mins) during the first run (due to package downloads).

---

### ▶️ Start JupyterLab

```bash
docker-compose up jupyter
```

* Open the provided link from the terminal (e.g. `http://localhost:8888/?token=...`)
* Use this environment to open `eda.ipynb`, run cells, or explore Kedro outputs

---

### 🛠️ Run the Kedro Pipeline (without Jupyter)

```bash
docker-compose run runner
```

* This runs `run.sh` inside the container.

---

### 📒 Alternative: Run Pipeline via Jupyter Terminal

1. Inside JupyterLab, go to the top menu bar and click:
   **File > New > Terminal**

2. In the new terminal window, run:

   ```bash
   kedro run
   ```

> ⏳ **Note**: The **first run may take \~15 minutes** due to downloading and installing dependencies.
> Subsequent runs will be significantly faster thanks to Docker layer caching and existing compiled packages.

---

### 📃 Notes

* The container automatically clones the GitHub repo inside `/home/kedro_docker`.
* All necessary packages (e.g. PySpark, Kedro, Torch, etc.) are installed.
* Runs as `kedro_docker` user with proper permissions.
* Make sure `run.sh` is executable:

  ```bash
  git update-index --chmod=+x run.sh
  ```

---

### ✅ Summary of Docker Commands

| Step               | Command                     |
| ------------------ | --------------------------- |
| Build containers   | `docker-compose build`      |
| Start JupyterLab   | `docker-compose up jupyter` |
| Run Kedro pipeline | `docker-compose run runner` |

---

### 🔎 Troubleshooting

* If you get `exec format error`, ensure:

  * `run.sh` has LF line endings
  * Starts with: `#!/bin/bash`

* If Jupyter doesn’t open, try copying the full URL with token from terminal.

---

## Overview

🔧 Setup Instructions for THE_TEAM-AISD (Kedro + Spark Project)
🚨 Prerequisites
Ensure Docker Desktop is installed and running on your machine:
👉 https://www.docker.com/products/docker-desktop

This is your new Kedro project with PySpark setup, which was generated using `kedro 0.19.12`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the files `src/tests/test_run.py` and `src/tests/pipelines/data_science/test_pipeline.py` for instructions on how to write your tests. Run the tests as follows:

```
pytest
```

To configure the coverage threshold, look at the `.coveragerc` file.

## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. Install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `catalog`, `context`, `pipelines` and `session`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
