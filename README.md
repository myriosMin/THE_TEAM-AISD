# THE TEAM

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

---

## 🔧 Setup Instructions for THE\_TEAM-AISD (Kedro + Spark Project)

### 🚨 Prerequisites

* Ensure **Docker Desktop** is installed and running on your machine:
  👉 [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)

---

### 🛠️ How to Build the Docker Image

1. Open your terminal in the folder containing the `Dockerfile`.
2. Run the following command:

   ```bash
   docker build -t <Image name> .
   ```

---

### ▶️ How to Start the Container

Run the following command in your terminal:

```bash
docker run -p 8888:8888 <Image name>
```

* This will start **JupyterLab** inside the container.
* Look for a URL in the terminal like:

  ```
  http://127.0.0.1:8888/lab?token=your_token_here
  ```
* Open this link in your browser to access the Jupyter environment.

---

### 📓 Running the Project Pipeline in Jupyter

Once JupyterLab is open:

1. Navigate to the `notebooks/` directory.

2. Open your `.ipynb` file (or create a new notebook).

3. Add the following in a cell to run the pipeline:

   ```python
   !chmod +x run.sh
   !./run.sh
   ```

4. Run the cell — this will execute the full Kedro pipeline via `run.sh`.

---

### 📁 Note on Project Files

* The container automatically **clones the GitHub repo** inside `/home/kedro_docker`.
* All data folders (`data/01_raw`, etc.) will be created on first run by the pipeline if not already present.
* Jupyter runs as the `kedro_docker` user and handles file permissions internally.

---

### ✅ Summary of Docker Commands

| Step          | Command                                        |
| ------------- | ---------------------------------------------- |
| Build image   | `docker build -t kedro-spark-notebook .`       |
| Run container | `docker run -p 8888:8888 kedro-spark-notebook` |
| Run pipeline  | Use `!./run.sh` inside Jupyter notebook        |

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
