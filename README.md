# AniMate

**AniMate** is a Python-based anime recommendation system that utilizes natural language processing (NLP) to suggest anime based on user preferences.

## Installation

To get started with AniMate, you need to install the following software:

- **Python**: Ensure you have Python 3.10 or higher installed. [Download Python](https://www.python.org/downloads/).
- **Poetry**: A tool for dependency management and packaging in Python. [Install Poetry](https://python-poetry.org/docs/#installation).
- **Git**: A version control system to manage your codebase. [Download Git for Windows](https://git-scm.com/download/win).

### Clone the Repository

To clone the repository, follow these steps:

1. Open your command line interface (CLI).
2. Run the following command:

    ```bash
    git clone https://github.com/Asifdotexe/AniMate.git
    ```

3. Navigate to the project directory:

    ```bash
    cd AniMate
    ```

### Install Dependencies

1. Install project dependencies using Poetry:

    ```bash
    poetry install
    ```

2. Run the application:

    ```bash
    poetry run streamlit run app.py
    ```

## Contributing

To contribute to the project, follow these steps:

1. **Create an Issue**: If you find a bug or have a feature request, create a new issue on the GitHub repository.

2. **Create a Branch**:
    - Branch names should follow the GitFlow strategy.
    - For a new feature: `feature/<feature-name>`
    - For a bug fix: `bugfix/<bugfix-name>`

    ```bash
    git checkout -b feature/<feature-name>
    ```

3. **Commit Changes**:
    - Make sure to commit your changes with a message referring to the issue number.
    - Use the format: `#<issue-number> "commit message"`

    ```bash
    git add .
    git commit -m "#<issue-number> commit message"
    ```

4. **Push Changes**:

    ```bash
    git push origin feature/<feature-name>
    ```

5. **Create a Pull Request**:
    - Go to the GitHub repository and create a pull request to merge your branch into the `develop` branch.
