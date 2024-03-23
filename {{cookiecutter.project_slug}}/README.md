# Template

## Cookiecutter

To apply your Cookiecutter template to a new project and manage updates with
Cruft, follow these steps:

1. **Create a new project from a Cookiecutter template**:

    - Run: `cookiecutter <template-path-or-url>`
    - Follow the prompts to customize your new project.

    - Better use cruft to manage the project.

2. **Navigate to your new project directory**:

    - `cd <your-new-project>`

3. **Link the project to the Cookiecutter template with Cruft** for future
   updates:

    - Run: `cruft link <template-path-or-url>`
    - This step records the template's details for future updates.

4. **To update your project with changes from the template** in the future, run:
    - `cruft update`
    - This command checks for updates from the linked template and attempts to
      merge them into your project.

Common gotcha

-   https://github.com/cookiecutter/cookiecutter/issues/1624#issuecomment-1076475537

## PEP-0561: `py.typed`

The `py.typed` marker indicates that a Python package includes type hints within
its Python code files (`.py`). When a package includes a `py.typed` file, it
signals to type checkers that the package supports type checking. This approach
is straightforward and allows you to write code and type hints together, making
it easy to maintain and update both simultaneously. The `py.typed` file itself
is typically empty and placed in the root of your package's directory structure.

```{admonition} References
:class: seealso

- [PEP-0561](https://peps.python.org/pep-0561/)
```

## Git: Git Attributes

```{admonition} References
:class: seealso

-   [8.2. Customizing Git - Git Attributes](https://git-scm.com/book/en/v2/Customizing-Git-Git-Attributes)
```

On a high level, the snippet below is used in a `.gitattributes` file to
conditionally apply Git LFS handling and disable text normalization for the
`.mutmut-cache` file or directory in projects that are not Jupyter notebook
projects. In addition, we add a linguistic information, which is used for
repository statistics, such as language breakdown and determining whether files
are counted as documentation (i.e. won't be dominated by jupyter notebooks).

```text
# {%- if cookiecutter.jupyter_notebook_project != 'yes' %}
.mutmut-cache filter=lfs diff=lfs merge=lfs -text
# {%- endif %}

*.ipynb linguist-documentation
```

## LICENSE, CONDUCT and CONTRIBUTING

...
