## Managing Development

### Architecture
    
- a description of the architecture is maintained at [ARCHITECTURE.md](ARCHITECTURE.md)
- [ARCHITECTURE.md](ARCHITECTURE.md) contains a written and graphical description
- the architecture is drawn using vector graphics, e.G. <https://www.diagrams.net/>

### Dependencies

- as few dependencies as possible
- dependencies should be actively maintained
- dependencies are tracked in [requirements.txt](requirements.txt)

Run the following to update the requirements.

	pip3 freeze > requirements.txt

### Containerization
  
- packages are installed into a virtual environment `venv`, see [README.md](README.md) for installation instructions
	
	
### Style

- we follow the [PEP8](https://www.python.org/dev/peps/pep-0008/) style guide
- we use static typing for functions' input and output; see [gradual typing](https://www.python.org/dev/peps/pep-0483/)
- we use formatter `black` to enforce PEP8
- we use linter `flake8` to find bugs
- we use type checker `mypy` to find bugs
- we use `pydocstyle` to enforce PEP 257 docstrings
- we use `pep8-naming` to enforce PEP8 naming conventions

Navigate to the project's root directory.

	cd ~/path/to/project/

Execute `black` on the project's root directory.

	black .

Execute `flake8` on `src` and `tests` directories; with plugins this also applies `pydocstyle` and `pep8-naming`.

	flake8 src/ tests/

Execute `mypy` on `src` and `tests` directories.

	mypy src/ tests/

### Testing

We use the `unittest` module which is available in the Python standard library.

To run all tests first navigate to the project's root directory.

	cd ~/path/to/project/

Then run the following command to let the `unittest` module discover test files.

	python -m unittest discover -s tests

So long as test files are named `test*.py` they will be automatically executed.

### Documentation

- comments are for developers
- docstrings are for users
- codetags make comments better searchable; for example TODO, BUG, IDEA; see [PEP 350](https://www.python.org/dev/peps/pep-0350/#mnemonics)
	
### Version control

- we use `git`; no data should be committed to the repository, see [gitignore](https://git-scm.com/docs/gitignore)
	
### Parameters

- [argparse](https://docs.python.org/3/library/argparse.html) is to preferred over hard coding and commenting code in and out
- [.env](https://github.com/theskumar/python-dotenv/) files can be used to keep secrets and to determine the environment where the pipeline is run
- parameters should be declared towards the top of any source code file
- [default argument values](https://docs.python.org/3/tutorial/controlflow.html#default-argument-values) are also an effective way of maintaining parameters

### Roadmap

- we define a roadmap and set deadlines
- we review failures to fulfill expectations