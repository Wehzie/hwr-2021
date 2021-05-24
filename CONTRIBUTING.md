## Managing Development

### Architecture
    
- [ARCHITECTURE.md](ARCHITECTURE.md)
- written and graphical architecture description
- make architecture figure using vector graphics, e.G. draw.io

### Dependencies

- as few dependencies as possible
- dependencies should be actively maintained
- [requirements.txt](requirements.txt)

### Containerization
  
- venv, see [README.md](README.md)
	
### Style

Follow the [PEP8](https://www.python.org/dev/peps/pep-0008/) style guide.
Use static typing for functions' input and output; see [gradual typing](https://www.python.org/dev/peps/pep-0483/).

Install code formatter `black` to enforce PEP8, linter `flake8` and static type checker `mypy` to find possible bugs.
		
	pip3 install black flake8 pep8-naming mypy

Navigate to the project's root directory.

	cd ~/path/to/project/

Execute `black` on the project's root directory.

	black .

Execute `flake8` on `src` and `tests` directories.

	flake8 src/ tests/

Execute `mypy` on `src` and `tests` directories.

	mypy src/ tests/

### Testing

Use unit testing.
We use the `unittest` module which is available in the Python standard library.

To run all tests first navigate to the project's root directory.

	cd ~/path/to/project/

Then run the following command to let the `unittest` module discover test files.

	python -m unittest discover -s tests

So long as test files are named `test*.py` they will be automatically executed.

### Documentation

- README.md
	- see inspiration at <https://github.com/matiassingers/awesome-readme>
	- software description, requirements, running, ...
- comments are for developers
- docstrings are for users
- use codetags for comments if applicable <https://www.python.org/dev/peps/pep-0350/#mnemonics>; TODO, BUG, IDEA, TODOC, ...
	
### Version control

- git
- github
- gitignore (don't include data)
- don't leave old code snippets uncommented across commits
- use branches
- use pull requests
	
### Parameters

- do not hard code
- no uncommenting-commenting of parameters and in-code editing
- use .env file https://github.com/theskumar/python-dotenv/
- use command line arguments with argparse

### Roadmap

- define a roadmap
- set deadlines
- review failures to fulfill expectations