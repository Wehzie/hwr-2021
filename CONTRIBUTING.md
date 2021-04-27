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

- [PEP8](https://www.python.org/dev/peps/pep-0008/) enforced with 

- static type hints for functions' input/output behavior; gradual typing <https://www.python.org/dev/peps/pep-0483/>. 
- FP; let functions be pure when easily possible
- KISS; keep it simple stupid; no huge functions or modules; only do one thing per method or module
- OOP; don't do Java but put all methods into appropriate classes

Install code formatter `black` to enforce PEP8.
		
	pip3 install black

Navigate to the project's root directory.

	cd ~/path/to/project/

Execute `black` on the project's root directory.

	black .

### Testing

- unit tests with pytest or unittest
	
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
- don't leave old code snippets uncommented accross committs
- use branches
- use pull requests
	
### Parameters

- do not hard code
- no uncommenting-commenting of parameters and in-code editing
- use .env file https://github.com/theskumar/python-dotenv/stargazers
- or use command line arguments with argparse

### Roadmap

- define a roadmap
- set deadlines
- review failures to fulfill expectations