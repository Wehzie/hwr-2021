## Managing Development

- architecture
    
    - ARCHITECTURE.md
    - written and graphical architecture description
    - make architecture figure using vector graphics, e.G. draw.io

- dependencies

	- as few dependencies as possible
	- dependencies should be actively maintained
	- requirements.txt

- containerization
  
	- docker or venv
	
- style

	- PEP8
	- could be automated and enforced with black
	- static type hints for functions' input/output behavior; gradual typing <https://www.python.org/dev/peps/pep-0483/>. 
	- FP; let functions be pure when easily possible
	- KISS; keep it simple stupid; no huge functions or modules; only do one thing per method or module
	- OOP; don't do Java but put all methods into appropriate classes

- testing

	- unit tests with pytest or unittest
	
- documentation

	- README.md
		- see inspiration at <https://github.com/matiassingers/awesome-readme>
		- software description, requirements, running, ...
	- comments are for developers
	- docstrings are for users
	- use codetags for comments if applicable <https://www.python.org/dev/peps/pep-0350/#mnemonics>; TODO, BUG, IDEA, TODOC, ...
	
- version control

	- git
	- github
	- gitignore (don't include data)
	- don't leave old code snippets uncommented accross committs
	- use branches
	- use pull requests
	
- parameters

	- do not hard code
	- no uncommenting-commenting of parameters and in-code editing
	- use .env file https://github.com/theskumar/python-dotenv/stargazers
	- or use command line arguments with argparse

- roadmap

    - define a roadmap
    - set deadlines
    - review failures of fulfilling expectations