# meeting minutes

## Organizational tasks

- track progress and update slides. Appointed to ?
    - set deadlines - are we on schedule?
	- which articles have been read + total number?
	- how much text was written for the report?
	- how far is programming of the classifier?
	- how far is empirical evaluation progress?
		- what statistical tests will we do; is this implemented?

- overlook overall system architecture. Appointed to ?

- design empirical evaluation, including scripts. Appointed to ?

## 2021, April 26, Monday

- best practices for managing the project
- made shared github
- made shared google slides
- instruction paper discussion
- architecture discussion
- next meeting 16:00 on Thursday, 29. April 2021
    - goal: define architecture

### to do

- literature review
    - character segmentation: Lars, Andele
    - character recognition: Mark, Rafael
- architecture diagram

## 2021, April 29, Thurdsay

- presented papers
- text segmentation: line recognition (problem: curves not straight) and then word recognition with ink histograms across the line
- character segmentation:
	1. e. G. CNN with pretraining
	2. stroke gravity
	
- character recognition: CNN
- character bounding-box: clean out non-character fragments (this is not solved by binarization since the fragment may be from another character)

- strategy regarding segmentation: ignore curved lines
- strategy regarding recognition: no n-grams just simple fully connected or CNN

- made a preliminary schedule
- made slides for monday's presentation




