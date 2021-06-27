# Meeting minutes

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

## 2021, May 4, Tuesday

- line segmentation
	- histogram, ignore curvature
	- histograms on split columns
	- histograms on split column rotated inputs
	- A*
	- AruNet

approach: histogram, ignore curvature

- word segmentation

approach: histogram

word-segmentation matters for information preservation
question: do we need whitespace in the output?
(probably yes, transcription)

- character segmentation

approach: sliding window for character segmentation

Todo:
andele and lars focus on line segmentation
mark and rafael focus on character recognition
we plan to use character recognition with a sliding window approach for character segmentation

## 2021, May 9, Sunday

- Save trained character recognition model
- Load trained character recognition model into fragment pipeline
- Fixed import problem
- Discussion of segmentation ideas
- Preparation of Monday presentation
- Plan to merge branches to main

## 2021, May 25, Tuesday

- TODO (Lars): send mail to have presentation on Monday at lab slot.
- Line segmentation works quite well
	- sometimes there are lines which have very small blobs
- We are behind on the integration of line-segmentation and character-recognition
	- TODO (Mark, Rafael) we need to focus on the character-segmentation; sliding window approach
	- Due to the cleaning 
- TODO (Rafael): character augmentation (image morphing)
- Style classifer (Lars, Andele); what approach?
	- Read literature
	- Idea: segment out each character. predict each character's epoch. use the aggregated character style-classes to vote on the period of a document


## 2021, June 9, Wednesday

- Andele, Lars: Style classifier works quite well
- TODOS
	1. Character segmentation
		- split connected characters
		- (Rafael) connectivity analysis for non-connected characters (e.g. trumpets)
		- (Mark) dismiss non-characters
		- size processing (reducing whitespace)
	2. (Lars) Transcription: Writing output in Hebrew font
	3. (Andele) Implementing plots and analysis for recognition
	4. (Lars) Shared report (Methods, Results)
	5. (Andele) Clean code.
		- Formally docstring >10LOC functions
		- Comments

## 2021, June 16, Wednesday

- Mark: maybe we need to abandon dismissing non-character
- Rafael: connectivity analysis works well
	- others: what happens when a good single character is fed as input?
	- others: make surrounding white to remove unwanted char fragments from main char
	- Mark: minimum width of single character to avoid breaking apart damaged characters
- Andele: got started on style and cleanup. using pydocstyle.
- Lars: did transcription

TODO:
- finish segmentation
	- (Rafael) connectivity analysis
	- (Andele, Lars) split connected chars
- image augmentation on train data
	- (Mark) erosion and dilation and elastic morphing on train data
	- goal: balance dataset
- style classifier
	- (Lars) write to text
- style and analysis tools (plots)