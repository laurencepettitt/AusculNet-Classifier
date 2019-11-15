# AusculNet

###### :speaking_head: aw·skuhl·net
AusculNet is a respiratory disease classifier that can predict a diagnosis of several respiratory diseases based on recordings (or "auscultations") from an electronic stethoscope.

## What is the need for AusculNet?

Respiratory disease is a common and significant cause of illness and death around the world. In 2012, respiratory conditions were the most frequent reasons for hospital stays among children.[1] The stethoscope is an acoustic medical device for listening to the internal sounds of a human body. It is often used to detect indicators of respiratory diseases. Studies have shown that auscultation skill (i.e., the ability to make a diagnosis based on what is heard through a stethoscope) has been in decline for some time, such that some medical educators are working to re-establish it.[2][3][4] Here we present an alternative that would require almost no human training. AusculNet is a respiratory disease classifier that takes recordings from an electric stethoscope to make a prediction for eight diagnoses, including healthy. The limited training required would make our method cheaper than a human physician. Moreover, results could be sent digitally for further analysis in the case of a positive diagnosis. All of these features are helpful for purposes of telemedicine (remote diagnosis). Physician’s diagnosis of pneumonia by auscultation had a sensitivity of around 60%.[5] AusculNet has a sensitivity of around 55%.

## Installation

### For usage
AusculNet is not yet ready for production usage.

### For development
Clone the repository to your local environment and change directory into it.
```
git clone git@github.com:laurencepettitt/AusculNet-Classifier.git && cd AusculNet-Classifier
```
Create a virtualenv for python3 in the project's root folder and activate it.
```
virtualenv -p python3 venv && source venv/bin/activate
```
Install the project's dependencies
```
pip install --editable .
```

## Usage

You may run `python ausculnet/training/dnn.py` to run an experiment on training the model yourself and see the results.

## Acknowledgements

Many thanks to the research teams at the University of Coimbra, Portugal; the University de Aveiro, Portugal and the Aristotle University of Thessaloniki, Greece for making the respiratory sounds dataset publicly available.

## References

1. Witt WP, Wiess AJ, Elixhauser A (December 2014). "Overview of Hospital Stays for Children in the United States, 2012". HCUP Statistical Brief #186. Rockville, MD: Agency for Healthcare Research and Quality.
2. Wilkins, RL (2004), "Is the stethoscope on the verge of becoming obsolete?", Respir Care, 49 (12): 1488–1489, PMID 15571638.
3. Murphy, R (2005), "The stethoscope – obsolescence or marriage?", Respir Care, 50 (5): 660–661.
4. Bernstein, Lenny (2016-01-02), "Heart doctors are listening for clues to the future of their stethoscopes", Washington Post.
5. Veterans Affairs Puget Sound Health Care System (1999), “Diagnosing pneumonia by physical examination: relevant or relic?”,  PMID 10335685
