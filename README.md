# Global Agility Solutions - Automatic License Plate Reader
The purpose of this engagement is to develop an ALPR (Automated License Plate Reader) solution that is
proprietary to and owned by Global Agility Solutions. The functionality of the ALPR software should be
developed per the specifications below.
## General Specifications:
● The ALPR solutions should be developed utilizing common machine learning algorithms using
transfer learning or similar to establish models for inference.
● The ALPR solution should provide the following functionality
o Registration recognition (characters representing the license plate number)
o Confidence level at the character level (one confidence score per character)
o Character recognition should include stacked characters
o Jurisdiction recognition to include
▪ All 50 US states
▪ Common US territories
▪ Mexico
▪ Canada
o Confidence level for Jurisdiction
o Plate Type for common plates as well as specialty plates for each state based on recent
plates identified in Wikipedia (post 1990)
o Confidence level for Plate Type
o ROI information for plate location (top left, top right, bottom left, bottom right
coordinates)

● The ALPR models should be developed using Tensorflow and scripts used for training should be
done in Python or C# using ML.NET
● Inference should be done on a Linux platform and accessible using a RESTful web service
through a JSON or similar interface in order to be accessed from a remote interface.
● Performance will be measured as follows and should meet or exceed the OpenALPR levels based
on a test set of data determined by Global Agility. This test set as well as truth values will be
provided to contractor. Metrics include:
o Speed of inference
o Accuracy
o ‘Hit Rate’
● ALPR engine should be able to be ‘tuned’ based on customer imagery. Images may vary based
on resolution and compression. These tuning parameters will be established based on further
review of the solution and need.
● ALPR engine should be able to return x results based on requested input descending on
confidence.

## Initial Requirement:
### Training Models: 
- [X] Plate Localization 
- [] Character Localization - For data preparation
- [X] Character Recognition
- [] Jurisdiction (Object Detection) - Unique Symbols
- [] Jurisdiction (Classifier)
- [] State Classification ( Standard Issue, Veterans, Etc)
- [] Stacked Characters (Character Recognition) - Additional Training
- [] Us Teritories and Other Countries (Character Recognition)- Additional Training 

### Deployment
- [X] Create RESTful API ( single Model)
- [X] Create RESTful API ( multiple Model) 
- [X] Create Client App for testing
- [X] Dockerize Application 
- [] Create Docker Compose NGINX,WSGI,Flask( for Production )
- [X] Build Container and Push to dockerhub
- [x] Connect Azure App Service for production 

### Helper and Utilities
- [] Annotation and Data Validation ( for data preprocessing)
- [] [in progress ]Data Augmentation ( for Data preprocesing)
- [] Create Simple Documentation on how to connect via API
- [] Automate Data Preparation 
- [] Data Annotation



## HOW TO INSTALL 

### Installation via Docker

1. git clone https://github.com/raymund07/globalalnpr.git && cd alnpr 
2. docker build -t alnpr .
3. docker run -it -d -p 5000:5000 alnpr

### Pulling the image from dockerhub

1. docker pull raymund07/alnpr
2. docker run -it -d -p 5000:5000 alnpr



### Let us see some result 
When you run \
**Local Environment** \
`curl -i -X POST -F model=plate  -F "image=@1.jpg" http://localhost:5000/apiv2 ` \
**Remote Testing - Azure App Service** \
`curl -i -X POST -F model=plate  -F "image=@images/p3.jpg" https://globalalnpr.azurewebsites.net/api/v2` 

![Result](https://github.com/raymund07/globalalnpr/blob/master/application/sample/test.JPG)



