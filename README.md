# Global Agility Solutions - Automatic License Plate Reader
The purpose of this engagement is to develop an ALPR (Automated License Plate Reader) solution that is
proprietary to and owned by Global Agility Solutions. The functionality of the ALPR software should be
developed per the specifications below.
## General Specifications: 
● The ALPR solutions should be developed utilizing common machine learning algorithms using
transfer learning or similar to establish models for inference. /
● The ALPR solution should provide the following functionality /
* Registration recognition (characters representing the license plate number) /
* Confidence level at the character level (one confidence score per character) /
* Character recognition should include stacked characters /
* Jurisdiction recognition to include /
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
### Training Models: 2/6
* [X] Plate Localization 
* [X] Character Localization and Recognition:
  * [X] Florida,Texas,California ( Current Series )
  * [X] Illinois, Maryland, Massachusetts
  * [X] Michigan, Missouri, North Carolina
  * [X] New Jersey, New York, Ohio
  * [X] Ontario, Pennsylvania, Quebec
  * [X] Virginia, Washington, Wisconsin.
- [X] Jurisdiction (Classifier) ( IN PROGRESS )
- [x] Data Generation
  * [X] DE
- [] State Classification ( Standard Issue, Veterans, Etc)
- [] Stacked Characters (Character Recognition) - Additional Training
- [] Us Teritories and Other Countries (Character Recognition)- Additional Training 

### Deployment  6 / 7
- [X] Create RESTful API ( single Model)
- [X] Create RESTful API ( multiple Model) 
- [X] Create Client App for testing
- [X] Dockerize Application 
- [] Create Docker Compose NGINX,WSGI,Flask( for Production )
- [X] Build Container and Push to dockerhub
- [x] Connect Azure App Service for production 

### Helper and Utilities 5/5
- [x] Annotation and Data Validation ( for data preprocessing)
- [x] Data Augmentation ( for Data preprocesing)
- [X] Create Simple Documentation on how to connect via API
- [X] Automate Data Preparation 
- [X] Data Annotation



## HOW TO INSTALL 

### Installation via Docker

1. git clone https://github.com/raymund07/globalalnpr.git && cd globalalnpr 
2. docker build -t alnpr .
3. docker run -it -d -p 5000:5000 alnpr

### Pulling the image from dockerhub

1. docker pull raymund07/alnpr
2. docker run -it -d -p 5000:5000 alnpr


### cloudrun
1. docker build -t gcr.io/mdta-348812/alnpr .
2. docker tag alnpr:lastest gcr.io/mdta-348812/alnpr
3. docker push



### Let us see some result 
You can test the initial models intalled hosted in the image. Test via local installation or via remote server. Result includes plate location, characters detection and top 10 registration with confidence level See example below \
**Local Environment** \
curl -o 1.jpg http://10.0.99.73:4000/images/roi//5/2021/12/17/15/502_5_20211217152701975_RF_1.jpg?
`curl -i -X POST -F  -F "image=@1.jpg" https://alpr-s4uhkej6la-de.a.run.app/api/v2 ` \
**Remote Testing - Azure App Service** \C:\Users\Isaac\global\training_demo\1.jpg
`curl -i -X POST -F model=plate  -F "image=@images/p3.jpg" https://globalalnpr.azurewebsites.net/api/v2` 





