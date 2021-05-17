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
`
# Download the TensorFlow Serving Docker image and repo
docker pull tensorflow/serving

git clone https://github.com/tensorflow/serving
# Location of demo models
TESTDATA="$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata"

# Start TensorFlow Serving container and open the REST API port
docker run -t --rm -p 8501:8501 \
    -v "$TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two" \
    -e MODEL_NAME=half_plus_two \
    tensorflow/serving &

# Query the model using the predict API
curl -d '{"instances": [1.0, 2.0, 5.0]}' \
    -X POST http://localhost:8501/v1/models/half_plus_two:predict

# Returns => { "predictions": [2.5, 3.0, 4.5] }
`


### Cloning The R
`
Sample Output
{
  "accuracy": [
    "97.41",
    "94.28",
    "95.88",
    "66.87",
    "99.87",
    "96.92"
  ],
  "boxes": [
    "[ymin, xmin, ymax, ymax]",
    "[93, 32, 201, 91]",
    "[91, 158, 196, 213]",
    "[88, 206, 196, 267]",
    "[86, 272, 192, 339]",
    "[85, 337, 189, 393]",
    "[83, 394, 191, 452]"
  ],
  "character": "vadthe",
  "imagename": "tl-horizontal_main.jpg",
  "model": "character detection",
  "processingTime": 0.19191694259643555
}
`
image=.

curl -i -X POST -F model=plate  -F "image=@1.jpg" http://localhost:5000/apiv2

docker run -it -d -p 5000:5000 raymund07/alnpr

docker build -t raymund07/alnpr .
