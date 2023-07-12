# Flask Representativeness Service
This is a Flask-based web service that provides machine learning functionalities for training and predicting representativeness scores. The service utilizes the scikit-learn library to implement a Random Forest Regression model and a Nearest Neighbors algorithm.

### Getting Started
To set up and run the Flask ML Service, follow the steps below:
1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo.git
```
3. Install the required dependencies:
```bash
pip install flask scikit-learn numpy
```
5. Start the Flask ML Service:
```bash
python app.py
```


### Endpoints
#### Train the Model
##### Endpoint: 'POST /train'
Trains the machine learning model with the provided training data.
##### Request:
* Method: POST
* Content-Type: application/json
* Body:
    - data (array): An array of training data.

##### Response:
* Status Code: 200 if training is successfully started.
* Status Code: 200 with the following JSON payload if training is already in progress:
```json
{
  "status": "Uczenie nadal trwa",
  "start_time": <start_time>
}
```
> - start_time (float): The timestamp when the training started.

---

#### Check Training Status
##### Endpoint: GET /training_status

Returns the status of the current training process.

##### Request:
* Method: GET
##### Response:

* Status Code: 200 if there is no ongoing training.
* Status Code: 200 with the following JSON payload if training has completed:
```json
{
  "status": "Brak trwajacego treningu",
  "last_training_start_time": <start_time>,
  "last_training_end_time": <end_time>
}
```
> - last_training_start_time (float): The timestamp when the last training started.
> - last_training_end_time (float): The timestamp when the last training ended.

- Status Code: 500 with the following JSON payload if an error occurred during training:
```json
{
  "status": "Blad podczas uczenia",
  "start_training_time": <start_time>,
  "errors": [
    {
      "exception": <exception_message>,
      "timestamp": <timestamp>
    },
    ...
  ]
}
```
> - start_training_time (float): The timestamp when the training started.
> - errors (array): An array of error messages and their corresponding timestamps.

---
#### Predict Relevance Scores
##### Endpoint: POST /predict

Predicts the relevance scores for the given objects using the trained model.

##### Request:

* Method: POST
* Content-Type: application/json
* Body:
  * objects (array): An array of objects for which relevance scores should be predicted.
##### Response:

* Status Code: 200 with the following JSON payload:
```json
{
  "relevance_scores": [<score1>, <score2>, ...]
}
```
> - relevance_scores (array): An array of predicted relevance scores.

---
### Additional Information
- The training data should be provided as an array of objects in the request body for the /train endpoint.
- The /train endpoint starts the training process in a separate thread to keep the service responsive.
- The training process divides the data into L subsets and performs parallel processing on each subset using a thread pool.
- The K parameter controls the number of neighbors used in the Nearest Neighbors algorithm.
- Once training is complete, the trained model can be used for predicting relevance scores using the /predict endpoint.
- If there was an error during training, the /training_status endpoint will return the error details along with the timestamp.
- The service utilizes a queue (exception_queue) to store and retrieve any exceptions that occur during training.


