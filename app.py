from flask import Flask, jsonify, request
from threading import Thread
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import queue 

exception_queue = queue.Queue()

app = Flask(__name__)

training_data = None  # Dane treningowe
model = None  # Wytrenowany model
training_in_progress = False  # Flaga informująca o trwającym treningu
training_start_time = None  # Czas rozpoczęcia treningu
training_end_time = None  # Czas zakończenia treningu
exceptions_with_timestamps = [] # Lista z błędami w pamięci
L = 10 # Liczba L odpowiada za ilość części na jaką dzielimy zbiór
K = 1 # Liczba K odpowiada za liczbę sąsiadów

def calculate_distance(X, k):
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    return np.mean(distances[:, 1:], axis=1)

def train_model(data, L, K):
    global model, training_in_progress, training_end_time
    exception_queue.queue.clear()
    try:
        np.random.shuffle(data)
        subsets = np.array_split(data, L)
        relevance_scores = []
        def process_subset(subset):
            nonlocal relevance_scores
            distances = calculate_distance(subset, K)
            relevance_scores = np.concatenate((relevance_scores, 1 / (1 + distances)))
        
        # Uruchamianie wpsółbieżne każdej częsci zbioru
        with ThreadPoolExecutor() as executor:
            executor.map(process_subset, subsets)

        if exception_queue.empty():
            X = np.concatenate(subsets)
            y = np.array(relevance_scores)
            model = RandomForestRegressor()
            model.fit(X, y)
            time.sleep(5)
            training_in_progress = False
            training_end_time = time.time()
    except Exception as e:
        training_in_progress = False
        training_end_time = time.time()
        exception_with_timestamp = (e, time.time())
        exception_queue.put(exception_with_timestamp)
        
@app.route('/train', methods=['POST'])
def train():
    global training_data, training_in_progress, training_start_time, training_end_time, exceptions_with_timestamps
    exceptions_with_timestamps = []

    if training_in_progress:
        return jsonify({'status': 'Uczenie nadal trwa', 'start_time': training_start_time}), 200

    training_data = request.get_json()
    training_in_progress = True
    training_start_time = time.time()

    # Rozpoczęcie treningu w oddzielnym wątku, aby serwis był responsywny
    thread = Thread(target=train_model, args=(training_data, L, K))
    thread.start()

    
    return jsonify({'status': 'Trening rozpoczety', 'start_time': training_start_time}), 200


@app.route('/training_status', methods=['GET'])
def training_status():
    global training_in_progress, training_start_time, training_end_time, exceptions_with_timestamps
    if not exception_queue.empty():
        exceptions_with_timestamps = []
        while not exception_queue.empty():
            exception_with_timestamp = exception_queue.get()
            exceptions_with_timestamps.append(exception_with_timestamp)
    if exceptions_with_timestamps:

        return jsonify({'status': 'Blad podczas uczenia',
                        'start_training_time': training_start_time,
                        'errors': [{'exception': str(e), 'timestamp': ts} for e, ts in exceptions_with_timestamps]}), 500

    if not training_in_progress:
        return jsonify({'status': 'Brak trwajacego treningu',
                        'last_training_start_time': training_start_time,
                        'last_training_end_time': training_end_time
                        }), 200

    return jsonify({'status': 'Uczenie nadal trwa', 'start_time': training_start_time}), 200


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model nie został jeszcze wytrenowany'}), 400

    objects = request.get_json()
    relevance_scores = model.predict(objects)
    return jsonify({'relevance_scores': list(relevance_scores)}), 200


if __name__ == '__main__':
    app.run()
