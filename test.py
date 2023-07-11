import unittest
import json
from app import app, calculate_distance

class FlaskTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_calculate_distance(self):
        data = [[1, 2], [1, 3], [1, 5]]
        distances = calculate_distance(data, 1)
        self.assertEqual(distances.tolist(), [1, 1, 2])

    def test_status_endpoint(self):
        response = self.client.get('/training_status')
        data = json.loads(response.data.decode())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'Brak trwajacego treningu')

    def test_train_endpoint(self):
        training_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        data_json = json.dumps(training_data)
    
        response = self.client.post('/train', data=data_json, content_type='application/json')
        data = json.loads(response.data.decode())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'Trening rozpoczety')
        self.assertIsNotNone(data['start_time'])




if __name__ == '__main__':
    unittest.main()
