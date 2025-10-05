from locust import HttpUser, task, between
import json
import random

class HarborAIUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts"""
        pass
    
    @task(3)
    def test_health_check(self):
        """Test health check endpoint"""
        self.client.get("/health")
    
    @task(2)
    def test_api_status(self):
        """Test API status endpoint"""
        self.client.get("/api/status")
    
    @task(1)
    def test_api_info(self):
        """Test API info endpoint"""
        self.client.get("/api/info")
    
    @task(1)
    def test_post_data(self):
        """Test POST request with data"""
        data = {
            "test_id": random.randint(1, 1000),
            "message": f"Performance test message {random.randint(1, 100)}"
        }
        self.client.post("/api/test", json=data)
