import psutil
import time
import json
import requests
from datetime import datetime

def monitor_memory(duration, endpoint):
    """Monitor memory usage during API calls"""
    start_time = time.time()
    memory_data = []
    
    while time.time() - start_time < duration:
        # Get current memory usage
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        # Make API call
        try:
            response = requests.get(f"{endpoint}/health", timeout=5)
            response_time = response.elapsed.total_seconds()
        except Exception as e:
            response_time = -1
        
        memory_data.append({
            'timestamp': datetime.now().isoformat(),
            'memory_percent': memory_info.percent,
            'memory_available': memory_info.available,
            'memory_used': memory_info.used,
            'cpu_percent': cpu_percent,
            'response_time': response_time
        })
        
        time.sleep(1)
    
    return memory_data

if __name__ == "__main__":
    endpoint = "http://localhost:8000"
    duration = 300
    
    print(f"Starting memory monitoring for {duration} seconds...")
    data = monitor_memory(duration, endpoint)
    
    # Save results
    with open("E:/project/harborai/tests/reports/performance/memory_test_results.json", 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Memory test completed. Results saved.")
