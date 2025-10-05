import requests
import time
import json
import statistics
from datetime import datetime

def test_response_times(endpoint, num_requests=100):
    """Test API response times"""
    response_times = []
    successful_requests = 0
    failed_requests = 0
    
    print(f"Testing response times with {num_requests} requests...")
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.get(f"{endpoint}/health", timeout=10)
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            response_times.append(response_time)
            successful_requests += 1
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_requests} requests")
                
        except Exception as e:
            failed_requests += 1
            print(f"Request {i + 1} failed: {e}")
    
    if response_times:
        results = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'total_requests': num_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'avg_response_time': statistics.mean(response_times),
            'median_response_time': statistics.median(response_times),
            'p95_response_time': statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
            'p99_response_time': statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times)
        }
    else:
        results = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'total_requests': num_requests,
            'successful_requests': 0,
            'failed_requests': failed_requests,
            'error': 'No successful requests'
        }
    
    return results

if __name__ == "__main__":
    endpoint = "http://localhost:8000"
    results = test_response_times(endpoint)
    
    # Save results
    with open("E:/project/harborai/tests/reports/performance/response_time_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Response time test completed. Results saved.")
    print(f"Average response time: {results.get('avg_response_time', 'N/A')} ms")
