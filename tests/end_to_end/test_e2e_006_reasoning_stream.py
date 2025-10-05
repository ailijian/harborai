#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2E-006: Reasoning Model Stream Thinking Process Test

This test case focuses on verifying the thinking process of reasoning models in streaming output, including:
1. Separate output of reasoning_content and content
2. Completeness and logical coherence of streaming data
3. Verification of thinking processes specific to reasoning models

Applicable models:
- deepseek-reasoner
- ernie-x1-turbo-32k  
- doubao-seed-1-6-250615

Author: HarborAI Test Team
Created: 2024-12-19
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入HarborAI客户端
try:
    from harborai import HarborAI
    from harborai.core.models import is_reasoning_model, get_model_capabilities
except ImportError as e:
    print(f"❌ 导入HarborAI失败: {e}")
    print("请确保HarborAI包已正确安装")
    sys.exit(1)

# 加载环境变量，优先加载.env.test
from dotenv import load_dotenv

# 优先尝试加载 .env.test 文件
env_test_file = project_root / ".env.test"
env_file = project_root / ".env"

target_file = env_test_file if env_test_file.exists() else env_file
print(f"加载环境变量文件: {target_file}")
load_dotenv(target_file)

class ReasoningStreamTestCase:
    """Reasoning Model Stream Thinking Process Test Case"""
    
    def __init__(self):
        """Initialize test case"""
        self.client = None
        self.test_results = []
        # Reasoning models list
        self.reasoning_models = [
            "deepseek-reasoner",
            "ernie-x1-turbo-32k", 
            "doubao-seed-1-6-250615"
        ]
        
    def setup_client(self) -> bool:
        """Setup HarborAI client"""
        try:
            # Check required environment variables
            required_vars = [
                "DEEPSEEK_API_KEY", "DOUBAO_API_KEY", "WENXIN_API_KEY"
            ]
            
            missing_vars = []
            for var in required_vars:
                value = os.getenv(var)
                if not value:
                    missing_vars.append(var)
                else:
                    print(f"✅ {var}: {value[:10]}...{value[-4:] if len(value) > 14 else value}")
            
            if missing_vars:
                print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
                print("Continuing test, but API key validation may fail...")
            
            # Initialize HarborAI client
            self.client = HarborAI()
            
            print(f"✅ HarborAI client initialized successfully")
            available_models = self.client.get_available_models()
            print(f"📋 Available models: {available_models}")
            
            # Verify reasoning model availability
            for model in self.reasoning_models:
                if model in available_models:
                    print(f"✅ Reasoning model {model} available")
                else:
                    print(f"⚠️  Reasoning model {model} not in available list")
            
            return True
            
        except Exception as e:
            print(f"❌ Client initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_reasoning_stream(self, model: str) -> Dict[str, Any]:
        """Test single reasoning model stream thinking process"""
        print(f"\n🧠 Testing reasoning model: {model}")
        
        test_result = {
            "model": model,
            "success": False,
            "reasoning_chunks_received": 0,
            "content_chunks_received": 0,
            "total_reasoning_content": "",
            "total_content": "",
            "first_reasoning_chunk_time": None,
            "first_content_chunk_time": None,
            "total_time": None,
            "error": None,
            "reasoning_content_valid": False,
            "content_valid": False,
            "logical_coherence": False
        }
        
        try:
            start_time = time.time()
            
            # Construct test message - use complex question to trigger reasoning process
            messages = [
                {
                    "role": "user",
                    "content": "Explain the basic principles of relativity, including the core concepts of special relativity and general relativity and their differences."
                }
            ]
            
            print(f"🔄 Starting streaming call to reasoning model {model}")
            
            # Reasoning model streaming call
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                max_tokens=500
            )
            
            reasoning_parts = []
            content_parts = []
            
            # Process streaming response
            for chunk in stream:
                try:
                    # Get choices
                    choices = None
                    if hasattr(chunk, 'choices'):
                        choices = chunk.choices
                    elif isinstance(chunk, dict) and "choices" in chunk:
                        choices = chunk["choices"]
                    
                    if choices and len(choices) > 0:
                        choice = choices[0]
                        delta = None
                        
                        if hasattr(choice, 'delta'):
                            delta = choice.delta
                        elif isinstance(choice, dict) and "delta" in choice:
                            delta = choice["delta"]
                        
                        if delta:
                            # Collect thinking process (reasoning_content)
                            reasoning_content = None
                            if hasattr(delta, 'reasoning_content'):
                                reasoning_content = delta.reasoning_content
                            elif isinstance(delta, dict) and "reasoning_content" in delta:
                                reasoning_content = delta["reasoning_content"]
                            
                            if reasoning_content:
                                if test_result["first_reasoning_chunk_time"] is None:
                                    test_result["first_reasoning_chunk_time"] = time.time() - start_time
                                
                                reasoning_parts.append(reasoning_content)
                                test_result["reasoning_chunks_received"] += 1
                                print(f"🧠 Reasoning chunk [{test_result['reasoning_chunks_received']}]: {reasoning_content[:80]}{'...' if len(reasoning_content) > 80 else ''}")
                            
                            # Collect final answer (content)
                            content = None
                            if hasattr(delta, 'content'):
                                content = delta.content
                            elif isinstance(delta, dict) and "content" in delta:
                                content = delta["content"]
                            
                            if content:
                                if test_result["first_content_chunk_time"] is None:
                                    test_result["first_content_chunk_time"] = time.time() - start_time
                                
                                content_parts.append(content)
                                test_result["content_chunks_received"] += 1
                                print(f"📝 Content chunk [{test_result['content_chunks_received']}]: {content[:80]}{'...' if len(content) > 80 else ''}")
                                
                except Exception as e:
                    print(f"⚠️  Error processing chunk: {e}")
                    continue
            
            # Merge complete content
            test_result["total_reasoning_content"] = ''.join(reasoning_parts)
            test_result["total_content"] = ''.join(content_parts)
            test_result["total_time"] = time.time() - start_time
            
            # Validate thinking process
            if reasoning_parts:
                test_result["reasoning_content_valid"] = len(test_result["total_reasoning_content"]) > 0
                print(f"🧠 Thinking process validation: length={len(test_result['total_reasoning_content'])} characters, chunks={len(reasoning_parts)}")
            else:
                print(f"⚠️  No reasoning content received (reasoning_content)")
            
            # Validate final answer
            if content_parts:
                test_result["content_valid"] = len(test_result["total_content"]) > 0
                print(f"📝 Final answer validation: length={len(test_result['total_content'])} characters, chunks={len(content_parts)}")
            else:
                print(f"⚠️  No final content received (content)")
            
            # Validate logical coherence - simple keyword check
            reasoning_text = test_result["total_reasoning_content"].lower()
            content_text = test_result["total_content"].lower()
            
            # Check relativity-related keywords
            relativity_keywords = ["relativity", "einstein", "spacetime", "light speed", "gravity", "special", "general"]
            reasoning_has_keywords = any(keyword in reasoning_text for keyword in relativity_keywords)
            content_has_keywords = any(keyword in content_text for keyword in relativity_keywords)
            
            test_result["logical_coherence"] = reasoning_has_keywords and content_has_keywords
            
            # Determine test success conditions
            test_result["success"] = (
                test_result["reasoning_content_valid"] and 
                test_result["content_valid"] and
                test_result["logical_coherence"]
            )
            
            if test_result["success"]:
                print(f"✅ {model} reasoning streaming test successful")
            else:
                print(f"❌ {model} reasoning streaming test failed")
                
            print(f"   - Reasoning chunks: {test_result['reasoning_chunks_received']}")
            print(f"   - Content chunks: {test_result['content_chunks_received']}")
            print(f"   - First reasoning chunk time: {test_result['first_reasoning_chunk_time']:.3f}s" if test_result['first_reasoning_chunk_time'] else "   - First reasoning chunk time: None")
            print(f"   - First content chunk time: {test_result['first_content_chunk_time']:.3f}s" if test_result['first_content_chunk_time'] else "   - First content chunk time: None")
            print(f"   - Total time: {test_result['total_time']:.3f}s")
            print(f"   - Reasoning content length: {len(test_result['total_reasoning_content'])} characters")
            print(f"   - Final answer length: {len(test_result['total_content'])} characters")
            print(f"   - Logical coherence: {test_result['logical_coherence']}")
            
        except Exception as e:
            test_result["error"] = str(e)
            print(f"❌ {model} test failed: {e}")
            import traceback
            traceback.print_exc()
        
        return test_result
    
    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all reasoning model streaming thinking process tests"""
        print("🚀 Starting E2E-006 reasoning model streaming thinking process tests")
        print(f"📋 Number of reasoning models to test: {len(self.reasoning_models)}")
        
        for model in self.reasoning_models:
            # Verify if it's a reasoning model
            if not is_reasoning_model(model):
                print(f"⚠️  {model} is not a reasoning model, skipping test")
                continue
                
            result = self.test_reasoning_stream(model)
            self.test_results.append(result)
            
            # Interval between models to avoid too frequent requests
            time.sleep(2)
        
        return self.test_results
    
    def generate_test_report(self) -> None:
        """Generate test report"""
        print("\n" + "="*80)
        print("📊 E2E-006 Reasoning Model Streaming Thinking Process Test Report")
        print("="*80)
        
        successful_tests = [r for r in self.test_results if r["success"]]
        failed_tests = [r for r in self.test_results if not r["success"]]
        
        print(f"\n📈 Test Overview:")
        print(f"   - Total tests: {len(self.test_results)}")
        print(f"   - Successful: {len(successful_tests)}")
        print(f"   - Failed: {len(failed_tests)}")
        print(f"   - Success rate: {len(successful_tests)/len(self.test_results)*100:.1f}%" if self.test_results else "   - Success rate: 0%")
        
        if successful_tests:
            print(f"\n✅ Successful tests:")
            for result in successful_tests:
                print(f"   - {result['model']}: ")
                print(f"     Reasoning chunks={result['reasoning_chunks_received']}, Content chunks={result['content_chunks_received']}")
                print(f"     Reasoning content={len(result['total_reasoning_content'])} chars, Final answer={len(result['total_content'])} chars")
                print(f"     Total time={result['total_time']:.3f}s, Logical coherence={result['logical_coherence']}")
        
        if failed_tests:
            print(f"\n❌ Failed tests:")
            for result in failed_tests:
                print(f"   - {result['model']}: {result['error'] if result['error'] else 'Validation failed'}")
                if not result['reasoning_content_valid']:
                    print(f"     ⚠️  Invalid reasoning content")
                if not result['content_valid']:
                    print(f"     ⚠️  Invalid final answer")
                if not result['logical_coherence']:
                    print(f"     ⚠️  Insufficient logical coherence")
        
        # Performance statistics
        if successful_tests:
            avg_reasoning_chunks = sum(r["reasoning_chunks_received"] for r in successful_tests) / len(successful_tests)
            avg_content_chunks = sum(r["content_chunks_received"] for r in successful_tests) / len(successful_tests)
            avg_total_time = sum(r["total_time"] for r in successful_tests) / len(successful_tests)
            
            print(f"\n📊 Performance Statistics (Successful tests):")
            print(f"   - Average reasoning chunks: {avg_reasoning_chunks:.1f}")
            print(f"   - Average content chunks: {avg_content_chunks:.1f}")
            print(f"   - Average total time: {avg_total_time:.3f}s")
        
        print("\n" + "="*80)

def main():
    """Main function"""
    print("🔧 E2E-006: Reasoning Model Streaming Thinking Process Test")
    print("="*60)
    
    # Create test instance
    test_case = ReasoningStreamTestCase()
    
    # Setup client
    if not test_case.setup_client():
        print("❌ Test terminated: Client setup failed")
        return
    
    try:
        # Run all tests
        test_case.run_all_tests()
        
        # Generate test report
        test_case.generate_test_report()
        
        # Save test results to JSON file
        results_file = project_root / "tests" / "reports" / "e2e_006_results.json"
        # 确保reports目录存在
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({
                "test_name": "E2E-006 Reasoning Model Streaming Thinking Process Test",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": test_case.test_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Test results saved to: {results_file}")
        
        # Return test results for assertion
        return test_case.test_results
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test execution error: {e}")
        import traceback
        traceback.print_exc()

def test_reasoning_stream():
    """pytest test function"""
    results = main()
    
    # Verify at least one test succeeded
    successful_tests = [r for r in results if r["success"]]
    assert len(successful_tests) > 0, f"All reasoning model tests failed. Failure details: {[r['error'] for r in results if not r['success']]}"
    
    # Verify successful tests have valid reasoning content and final answers
    for result in successful_tests:
        assert result["reasoning_content_valid"], f"Model {result['model']} has invalid reasoning content"
        assert result["content_valid"], f"Model {result['model']} has invalid final answer"
        assert result["logical_coherence"], f"Model {result['model']} has insufficient logical coherence"
    
    print(f"\n✅ Test passed! Successfully tested {len(successful_tests)} reasoning models' streaming thinking process")

if __name__ == "__main__":
    main()