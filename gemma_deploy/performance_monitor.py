#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控和优化工具
================

用于监控Gemma服务的性能指标，包括：
- GPU利用率
- 内存使用情况
- 推理延迟
- 吞吐量统计
"""

import time
import psutil
import requests
import json
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, service_url: str = "http://localhost:8000"):
        self.service_url = service_url
        self.metrics = []
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """获取GPU信息"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info = {}
                for i in range(gpu_count):
                    gpu_info[f"gpu_{i}"] = {
                        "name": torch.cuda.get_device_name(i),
                        "memory_allocated": torch.cuda.memory_allocated(i),
                        "memory_reserved": torch.cuda.memory_reserved(i),
                        "memory_total": torch.cuda.get_device_properties(i).total_memory,
                        "utilization": self._get_gpu_utilization(i)
                    }
                return gpu_info
        except ImportError:
            logger.warning("PyTorch not available, cannot get GPU info")
        return {}
    
    def _get_gpu_utilization(self, device_id: int) -> float:
        """获取GPU利用率"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu
        except ImportError:
            logger.warning("pynvml not available, cannot get GPU utilization")
            return 0.0
        except Exception as e:
            logger.error(f"Error getting GPU utilization: {e}")
            return 0.0
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available": psutil.virtual_memory().available,
            "memory_total": psutil.virtual_memory().total,
            "disk_usage": psutil.disk_usage('/').percent
        }
    
    def test_inference_speed(self, test_prompts: List[str]) -> Dict[str, Any]:
        """测试推理速度"""
        results = []
        
        for prompt in test_prompts:
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.service_url}/chat",
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 512,
                        "stream": False
                    },
                    timeout=30
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    results.append({
                        "prompt_length": len(prompt),
                        "response_length": len(data.get("response", "")),
                        "latency": end_time - start_time,
                        "tokens_per_second": data.get("usage", {}).get("total_tokens", 0) / (end_time - start_time),
                        "success": True
                    })
                else:
                    results.append({
                        "prompt_length": len(prompt),
                        "latency": end_time - start_time,
                        "success": False,
                        "error": response.text
                    })
            except Exception as e:
                end_time = time.time()
                results.append({
                    "prompt_length": len(prompt),
                    "latency": end_time - start_time,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "total_tests": len(test_prompts),
            "successful_tests": sum(1 for r in results if r["success"]),
            "average_latency": sum(r["latency"] for r in results) / len(results),
            "average_tokens_per_second": sum(r.get("tokens_per_second", 0) for r in results if r["success"]) / max(1, sum(1 for r in results if r["success"])),
            "results": results
        }
    
    def get_service_health(self) -> Dict[str, Any]:
        """获取服务健康状态"""
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "unhealthy", "error": response.text}
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}
    
    def collect_metrics(self) -> Dict[str, Any]:
        """收集所有性能指标"""
        metrics = {
            "timestamp": time.time(),
            "gpu_info": self.get_gpu_info(),
            "system_info": self.get_system_info(),
            "service_health": self.get_service_health()
        }
        
        self.metrics.append(metrics)
        return metrics
    
    def generate_report(self) -> str:
        """生成性能报告"""
        if not self.metrics:
            return "No metrics collected yet."
        
        latest = self.metrics[-1]
        report = f"""
=== 性能监控报告 ===
时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest['timestamp']))}

系统状态:
- CPU使用率: {latest['system_info']['cpu_percent']:.1f}%
- 内存使用率: {latest['system_info']['memory_percent']:.1f}%
- 可用内存: {latest['system_info']['memory_available'] / 1024**3:.1f} GB

GPU状态:
"""
        
        for gpu_id, gpu_info in latest['gpu_info'].items():
            memory_used = gpu_info['memory_allocated'] / 1024**3
            memory_total = gpu_info['memory_total'] / 1024**3
            report += f"- {gpu_id}: {gpu_info['name']}\n"
            report += f"  内存使用: {memory_used:.1f}/{memory_total:.1f} GB ({memory_used/memory_total*100:.1f}%)\n"
            report += f"  GPU利用率: {gpu_info['utilization']:.1f}%\n"
        
        report += f"""
服务状态:
- 状态: {latest['service_health'].get('status', 'unknown')}
- 模型加载: {latest['service_health'].get('model_loaded', False)}
"""
        
        return report

def main():
    """主函数"""
    monitor = PerformanceMonitor()
    
    print("开始性能监控...")
    print("按 Ctrl+C 停止监控")
    
    try:
        while True:
            metrics = monitor.collect_metrics()
            print(f"\r{time.strftime('%H:%M:%S')} - CPU: {metrics['system_info']['cpu_percent']:.1f}% - 内存: {metrics['system_info']['memory_percent']:.1f}%", end="")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\n生成性能报告...")
        print(monitor.generate_report())

if __name__ == "__main__":
    main()
