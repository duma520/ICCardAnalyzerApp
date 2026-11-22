#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# IC卡数据分析系统完整实现 v1.0
# 作者: DeepSeek Chat
# 日期: 2023-11-15

import os
import sys
import time
import logging
import traceback
import hashlib
import threading
import queue
import multiprocessing
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from enum import Enum, auto
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import yaml
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import weasyprint
import psutil
import GPUtil

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ic_card_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置常量
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_FILE_COUNT = 1000
SUPPORTED_FILE_TYPES = ['.bin', '.hex', '.dump']
DEFAULT_CONFIG = {
    'system': {
        'max_files': 1000,
        'auto_save': True,
        'dark_mode': False,
        'timeout': 30,
        'max_threads': 4,
        'max_processes': 2
    },
    'crypto': {
        'aes_bruteforce': False,
        'mifare_timeout': 30,
        'max_key_attempts': 10000
    },
    'clustering': {
        'max_clusters': 10,
        'eps': 0.5,
        'min_samples': 5
    },
    'performance': {
        'use_gpu': True,
        'batch_size': 1024,
        'cache_size': 10000
    }
}

class AnalysisAlgorithm(Enum):
    """分析算法枚举"""
    AES_KEY_RECOVERY = auto()
    MIFARE_CRACK = auto()
    ROLLING_CODE_PREDICTION = auto()
    CLUSTERING = auto()
    ANOMALY_DETECTION = auto()

@dataclass
class AnalysisResult:
    """分析结果数据类"""
    success: bool
    data: Any
    message: str
    execution_time: float
    metadata: Dict[str, Any] = None

class ICDataValidator:
    """IC卡数据验证器"""
    
    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """验证文件路径是否合法"""
        if not file_path:
            logger.warning("文件路径为空")
            return False
        
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return False
            
        if not os.path.isfile(file_path):
            logger.error(f"路径不是文件: {file_path}")
            return False
            
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            logger.error(f"文件大小超过限制: {file_path} ({file_size} bytes)")
            return False
            
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in SUPPORTED_FILE_TYPES:
            logger.error(f"不支持的文件类型: {ext}")
            return False
            
        return True
    
    @staticmethod
    def validate_dump_data(data: np.ndarray) -> bool:
        """验证dump数据是否合法"""
        if data is None:
            logger.warning("数据为空")
            return False
            
        if not isinstance(data, np.ndarray):
            logger.error("数据不是numpy数组")
            return False
            
        if data.size == 0:
            logger.warning("数据为空数组")
            return False
            
        if data.dtype != np.uint8:
            logger.error("数据不是uint8类型")
            return False
            
        return True

class ICDataLoader:
    """IC卡数据加载器"""
    
    def __init__(self):
        self.cache = {}
        self.cache_size = DEFAULT_CONFIG['performance']['cache_size']
        self.lock = threading.Lock()
        
    def load_file(self, file_path: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """加载IC卡dump文件"""
        if not ICDataValidator.validate_file_path(file_path):
            return None
            
        cache_key = hashlib.md5(file_path.encode()).hexdigest()
        
        # 检查缓存
        if use_cache and cache_key in self.cache:
            logger.debug(f"从缓存加载文件: {file_path}")
            return self.cache[cache_key]
            
        try:
            # 使用内存映射文件减少内存占用
            data = np.memmap(file_path, dtype=np.uint8, mode='r')
            data = np.array(data)  # 转换为普通数组
            
            if not ICDataValidator.validate_dump_data(data):
                return None
                
            # 更新缓存
            with self.lock:
                if len(self.cache) >= self.cache_size:
                    self.cache.popitem()  # 移除最旧的缓存
                self.cache[cache_key] = data
                
            return data
        except Exception as e:
            logger.error(f"加载文件失败: {file_path}, 错误: {str(e)}")
            logger.error(traceback.format_exc())
            return None
            
    def batch_load_files(self, file_paths: List[str], progress_callback: Callable = None) -> Dict[str, np.ndarray]:
        """批量加载文件"""
        if not file_paths:
            return {}
            
        if len(file_paths) > MAX_FILE_COUNT:
            logger.warning(f"文件数量超过限制: {len(file_paths)} > {MAX_FILE_COUNT}")
            file_paths = file_paths[:MAX_FILE_COUNT]
            
        results = {}
        with ThreadPoolExecutor(max_workers=DEFAULT_CONFIG['system']['max_threads']) as executor:
            futures = {executor.submit(self.load_file, fp): fp for fp in file_paths}
            
            for i, future in enumerate(as_completed(futures)):
                file_path = futures[future]
                try:
                    data = future.result()
                    if data is not None:
                        results[file_path] = data
                except Exception as e:
                    logger.error(f"加载文件失败: {file_path}, 错误: {str(e)}")
                    
                if progress_callback:
                    progress_callback((i + 1) / len(file_paths))
                    
        return results

class ICDataPreprocessor:
    """IC卡数据预处理器"""
    
    @staticmethod
    def clean_data(data: np.ndarray) -> Optional[np.ndarray]:
        """数据清洗"""
        if not ICDataValidator.validate_dump_data(data):
            return None
            
        try:
            # 移除全0或全FF的块
            cleaned = []
            for i in range(0, len(data), 16):
                block = data[i:i+16]
                if not (np.all(block == 0) and not (np.all(block == 0xFF)):
                    cleaned.extend(block)
                    
            if not cleaned:
                logger.warning("清洗后数据为空")
                return None
                
            return np.array(cleaned, dtype=np.uint8)
        except Exception as e:
            logger.error(f"数据清洗失败: {str(e)}")
            logger.error(traceback.format_exc())
            return None
            
    @staticmethod
    def extract_features(data: np.ndarray, ngram_size: int = 3) -> Optional[pd.DataFrame]:
        """特征提取"""
        if not ICDataValidator.validate_dump_data(data):
            return None
            
        try:
            features = []
            
            # 1. N-gram特征
            for i in range(len(data) - ngram_size + 1):
                ngram = data[i:i+ngram_size]
                features.append({
                    'position': i,
                    'ngram': '-'.join(str(x) for x in ngram),
                    'mean': np.mean(ngram),
                    'std': np.std(ngram),
                    'entropy': ICDataPreprocessor.calculate_entropy(ngram)
                })
                
            # 2. 滑动窗口统计特征
            window_size = 16
            for i in range(0, len(data), window_size):
                window = data[i:i+window_size]
                if len(window) < window_size:
                    continue
                    
                features.append({
                    'position': i,
                    'window_mean': np.mean(window),
                    'window_std': np.std(window),
                    'window_min': np.min(window),
                    'window_max': np.max(window),
                    'window_entropy': ICDataPreprocessor.calculate_entropy(window)
                })
                
            # 3. 整体统计特征
            features.append({
                'position': -1,  # 表示整体特征
                'total_mean': np.mean(data),
                'total_std': np.std(data),
                'total_min': np.min(data),
                'total_max': np.max(data),
                'total_entropy': ICDataPreprocessor.calculate_entropy(data),
                'zero_ratio': np.sum(data == 0) / len(data),
                'ff_ratio': np.sum(data == 0xFF) / len(data)
            })
            
            return pd.DataFrame(features)
        except Exception as e:
            logger.error(f"特征提取失败: {str(e)}")
            logger.error(traceback.format_exc())
            return None
            
    @staticmethod
    def calculate_entropy(data: np.ndarray) -> float:
        """计算熵值"""
        if len(data) == 0:
            return 0.0
            
        value_counts = np.bincount(data)
        probabilities = value_counts / len(data)
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        return entropy

class ICDataAnalyzer(ABC):
    """IC卡数据分析器抽象基类"""
    
    def __init__(self):
        self.timeout = DEFAULT_CONFIG['system']['timeout']
        self.last_result = None
        
    @abstractmethod
    def analyze(self, data: Union[np.ndarray, List[np.ndarray]]) -> AnalysisResult:
        """分析数据"""
        pass
        
    def set_timeout(self, timeout: int):
        """设置超时时间"""
        self.timeout = timeout
        
    def get_last_result(self) -> Optional[AnalysisResult]:
        """获取上次分析结果"""
        return self.last_result

class AESKeyRecoveryAnalyzer(ICDataAnalyzer):
    """AES密钥恢复分析器"""
    
    def __init__(self):
        super().__init__()
        self.max_attempts = DEFAULT_CONFIG['crypto']['max_key_attempts']
        
    def analyze(self, data: Union[np.ndarray, List[np.ndarray]]) -> AnalysisResult:
        """尝试恢复AES密钥"""
        start_time = time.time()
        
        if isinstance(data, np.ndarray):
            data = [data]
            
        if not data or len(data) < 3:
            return AnalysisResult(
                success=False,
                data=None,
                message="至少需要3组数据才能进行密钥恢复",
                execution_time=0
            )
            
        try:
            # 简化版密钥恢复，实际应用中应使用更安全的算法
            for i in range(min(self.max_attempts, 0xFFFF)):
                key = i.to_bytes(16, 'big')
                cipher = AES.new(key, AES.MODE_ECB)
                
                # 检查前3个数据块是否能解密
                match_count = 0
                for d in data[:3]:
                    try:
                        decrypted = cipher.decrypt(d[:16].tobytes())
                        if decrypted.startswith(b'\x00') or decrypted.startswith(b'\xFF'):
                            match_count += 1
                    except:
                        break
                        
                if match_count >= 2:  # 至少2个匹配
                    self.last_result = AnalysisResult(
                        success=True,
                        data=key.hex(),
                        message=f"找到可能的AES密钥: {key.hex()}",
                        execution_time=time.time() - start_time,
                        metadata={'attempts': i + 1}
                    )
                    return self.last_result
                    
            self.last_result = AnalysisResult(
                success=False,
                data=None,
                message=f"在{self.max_attempts}次尝试内未找到AES密钥",
                execution_time=time.time() - start_time,
                metadata={'attempts': self.max_attempts}
            )
            return self.last_result
        except Exception as e:
            logger.error(f"AES密钥恢复失败: {str(e)}")
            logger.error(traceback.format_exc())
            return AnalysisResult(
                success=False,
                data=None,
                message=f"AES密钥恢复失败: {str(e)}",
                execution_time=time.time() - start_time
            )

class RollingCodePredictor(ICDataAnalyzer):
    """滚动码预测分析器"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.batch_size = DEFAULT_CONFIG['performance']['batch_size']
        
    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """构建LSTM预测模型"""
        inputs = Input(shape=input_shape)
        x = LSTM(32, return_sequences=True)(inputs)
        x = LSTM(16)(x)
        outputs = Dense(8, activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
        
    def analyze(self, data: Union[np.ndarray, List[np.ndarray]]) -> AnalysisResult:
        """预测滚动码"""
        start_time = time.time()
        
        if isinstance(data, np.ndarray):
            data = [data]
            
        if not data or len(data) < 10:
            return AnalysisResult(
                success=False,
                data=None,
                message="至少需要10组数据才能进行滚动码预测",
                execution_time=0
            )
            
        try:
            # 准备训练数据
            X = []
            y = []
            for i in range(len(data) - 1):
                X.append(data[i][:8])  # 取前8字节作为输入
                y.append(data[i+1][:8])  # 下一个数据的前8字节作为输出
                
            X = np.array(X).reshape(-1, 1, 8)
            y = np.array(y)
            
            # 构建和训练模型
            if self.model is None:
                self.model = self.build_model((1, 8))
                
            self.model.fit(
                X, y,
                epochs=50,
                batch_size=self.batch_size,
                verbose=0
            )
            
            # 预测下一个值
            last_data = data[-1][:8].reshape(1, 1, 8)
            prediction = self.model.predict(last_data, verbose=0)
            
            self.last_result = AnalysisResult(
                success=True,
                data=prediction[0],
                message="滚动码预测完成",
                execution_time=time.time() - start_time,
                metadata={'model_summary': self._get_model_summary()}
            )
            return self.last_result
        except Exception as e:
            logger.error(f"滚动码预测失败: {str(e)}")
            logger.error(traceback.format_exc())
            return AnalysisResult(
                success=False,
                data=None,
                message=f"滚动码预测失败: {str(e)}",
                execution_time=time.time() - start_time
            )
            
    def _get_model_summary(self) -> str:
        """获取模型摘要信息"""
        if self.model is None:
            return "模型未初始化"
            
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))
        return '\n'.join(summary)

class ClusterAnalyzer(ICDataAnalyzer):
    """聚类分析器"""
    
    def __init__(self):
        super().__init__()
        self.max_clusters = DEFAULT_CONFIG['clustering']['max_clusters']
        self.eps = DEFAULT_CONFIG['clustering']['eps']
        self.min_samples = DEFAULT_CONFIG['clustering']['min_samples']
        
    def analyze(self, data: Union[np.ndarray, List[np.ndarray]]) -> AnalysisResult:
        """执行聚类分析"""
        start_time = time.time()
        
        if isinstance(data, np.ndarray):
            data = [data]
            
        if not data:
            return AnalysisResult(
                success=False,
                data=None,
                message="没有数据可用于聚类分析",
                execution_time=0
            )
            
        try:
            # 提取特征
            features = []
            for d in data:
                if not ICDataValidator.validate_dump_data(d):
                    continue
                    
                df = ICDataPreprocessor.extract_features(d)
                if df is not None:
                    # 使用整体统计特征
                    overall = df[df['position'] == -1].drop(columns=['position', 'ngram'])
                    features.append(overall.values.flatten())
                    
            if len(features) < 3:
                return AnalysisResult(
                    success=False,
                    data=None,
                    message="至少需要3组有效数据才能进行聚类分析",
                    execution_time=time.time() - start_time
                )
                
            # 标准化特征
            features = np.array(features)
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # PCA降维
            pca = PCA(n_components=3)
            reduced_features = pca.fit_transform(scaled_features)
            
            # 确定最佳聚类数
            best_k = 1
            best_score = -1
            silhouette_scores = []
            
            for k in range(2, min(self.max_clusters, len(reduced_features))):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(reduced_features)
                score = silhouette_score(reduced_features, labels)
                silhouette_scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    
            # 使用最佳k值进行KMeans聚类
            kmeans = KMeans(n_clusters=best_k, random_state=42)
            kmeans_labels = kmeans.fit_predict(reduced_features)
            
            # 使用DBSCAN聚类
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            dbscan_labels = dbscan.fit_predict(reduced_features)
            
            # 准备结果
            result_data = {
                'features': reduced_features,
                'kmeans_labels': kmeans_labels,
                'dbscan_labels': dbscan_labels,
                'best_k': best_k,
                'silhouette_scores': silhouette_scores,
                'feature_names': ['total_mean', 'total_std', 'total_min', 'total_max', 
                                  'total_entropy', 'zero_ratio', 'ff_ratio']
            }
            
            self.last_result = AnalysisResult(
                success=True,
                data=result_data,
                message=f"聚类分析完成，最佳聚类数: {best_k}",
                execution_time=time.time() - start_time,
                metadata={
                    'best_k': best_k,
                    'silhouette_score': best_score,
                    'n_samples': len(reduced_features)
                }
            )
            return self.last_result
        except Exception as e:
            logger.error(f"聚类分析失败: {str(e)}")
            logger.error(traceback.format_exc())
            return AnalysisResult(
                success=False,
                data=None,
                message=f"聚类分析失败: {str(e)}",
                execution_time=time.time() - start_time
            )

class AnalysisTask(threading.Thread):
    """分析任务线程"""
    
    def __init__(self, analyzer: ICDataAnalyzer, data: Any, callback: Callable):
        super().__init__()
        self.analyzer = analyzer
        self.data = data
        self.callback = callback
        self.result = None
        self._stop_event = threading.Event()
        
    def run(self):
        """执行分析任务"""
        try:
            self.result = self.analyzer.analyze(self.data)
        except Exception as e:
            logger.error(f"分析任务执行失败: {str(e)}")
            logger.error(traceback.format_exc())
            self.result = AnalysisResult(
                success=False,
                data=None,
                message=f"分析任务执行失败: {str(e)}",
                execution_time=0
            )
        finally:
            if not self._stop_event.is_set():
                self.callback(self.result)
                
    def stop(self):
        """停止任务"""
        self._stop_event.set()

class AnalysisEngine:
    """分析引擎"""
    
    def __init__(self):
        self.analyzers = {
            AnalysisAlgorithm.AES_KEY_RECOVERY: AESKeyRecoveryAnalyzer(),
            AnalysisAlgorithm.ROLLING_CODE_PREDICTION: RollingCodePredictor(),
            AnalysisAlgorithm.CLUSTERING: ClusterAnalyzer()
        }
        self.task_queue = queue.Queue()
        self.worker_thread = None
        self.current_task = None
        
    def start_worker(self):
        """启动工作线程"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._process_tasks, daemon=True)
            self.worker_thread.start()
            
    def _process_tasks(self):
        """处理任务队列中的任务"""
        while True:
            task = self.task_queue.get()
            if task is None:  # 退出信号
                break
                
            self.current_task = task
            task.start()
            task.join()
            self.current_task = None
            self.task_queue.task_done()
            
    def submit_task(self, algorithm: AnalysisAlgorithm, data: Any, callback: Callable):
        """提交分析任务"""
        if algorithm not in self.analyzers:
            logger.error(f"不支持的分析算法: {algorithm}")
            callback(AnalysisResult(
                success=False,
                data=None,
                message=f"不支持的分析算法: {algorithm}",
                execution_time=0
            ))
            return
            
        task = AnalysisTask(self.analyzers[algorithm], data, callback)
        self.task_queue.put(task)
        self.start_worker()
        
    def stop_current_task(self):
        """停止当前任务"""
        if self.current_task:
            self.current_task.stop()
            
    def shutdown(self):
        """关闭引擎"""
        self.stop_current_task()
        self.task_queue.put(None)  # 发送退出信号
        if self.worker_thread:
            self.worker_thread.join(timeout=1)

class ReportGenerator:
    """报告生成器"""
    
    @staticmethod
    def generate_html_report(results: Dict[str, AnalysisResult], title: str = "IC卡分析报告") -> str:
        """生成HTML格式报告"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .success {{ color: #27ae60; }}
                .failure {{ color: #e74c3c; }}
                .metadata {{ background-color: #f9f9f9; padding: 10px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        for section, result in results.items():
            status_class = "success" if result.success else "failure"
            status_text = "成功" if result.success else "失败"
            
            html += f"""
            <h2>{section}</h2>
            <table>
                <tr>
                    <th>状态</th>
                    <td><span class="{status_class}">{status_text}</span></td>
                </tr>
                <tr>
                    <th>执行时间</th>
                    <td>{result.execution_time:.2f} 秒</td>
                </tr>
                <tr>
                    <th>消息</th>
                    <td>{result.message}</td>
                </tr>
            </table>
            """
            
            if result.metadata:
                html += "<h3>元数据</h3><div class='metadata'>"
                for key, value in result.metadata.items():
                    html += f"<p><strong>{key}:</strong> {value}</p>"
                html += "</div>"
                
        html += """
        </body>
        </html>
        """
        
        return html
        
    @staticmethod
    def generate_pdf_report(html_content: str, output_path: str = "report.pdf") -> bool:
        """生成PDF报告"""
        try:
            weasyprint.HTML(string=html_content).write_pdf(output_path)
            return True
        except Exception as e:
            logger.error(f"生成PDF报告失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False

class ResourceMonitor:
    """资源监视器"""
    
    def __init__(self):
        self.running = False
        self.thread = None
        self.callback = None
        self.interval = 1  # 秒
        
    def start(self, callback: Callable):
        """启动资源监视"""
        if self.running:
            return
            
        self.callback = callback
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        
    def stop(self):
        """停止资源监视"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
            
    def _monitor(self):
        """监视资源使用情况"""
        while self.running:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=None)
                
                # 内存使用
                memory = psutil.virtual_memory()
                
                # GPU使用率 (如果可用)
                gpu_info = []
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        gpu_info.append({
                            'id': gpu.id,
                            'load': gpu.load * 100,
                            'memory_used': gpu.memoryUsed,
                            'memory_total': gpu.memoryTotal
                        })
                except:
                    pass
                    
                # 磁盘使用
                disk = psutil.disk_usage('/')
                
                # 网络IO
                net_io = psutil.net_io_counters()
                
                # 准备数据
                data = {
                    'cpu': cpu_percent,
                    'memory': {
                        'total': memory.total,
                        'available': memory.available,
                        'used': memory.used,
                        'percent': memory.percent
                    },
                    'gpu': gpu_info,
                    'disk': {
                        'total': disk.total,
                        'used': disk.used,
                        'free': disk.free,
                        'percent': disk.percent
                    },
                    'network': {
                        'bytes_sent': net_io.bytes_sent,
                        'bytes_recv': net_io.bytes_recv
                    },
                    'timestamp': time.time()
                }
                
                if self.callback:
                    self.callback(data)
                    
            except Exception as e:
                logger.error(f"资源监视出错: {str(e)}")
                
            time.sleep(self.interval)

class PluginManager:
    """插件管理器"""
    
    def __init__(self):
        self.plugins = {}
        self.plugin_dir = "plugins"
        self.load_plugins()
        
    def load_plugins(self):
        """加载所有插件"""
        if not os.path.exists(self.plugin_dir):
            os.makedirs(self.plugin_dir)
            return
            
        for filename in os.listdir(self.plugin_dir):
            if filename.endswith('.py') and not filename.startswith('_'):
                plugin_name = filename[:-3]
                try:
                    module = __import__(f"{self.plugin_dir}.{plugin_name}", fromlist=[''])
                    if hasattr(module, 'Plugin'):
                        plugin_class = getattr(module, 'Plugin')
                        self.plugins[plugin_name] = plugin_class()
                        logger.info(f"加载插件: {plugin_name}")
                except Exception as e:
                    logger.error(f"加载插件 {plugin_name} 失败: {str(e)}")
                    logger.error(traceback.format_exc())
                    
    def get_plugin(self, name: str) -> Optional[Any]:
        """获取指定插件"""
        return self.plugins.get(name)
        
    def list_plugins(self) -> List[str]:
        """列出所有可用插件"""
        return list(self.plugins.keys())
        
    def execute_plugin(self, name: str, data: Any) -> Optional[AnalysisResult]:
        """执行插件"""
        plugin = self.get_plugin(name)
        if plugin is None:
            return None
            
        try:
            start_time = time.time()
            result = plugin.process(data)
            return AnalysisResult(
                success=True,
                data=result,
                message=f"插件 {name} 执行成功",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            logger.error(f"执行插件 {name} 失败: {str(e)}")
            logger.error(traceback.format_exc())
            return AnalysisResult(
                success=False,
                data=None,
                message=f"插件 {name} 执行失败: {str(e)}",
                execution_time=time.time() - start_time
            )

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = 'config.yaml'):
        self.config_file = config_file
        self.config = DEFAULT_CONFIG
        self.load_config()
        
    def load_config(self) -> bool:
        """加载配置文件"""
        if not os.path.exists(self.config_file):
            logger.warning(f"配置文件 {self.config_file} 不存在，使用默认配置")
            return False
            
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f) or {}
                self._merge_configs(loaded_config)
            logger.info(f"从 {self.config_file} 加载配置成功")
            return True
        except Exception as e:
            logger.error(f"加载配置文件 {self.config_file} 失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    def save_config(self) -> bool:
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self.config, f)
            logger.info(f"配置已保存到 {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"保存配置文件 {self.config_file} 失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    def _merge_configs(self, new_config: Dict[str, Any]):
        """合并配置"""
        for key, value in new_config.items():
            if key in self.config and isinstance(value, dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
                
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default
            
    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        current = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in current:
                current[k] = {}
            current = current[k]
            
        current[keys[-1]] = value

class ICCardAnalyzerPro:
    """IC卡分析专家系统主程序"""
    
    def __init__(self):
        # 初始化配置
        self.config = ConfigManager()
        
        # 初始化核心组件
        self.data_loader = ICDataLoader()
        self.engine = AnalysisEngine()
        self.report_generator = ReportGenerator()
        self.plugin_manager = PluginManager()
        self.resource_monitor = ResourceMonitor()
        
        # 初始化GUI
        self.window = tk.Tk()
        self.window.title("IC卡分析专家系统 v3.0")
        self.setup_window()
        
        # 数据存储
        self.dumps = {}
        self.current_results = {}
        
        # 状态变量
        self.dark_mode = self.config.get('system.dark_mode', False)
        self.auto_save = self.config.get('system.auto_save', True)
        
        # 启动资源监视
        self.resource_monitor.start(self.update_resource_display)
        
    def setup_window(self):
        """设置主窗口"""
        # 窗口大小和位置
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        self.window.geometry(f"{window_width}x{window_height}+{int((screen_width-window_width)/2)}+{int((screen_height-window_height)/2)}")
        self.window.minsize(800, 600)
        
        # 菜单栏
        self.setup_menubar()
        
        # 主布局
        self.main_paned = ttk.PanedWindow(self.window, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板
        self.control_frame = ttk.LabelFrame(self.main_paned, text="控制面板", padding=10)
        self.main_paned.add(self.control_frame, weight=1)
        self.setup_control_panel()
        
        # 右侧主内容区
        self.content_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.content_frame, weight=4)
        self.setup_content_area()
        
        # 状态栏
        self.status_bar = ttk.Frame(self.window)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.setup_status_bar()
        
        # 应用主题
        self.apply_theme()
        
    def setup_menubar(self):
        """设置菜单栏"""
        menubar = tk.Menu(self.window)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="导入数据", command=self.load_dumps)
        file_menu.add_command(label="导出数据", command=self.export_data)
        file_menu.add_separator()
        file_menu.add_command(label="生成报告", command=self.generate_report)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.quit_app)
        menubar.add_cascade(label="文件", menu=file_menu)
        
        # 编辑菜单
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="配置", command=self.show_config_dialog)
        edit_menu.add_checkbutton(label="自动保存", variable=tk.BooleanVar(value=self.auto_save), 
                                command=self.toggle_auto_save)
        edit_menu.add_checkbutton(label="黑暗模式", variable=tk.BooleanVar(value=self.dark_mode), 
                                command=self.toggle_dark_mode)
        menubar.add_cascade(label="编辑", menu=edit_menu)
        
        # 分析菜单
        analysis_menu = tk.Menu(menubar, tearoff=0)
        analysis_menu.add_command(label="AES密钥恢复", 
                                command=lambda: self.run_analysis(AnalysisAlgorithm.AES_KEY_RECOVERY))
        analysis_menu.add_command(label="滚动码预测", 
                                command=lambda: self.run_analysis(AnalysisAlgorithm.ROLLING_CODE_PREDICTION))
        analysis_menu.add_command(label="聚类分析", 
                                command=lambda: self.run_analysis(AnalysisAlgorithm.CLUSTERING))
        menubar.add_cascade(label="分析", menu=analysis_menu)
        
        # 插件菜单
        plugin_menu = tk.Menu(menubar, tearoff=0)
        for plugin_name in self.plugin_manager.list_plugins():
            plugin_menu.add_command(label=plugin_name, 
                                  command=lambda n=plugin_name: self.run_plugin(n))
        menubar.add_cascade(label="插件", menu=plugin_menu)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="关于", command=self.show_about)
        help_menu.add_command(label="文档", command=self.show_documentation)
        menubar.add_cascade(label="帮助", menu=help_menu)
        
        self.window.config(menu=menubar)
        
    def setup_control_panel(self):
        """设置控制面板"""
        # 文件加载区域
        file_frame = ttk.LabelFrame(self.control_frame, text="文件操作", padding=5)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_frame, text="导入文件", command=self.load_dumps).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="清除数据", command=self.clear_data).pack(fill=tk.X, pady=2)
        
        # 文件列表
        self.file_listbox = tk.Listbox(file_frame, height=10)
        self.file_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 分析选项区域
        analysis_frame = ttk.LabelFrame(self.control_frame, text="分析选项", padding=5)
        analysis_frame.pack(fill=tk.X, pady=5)
        
        self.algorithm_var = tk.StringVar()
        algorithms = ["AES密钥恢复", "滚动码预测", "聚类分析"]
        ttk.Combobox(analysis_frame, textvariable=self.algorithm_var, values=algorithms, state="readonly").pack(fill=tk.X, pady=2)
        
        ttk.Button(analysis_frame, text="执行分析", command=self.run_selected_analysis).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="停止分析", command=self.stop_analysis).pack(fill=tk.X, pady=2)
        
        # 进度条
        self.progress = ttk.Progressbar(self.control_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # 资源监控区域
        monitor_frame = ttk.LabelFrame(self.control_frame, text="系统资源", padding=5)
        monitor_frame.pack(fill=tk.X, pady=5)
        
        self.cpu_label = ttk.Label(monitor_frame, text="CPU: 0%")
        self.cpu_label.pack(anchor=tk.W)
        
        self.memory_label = ttk.Label(monitor_frame, text="内存: 0MB/0MB (0%)")
        self.memory_label.pack(anchor=tk.W)
        
        self.gpu_label = ttk.Label(monitor_frame, text="GPU: 未检测到")
        self.gpu_label.pack(anchor=tk.W)
        
    def setup_content_area(self):
        """设置主内容区域"""
        # 笔记本控件
        self.notebook = ttk.Notebook(self.content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 数据预览标签页
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="数据预览")
        self.setup_data_tab()
        
        # 分析结果标签页
        self.result_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.result_tab, text="分析结果")
        self.setup_result_tab()
        
        # 日志标签页
        self.log_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.log_tab, text="系统日志")
        self.setup_log_tab()
        
    def setup_data_tab(self):
        """设置数据预览标签页"""
        # 十六进制视图
        hex_frame = ttk.LabelFrame(self.data_tab, text="十六进制视图", padding=5)
        hex_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.hex_text = scrolledtext.ScrolledText(hex_frame, wrap=tk.NONE, font=('Courier', 10))
        self.hex_text.pack(fill=tk.BOTH, expand=True)
        
        # 底部控制按钮
        control_frame = ttk.Frame(self.data_tab)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.file_var = tk.StringVar()
        ttk.Combobox(control_frame, textvariable=self.file_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(control_frame, text="刷新视图", command=self.refresh_data_view).pack(side=tk.RIGHT, padx=2)
        
    def setup_result_tab(self):
        """设置分析结果标签页"""
        # 结果笔记本
        self.result_notebook = ttk.Notebook(self.result_tab)
        self.result_notebook.pack(fill=tk.BOTH, expand=True)
        
        # 文本结果标签页
        text_result_frame = ttk.Frame(self.result_notebook)
        self.result_notebook.add(text_result_frame, text="文本结果")
        
        self.result_text = scrolledtext.ScrolledText(text_result_frame, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # 图形结果标签页
        graph_result_frame = ttk.Frame(self.result_notebook)
        self.result_notebook.add(graph_result_frame, text="图形结果")
        
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, graph_result_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 元数据标签页
        metadata_frame = ttk.Frame(self.result_notebook)
        self.result_notebook.add(metadata_frame, text="元数据")
        
        self.metadata_text = scrolledtext.ScrolledText(metadata_frame, wrap=tk.WORD)
        self.metadata_text.pack(fill=tk.BOTH, expand=True)
        
    def setup_log_tab(self):
        """设置日志标签页"""
        self.log_text = scrolledtext.ScrolledText(self.log_tab, wrap=tk.WORD, state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # 重定向日志
        class LogHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
                
            def emit(self, record):
                msg = self.format(record)
                self.text_widget.config(state='normal')
                self.text_widget.insert(tk.END, msg + '\n')
                self.text_widget.config(state='disabled')
                self.text_widget.see(tk.END)
                
        log_handler = LogHandler(self.log_text)
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(log_handler)
        
    def setup_status_bar(self):
        """设置状态栏"""
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        
        ttk.Label(self.status_bar, textvariable=self.status_var).pack(side=tk.LEFT)
        
        self.time_var = tk.StringVar()
        self.time_var.set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ttk.Label(self.status_bar, textvariable=self.time_var).pack(side=tk.RIGHT)
        
        # 更新时间
        def update_time():
            self.time_var.set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            self.window.after(1000, update_time)
            
        update_time()
        
    def apply_theme(self):
        """应用主题"""
        if self.dark_mode:
            self.window.tk_setPalette(background='#2d2d2d', foreground='#ffffff',
                                    activeBackground='#3d3d3d', activeForeground='#ffffff')
            style = ttk.Style()
            style.theme_use('alt')
            style.configure('.', background='#2d2d2d', foreground='#ffffff')
            style.configure('TFrame', background='#2d2d2d')
            style.configure('TLabel', background='#2d2d2d', foreground='#ffffff')
            style.configure('TButton', background='#3d3d3d', foreground='#ffffff')
            style.configure('TEntry', fieldbackground='#3d3d3d', foreground='#ffffff')
            style.configure('TCombobox', fieldbackground='#3d3d3d', foreground='#ffffff')
            style.configure('TNotebook', background='#2d2d2d')
            style.configure('TNotebook.Tab', background='#3d3d3d', foreground='#ffffff')
            style.map('TCombobox', fieldbackground=[('readonly', '#3d3d3d')])
            style.map('TCombobox', selectbackground=[('readonly', '#4d4d4d')])
            style.map('TCombobox', selectforeground=[('readonly', '#ffffff')])
            
            # 配置文本区域
            bg_color = '#1e1e1e'
            fg_color = '#ffffff'
            self.hex_text.config(bg=bg_color, fg=fg_color, insertbackground=fg_color)
            self.result_text.config(bg=bg_color, fg=fg_color, insertbackground=fg_color)
            self.metadata_text.config(bg=bg_color, fg=fg_color, insertbackground=fg_color)
            self.log_text.config(bg=bg_color, fg=fg_color, insertbackground=fg_color)
            self.file_listbox.config(bg=bg_color, fg=fg_color, selectbackground='#4d4d4d')
        else:
            self.window.tk_setPalette(background='#f0f0f0', foreground='#000000',
                                    activeBackground='#e0e0e0', activeForeground='#000000')
            style = ttk.Style()
            style.theme_use('clam')
            style.configure('.', background='#f0f0f0', foreground='#000000')
            
            # 恢复默认颜色
            bg_color = 'white'
            fg_color = 'black'
            self.hex_text.config(bg=bg_color, fg=fg_color, insertbackground=fg_color)
            self.result_text.config(bg=bg_color, fg=fg_color, insertbackground=fg_color)
            self.metadata_text.config(bg=bg_color, fg=fg_color, insertbackground=fg_color)
            self.log_text.config(bg=bg_color, fg=fg_color, insertbackground=fg_color)
            self.file_listbox.config(bg=bg_color, fg=fg_color, selectbackground='#d0d0d0')
            
    def toggle_dark_mode(self):
        """切换黑暗模式"""
        self.dark_mode = not self.dark_mode
        self.config.set('system.dark_mode', self.dark_mode)
        self.config.save_config()
        self.apply_theme()
        
    def toggle_auto_save(self):
        """切换自动保存"""
        self.auto_save = not self.auto_save
        self.config.set('system.auto_save', self.auto_save)
        self.config.save_config()
        
    def update_status(self, message: str):
        """更新状态栏"""
        self.status_var.set(message)
        logger.info(message)
        
    def update_progress(self, value: float):
        """更新进度条"""
        self.progress['value'] = value * 100
        self.window.update_idletasks()
        
    def update_resource_display(self, data: Dict[str, Any]):
        """更新资源显示"""
        # CPU
        cpu_text = f"CPU: {data['cpu']:.1f}%"
        self.cpu_label.config(text=cpu_text)
        
        # 内存
        memory = data['memory']
        memory_text = f"内存: {memory['used']/1024/1024:.1f}MB/{memory['total']/1024/1024:.1f}MB ({memory['percent']}%)"
        self.memory_label.config(text=memory_text)
        
        # GPU
        if data['gpu']:
            gpu_text = f"GPU: {data['gpu'][0]['load']:.1f}% ({data['gpu'][0]['memory_used']:.1f}MB/{data['gpu'][0]['memory_total']:.1f}MB)"
            self.gpu_label.config(text=gpu_text)
        else:
            self.gpu_label.config(text="GPU: 未检测到")
            
    def load_dumps(self):
        """加载dump文件"""
        file_paths = filedialog.askopenfilenames(
            title="选择IC卡dump文件",
            filetypes=[("DUMP文件", "*.bin *.hex *.dump"), ("所有文件", "*.*")]
        )
        
        if not file_paths:
            return
            
        self.update_status(f"正在加载 {len(file_paths)} 个文件...")
        
        # 清空当前数据
        self.dumps.clear()
        self.file_listbox.delete(0, tk.END)
        self.file_var.set('')
        
        # 批量加载文件
        loaded_data = self.data_loader.batch_load_files(file_paths, self.update_progress)
        
        if loaded_data:
            self.dumps = loaded_data
            for file_path in loaded_data.keys():
                self.file_listbox.insert(tk.END, os.path.basename(file_path))
                
            self.file_var.set(os.path.basename(next(iter(loaded_data.keys()))))
            self.refresh_data_view()
            self.update_status(f"成功加载 {len(loaded_data)}/{len(file_paths)} 个文件")
        else:
            self.update_status("文件加载失败")
            
    def refresh_data_view(self):
        """刷新数据视图"""
        selected_file = self.file_var.get()
        if not selected_file or not self.dumps:
            return
            
        # 查找完整路径
        full_path = None
        for path in self.dumps.keys():
            if os.path.basename(path) == selected_file:
                full_path = path
                break
                
        if not full_path:
            return
            
        data = self.dumps[full_path]
        if data is None:
            return
            
        # 显示十六进制数据
        self.hex_text.delete(1.0, tk.END)
        
        hex_lines = []
        for i in range(0, min(len(data), 1024), 16):  # 只显示前1024字节
            chunk = data[i:i+16]
            hex_str = ' '.join(f'{b:02X}' for b in chunk)
            ascii_str = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
            hex_lines.append(f"{i:04X}: {hex_str.ljust(47)}  {ascii_str}")
            
        self.hex_text.insert(tk.END, '\n'.join(hex_lines))
        
    def clear_data(self):
        """清除所有数据"""
        if not self.dumps:
            return
            
        if messagebox.askyesno("确认", "确定要清除所有数据吗？"):
            self.dumps.clear()
            self.file_listbox.delete(0, tk.END)
            self.file_var.set('')
            self.hex_text.delete(1.0, tk.END)
            self.update_status("已清除所有数据")
            
    def run_selected_analysis(self):
        """运行选中的分析"""
        algorithm = self.algorithm_var.get()
        if not algorithm:
            messagebox.showwarning("警告", "请先选择分析算法")
            return
            
        if not self.dumps:
            messagebox.showwarning("警告", "没有可分析的数据")
            return
            
        # 映射算法名称到枚举
        algo_map = {
            "AES密钥恢复": AnalysisAlgorithm.AES_KEY_RECOVERY,
            "滚动码预测": AnalysisAlgorithm.ROLLING_CODE_PREDICTION,
            "聚类分析": AnalysisAlgorithm.CLUSTERING
        }
        
        self.run_analysis(algo_map[algorithm])
        
    def run_analysis(self, algorithm: AnalysisAlgorithm):
        """运行分析"""
        if not self.dumps:
            messagebox.showwarning("警告", "没有可分析的数据")
            return
            
        self.update_status(f"开始 {algorithm.name} 分析...")
        self.progress['value'] = 0
        
        # 准备数据
        data = list(self.dumps.values())
        
        # 提交分析任务
        self.engine.submit_task(algorithm, data, self.analysis_callback)
        
    def run_plugin(self, plugin_name: str):
        """运行插件"""
        if not self.dumps:
            messagebox.showwarning("警告", "没有可分析的数据")
            return
            
        self.update_status(f"开始执行插件: {plugin_name}")
        self.progress['value'] = 0
        
        # 使用线程执行插件
        def plugin_worker():
            try:
                result = self.plugin_manager.execute_plugin(plugin_name, list(self.dumps.values()))
                self.window.after(0, lambda: self.plugin_callback(plugin_name, result))
            except Exception as e:
                logger.error(f"插件执行失败: {str(e)}")
                self.window.after(0, lambda: self.update_status(f"插件 {plugin_name} 执行失败"))
                
        threading.Thread(target=plugin_worker, daemon=True).start()
        
    def stop_analysis(self):
        """停止当前分析"""
        self.engine.stop_current_task()
        self.update_status("分析已停止")
        
    def analysis_callback(self, result: AnalysisResult):
        """分析完成回调"""
        self.progress['value'] = 100
        self.current_results[result.message.split(':')[0]] = result
        
        # 显示结果
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"分析结果: {result.message}\n")
        self.result_text.insert(tk.END, f"执行时间: {result.execution_time:.2f}秒\n\n")
        
        if result.data is not None:
            if isinstance(result.data, (str, int, float, bool)):
                self.result_text.insert(tk.END, f"结果数据: {result.data}\n")
            elif isinstance(result.data, dict):
                self.result_text.insert(tk.END, "结果数据:\n")
                for key, value in result.data.items():
                    self.result_text.insert(tk.END, f"  {key}: {value}\n")
            else:
                self.result_text.insert(tk.END, f"结果数据类型: {type(result.data)}\n")
                
        # 显示元数据
        self.metadata_text.delete(1.0, tk.END)
        if result.metadata:
            self.metadata_text.insert(tk.END, "元数据:\n")
            for key, value in result.metadata.items():
                self.metadata_text.insert(tk.END, f"  {key}: {value}\n")
                
        # 如果是聚类分析，绘制图形
        if isinstance(result.data, dict) and 'features' in result.data:
            self.figure.clear()
            ax = self.figure.add_subplot(111, projection='3d')
            
            features = result.data['features']
            labels = result.data['kmeans_labels']
            
            ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels)
            ax.set_title('聚类分析结果 (3D PCA降维)')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            
            self.canvas.draw()
            
        self.update_status(f"分析完成: {result.message}")
        
    def plugin_callback(self, plugin_name: str, result: AnalysisResult):
        """插件完成回调"""
        self.progress['value'] = 100
        self.current_results[plugin_name] = result
        
        # 显示结果
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"插件结果: {result.message}\n")
        self.result_text.insert(tk.END, f"执行时间: {result.execution_time:.2f}秒\n\n")
        
        if result.data is not None:
            self.result_text.insert(tk.END, f"结果数据: {result.data}\n")
            
        self.update_status(f"插件 {plugin_name} 执行完成")
        
    def generate_report(self):
        """生成分析报告"""
        if not self.current_results:
            messagebox.showwarning("警告", "没有可报告的分析结果")
            return
            
        self.update_status("正在生成报告...")
        
        # 生成HTML报告
        html = self.report_generator.generate_html_report(self.current_results)
        
        # 保存报告
        file_path = filedialog.asksaveasfilename(
            title="保存报告",
            defaultextension=".pdf",
            filetypes=[("PDF文件", "*.pdf"), ("HTML文件", "*.html")]
        )
        
        if not file_path:
            return
            
        if file_path.endswith('.html'):
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html)
                self.update_status(f"HTML报告已保存: {file_path}")
                messagebox.showinfo("成功", f"HTML报告已保存到 {file_path}")
            except Exception as e:
                logger.error(f"保存HTML报告失败: {str(e)}")
                messagebox.showerror("错误", f"保存HTML报告失败: {str(e)}")
        else:
            if self.report_generator.generate_pdf_report(html, file_path):
                self.update_status(f"PDF报告已保存: {file_path}")
                messagebox.showinfo("成功", f"PDF报告已保存到 {file_path}")
            else:
                messagebox.showerror("错误", "生成PDF报告失败")
                
    def export_data(self):
        """导出数据"""
        if not self.dumps:
            messagebox.showwarning("警告", "没有可导出的数据")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="导出数据",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("Excel文件", "*.xlsx"), ("所有文件", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # 提取所有特征
            all_features = []
            for file_name, data in self.dumps.items():
                features = ICDataPreprocessor.extract_features(data)
                if features is not None:
                    features['file'] = os.path.basename(file_name)
                    all_features.append(features)
                    
            if not all_features:
                messagebox.showwarning("警告", "没有可导出的特征数据")
                return
                
            # 合并所有特征
            combined = pd.concat(all_features, ignore_index=True)
            
            if file_path.endswith('.csv'):
                combined.to_csv(file_path, index=False)
            else:
                combined.to_excel(file_path, index=False)
                
            self.update_status(f"数据已导出到 {file_path}")
            messagebox.showinfo("成功", f"数据已导出到 {file_path}")
        except Exception as e:
            logger.error(f"导出数据失败: {str(e)}")
            messagebox.showerror("错误", f"导出数据失败: {str(e)}")
            
    def show_config_dialog(self):
        """显示配置对话框"""
        dialog = tk.Toplevel(self.window)
        dialog.title("系统配置")
        dialog.transient(self.window)
        dialog.grab_set()
        
        # 系统配置
        sys_frame = ttk.LabelFrame(dialog, text="系统配置", padding=10)
        sys_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(sys_frame, text="最大文件数:").grid(row=0, column=0, sticky=tk.W)
        max_files_var = tk.IntVar(value=self.config.get('system.max_files'))
        ttk.Entry(sys_frame, textvariable=max_files_var).grid(row=0, column=1, sticky=tk.EW)
        
        ttk.Label(sys_frame, text="超时时间(秒):").grid(row=1, column=0, sticky=tk.W)
        timeout_var = tk.IntVar(value=self.config.get('system.timeout'))
        ttk.Entry(sys_frame, textvariable=timeout_var).grid(row=1, column=1, sticky=tk.EW)
        
        ttk.Label(sys_frame, text="最大线程数:").grid(row=2, column=0, sticky=tk.W)
        max_threads_var = tk.IntVar(value=self.config.get('system.max_threads'))
        ttk.Entry(sys_frame, textvariable=max_threads_var).grid(row=2, column=1, sticky=tk.EW)
        
        # 加密配置
        crypto_frame = ttk.LabelFrame(dialog, text="加密配置", padding=10)
        crypto_frame.pack(fill=tk.X, padx=10, pady=5)
        
        aes_brute_var = tk.BooleanVar(value=self.config.get('crypto.aes_bruteforce'))
        ttk.Checkbutton(crypto_frame, text="启用AES暴力破解", variable=aes_brute_var).grid(row=0, column=0, sticky=tk.W)
        
        ttk.Label(crypto_frame, text="最大密钥尝试次数:").grid(row=1, column=0, sticky=tk.W)
        max_key_var = tk.IntVar(value=self.config.get('crypto.max_key_attempts'))
        ttk.Entry(crypto_frame, textvariable=max_key_var).grid(row=1, column=1, sticky=tk.EW)
        
        # 按钮
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def save_config():
            self.config.set('system.max_files', max_files_var.get())
            self.config.set('system.timeout', timeout_var.get())
            self.config.set('system.max_threads', max_threads_var.get())
            self.config.set('crypto.aes_bruteforce', aes_brute_var.get())
            self.config.set('crypto.max_key_attempts', max_key_var.get())
            
            if self.config.save_config():
                messagebox.showinfo("成功", "配置已保存")
                dialog.destroy()
            else:
                messagebox.showerror("错误", "保存配置失败")
                
        ttk.Button(button_frame, text="保存", command=save_config).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="取消", command=dialog.destroy).pack(side=tk.RIGHT)
        
    def show_about(self):
        """显示关于对话框"""
        messagebox.showinfo("关于", "IC卡分析专家系统 v3.0\n\n一个强大的IC卡数据分析工具")
        
    def show_documentation(self):
        """显示文档"""
        doc = """
        IC卡分析专家系统使用说明
        
        1. 文件操作
        - 导入数据: 加载IC卡dump文件
        - 导出数据: 将分析结果导出为CSV或Excel
        - 生成报告: 创建HTML或PDF格式的分析报告
        
        2. 分析功能
        - AES密钥恢复: 尝试从多个dump中恢复AES密钥
        - 滚动码预测: 使用LSTM预测滚动码
        - 聚类分析: 对dump数据进行聚类
        
        3. 插件系统
        - 支持动态加载插件扩展功能
        
        4. 系统配置
        - 可调整系统参数和算法设置
        """
        
        dialog = tk.Toplevel(self.window)
        dialog.title("使用文档")
        dialog.transient(self.window)
        
        text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, width=80, height=25)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text.insert(tk.END, doc)
        text.config(state='disabled')
        
        ttk.Button(dialog, text="关闭", command=dialog.destroy).pack(pady=10)
        
    def quit_app(self):
        """退出应用程序"""
        if messagebox.askokcancel("退出", "确定要退出IC卡分析专家系统吗？"):
            self.resource_monitor.stop()
            self.engine.shutdown()
            self.window.quit()
            
    def run(self):
        """运行应用程序"""
        self.window.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.window.mainloop()

if __name__ == "__main__":
    # 设置高DPI感知 (Windows)
    if sys.platform == 'win32':
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
        
    app = ICCardAnalyzerPro()
    app.run()