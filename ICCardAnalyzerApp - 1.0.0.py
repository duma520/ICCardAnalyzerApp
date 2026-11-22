import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import os
import logging
import yaml
import hashlib
import time
import threading
import queue
import traceback
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple, Any, Union
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from pyod.models.cblof import CBLOF
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from neo4j import GraphDatabase
import redis
from datetime import datetime
import unittest
from contextlib import contextmanager
import gc
import psutil
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ic_card_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置加载
class ConfigManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.config_path = 'config.yaml'
        self.default_config = {
            'clustering': {
                'default_algorithm': 'gmm',
                'max_clusters': 5,
                'timeout': 30
            },
            'feature': {
                'ngram_range': [2, 4],
                'entropy_window': 16,
                'max_file_size_mb': 100
            },
            'system': {
                'max_threads': 4,
                'enable_redis': True,
                'redis_ttl': 3600,
                'log_level': 'INFO'
            }
        }
        self._load_config()
    
    def _load_config(self):
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f) or self.default_config
                    # 合并默认配置和用户配置
                    for section, values in self.default_config.items():
                        if section not in self.config:
                            self.config[section] = values
                        else:
                            for key, default_value in values.items():
                                if key not in self.config[section]:
                                    self.config[section][key] = default_value
            else:
                self.config = self.default_config
                self._save_config()
                
            # 更新日志级别
            log_level = self.config['system'].get('log_level', 'INFO')
            logger.setLevel(getattr(logging, log_level.upper()))
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            self.config = self.default_config
    
    def _save_config(self):
        try:
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(self.config, f)
        except Exception as e:
            logger.error(f"保存配置文件失败: {str(e)}")
    
    def get(self, key: str, default=None) -> Any:
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default
    
    def set(self, key: str, value: Any) -> bool:
        keys = key.split('.')
        current = self.config
        try:
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
            self._save_config()
            return True
        except Exception as e:
            logger.error(f"设置配置项失败: {str(e)}")
            return False

# Redis缓存
class RedisCache:
    def __init__(self):
        self.config = ConfigManager().get('system')
        self.enabled = self.config.get('enable_redis', False)
        self.redis_client = None
        self._initialize()
    
    def _initialize(self):
        if self.enabled:
            try:
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=0,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
                # 测试连接
                self.redis_client.ping()
                logger.info("Redis连接成功")
            except Exception as e:
                logger.warning(f"Redis连接失败: {str(e)}")
                self.enabled = False
                self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value is not None:
                return value.decode('utf-8')
            return None
        except Exception as e:
            logger.error(f"Redis获取数据失败: {str(e)}")
            return None
    
    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            if ttl is None:
                ttl = self.config.get('redis_ttl', 3600)
            self.redis_client.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error(f"Redis设置数据失败: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis删除数据失败: {str(e)}")
            return False
    
    def clear_cache(self) -> bool:
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            self.redis_client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis清空缓存失败: {str(e)}")
            return False

# 异常类
class ICardAnalysisError(Exception):
    """IC卡分析基础异常类"""
    pass

class FileValidationError(ICardAnalysisError):
    """文件验证失败异常"""
    pass

class AlgorithmTimeoutError(ICardAnalysisError):
    """算法超时异常"""
    pass

class ResourceExhaustedError(ICardAnalysisError):
    """资源耗尽异常"""
    pass

# 插件接口
class AnalysisPlugin(ABC):
    """分析插件抽象基类"""
    
    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """处理数据方法"""
        pass
    
    @abstractmethod
    def validate_input(self, data: Any) -> bool:
        """验证输入数据"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """清理资源"""
        pass

# 特征提取插件
class FeatureExtractor(AnalysisPlugin):
    """特征提取插件"""
    
    def __init__(self):
        self.ngram_range = ConfigManager().get('feature.ngram_range', [2, 4])
        self.entropy_window = ConfigManager().get('feature.entropy_window', 16)
        self.cache = RedisCache()
    
    def process(self, data: bytes, **kwargs) -> Dict[str, Any]:
        """处理数据方法"""
        if not self.validate_input(data):
            raise FileValidationError("无效的输入数据")
        
        # 检查缓存
        data_hash = hashlib.sha256(data).hexdigest()
        cache_key = f"features_{data_hash}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            logger.info("从缓存中获取特征数据")
            return eval(cached_result)
        
        try:
            # 提取特征
            features = {
                'byte_ngrams': self._extract_byte_ngrams(data),
                'entropy_features': self._calculate_entropy(data),
                'size': len(data),
                'header': data[:4].hex() if len(data) >= 4 else ''
            }
            
            # 缓存结果
            self.cache.set(cache_key, str(features))
            
            return features
        except Exception as e:
            logger.error(f"特征提取失败: {str(e)}")
            raise ICardAnalysisError(f"特征提取失败: {str(e)}")
    
    def validate_input(self, data: Any) -> bool:
        """验证输入数据"""
        if not isinstance(data, bytes):
            return False
        if len(data) == 0:
            return False
        return True
    
    def _extract_byte_ngrams(self, data: bytes) -> Dict[str, int]:
        """提取字节N-gram特征"""
        ngrams = {}
        min_n, max_n = self.ngram_range
        
        for n in range(min_n, max_n + 1):
            for i in range(len(data) - n + 1):
                ngram = data[i:i+n].hex()
                ngrams[ngram] = ngrams.get(ngram, 0) + 1
        
        return ngrams
    
    def _calculate_entropy(self, data: bytes) -> List[float]:
        """计算滑动窗口熵值"""
        if len(data) < self.entropy_window:
            return []
        
        entropy_values = []
        for i in range(len(data) - self.entropy_window + 1):
            window = data[i:i+self.entropy_window]
            freq = {}
            for byte in window:
                freq[byte] = freq.get(byte, 0) + 1
            
            entropy = 0.0
            total = len(window)
            for count in freq.values():
                p = count / total
                entropy -= p * np.log2(p)
            
            entropy_values.append(entropy)
        
        return entropy_values
    
    def cleanup(self):
        """清理资源"""
        pass

# 聚类分析插件
class ClusterAnalyzer(AnalysisPlugin):
    """聚类分析插件"""
    
    def __init__(self):
        self.max_clusters = ConfigManager().get('clustering.max_clusters', 5)
        self.timeout = ConfigManager().get('clustering.timeout', 30)
        self.cache = RedisCache()
    
    def process(self, data: Union[np.ndarray, List[Dict[str, Any]]], **kwargs) -> Dict[str, Any]:
        """执行聚类分析"""
        if not self.validate_input(data):
            raise FileValidationError("无效的输入数据")
        
        algorithm = kwargs.get('algorithm', 'gmm')
        features = self._prepare_features(data)
        
        # 检查缓存
        data_hash = hashlib.sha256(features.tobytes()).hexdigest()
        cache_key = f"clustering_{algorithm}_{data_hash}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            logger.info(f"从缓存中获取{algorithm}聚类结果")
            return eval(cached_result)
        
        try:
            # 设置超时
            result_queue = queue.Queue()
            thread = threading.Thread(
                target=self._run_clustering,
                args=(features, algorithm, result_queue)
            )
            thread.start()
            thread.join(timeout=self.timeout)
            
            if thread.is_alive():
                raise AlgorithmTimeoutError(f"{algorithm}聚类超时")
            
            if result_queue.empty():
                raise ICardAnalysisError("聚类分析未返回结果")
            
            result = result_queue.get()
            
            # 缓存结果
            self.cache.set(cache_key, str(result))
            
            return result
        except Exception as e:
            logger.error(f"聚类分析失败: {str(e)}")
            raise ICardAnalysisError(f"聚类分析失败: {str(e)}")
    
    def validate_input(self, data: Any) -> bool:
        """验证输入数据"""
        if isinstance(data, list):
            if len(data) == 0:
                return False
            return all(isinstance(item, dict) for item in data)
        elif isinstance(data, np.ndarray):
            return data.size > 0
        return False
    
    def _prepare_features(self, data: Union[np.ndarray, List[Dict[str, Any]]]) -> np.ndarray:
        """准备特征矩阵"""
        if isinstance(data, np.ndarray):
            return data
        
        # 从特征字典列表创建特征矩阵
        feature_matrix = []
        for item in data:
            features = []
            # 添加N-gram特征
            ngrams = item.get('byte_ngrams', {})
            features.extend(ngrams.values())
            # 添加熵值特征
            entropy = item.get('entropy_features', [])
            features.extend(entropy)
            # 添加其他特征
            features.append(item.get('size', 0))
            
            feature_matrix.append(features)
        
        return np.array(feature_matrix)
    
    def _run_clustering(self, features: np.ndarray, algorithm: str, result_queue: queue.Queue):
        """执行聚类算法"""
        try:
            result = {}
            
            if algorithm == 'gmm':
                # 高斯混合模型
                n_components = min(self.max_clusters, len(features))
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    random_state=42
                )
                labels = gmm.fit_predict(features)
                result = {
                    'algorithm': 'GMM',
                    'labels': labels.tolist(),
                    'n_clusters': n_components,
                    'probabilities': gmm.predict_proba(features).tolist()
                }
            elif algorithm == 'hierarchical':
                # 层次聚类
                n_clusters = min(self.max_clusters, len(features))
                agg = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='ward'
                )
                labels = agg.fit_predict(features)
                result = {
                    'algorithm': 'Hierarchical',
                    'labels': labels.tolist(),
                    'n_clusters': n_clusters
                }
            else:
                raise ValueError(f"未知的聚类算法: {algorithm}")
            
            result_queue.put(result)
        except Exception as e:
            logger.error(f"聚类算法执行失败: {str(e)}")
            result_queue.put({'error': str(e)})
    
    def cleanup(self):
        """清理资源"""
        pass

# 异常检测插件
class AnomalyDetector(AnalysisPlugin):
    """异常检测插件"""
    
    def __init__(self):
        self.cache = RedisCache()
    
    def process(self, data: Union[np.ndarray, List[Dict[str, Any]]], **kwargs) -> Dict[str, Any]:
        """执行异常检测"""
        if not self.validate_input(data):
            raise FileValidationError("无效的输入数据")
        
        features = self._prepare_features(data)
        
        # 检查缓存
        data_hash = hashlib.sha256(features.tobytes()).hexdigest()
        cache_key = f"anomaly_{data_hash}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            logger.info("从缓存中获取异常检测结果")
            return eval(cached_result)
        
        try:
            # 使用CBLOF算法检测异常
            clf = CBLOF(
                contamination=0.1,  # 假设10%的数据是异常的
                check_estimator=False,
                random_state=42
            )
            clf.fit(features)
            scores = clf.decision_function(features)
            labels = clf.predict(features)
            
            result = {
                'algorithm': 'CBLOF',
                'scores': scores.tolist(),
                'labels': labels.tolist(),
                'threshold': np.percentile(scores, 90)  # 使用90百分位作为阈值
            }
            
            # 缓存结果
            self.cache.set(cache_key, str(result))
            
            return result
        except Exception as e:
            logger.error(f"异常检测失败: {str(e)}")
            raise ICardAnalysisError(f"异常检测失败: {str(e)}")
    
    def validate_input(self, data: Any) -> bool:
        """验证输入数据"""
        if isinstance(data, list):
            if len(data) == 0:
                return False
            return all(isinstance(item, dict) for item in data)
        elif isinstance(data, np.ndarray):
            return data.size > 0
        return False
    
    def _prepare_features(self, data: Union[np.ndarray, List[Dict[str, Any]]]) -> np.ndarray:
        """准备特征矩阵"""
        if isinstance(data, np.ndarray):
            return data
        
        # 从特征字典列表创建特征矩阵
        feature_matrix = []
        for item in data:
            features = []
            # 添加N-gram特征
            ngrams = item.get('byte_ngrams', {})
            features.extend(ngrams.values())
            # 添加熵值特征
            entropy = item.get('entropy_features', [])
            features.extend(entropy)
            # 添加其他特征
            features.append(item.get('size', 0))
            
            feature_matrix.append(features)
        
        return np.array(feature_matrix)
    
    def cleanup(self):
        """清理资源"""
        pass

# 时序预测插件
class TimeSeriesPredictor(AnalysisPlugin):
    """时序预测插件"""
    
    def __init__(self):
        self.model = None
        self.cache = RedisCache()
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化LSTM模型"""
        self.model = Sequential([
            LSTM(64, input_shape=(10, 1), return_sequences=True),
            LSTM(32),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
    
    def process(self, data: List[float], **kwargs) -> Dict[str, Any]:
        """执行时序预测"""
        if not self.validate_input(data):
            raise FileValidationError("无效的输入数据")
        
        # 检查缓存
        data_hash = hashlib.sha256(str(data).encode()).hexdigest()
        cache_key = f"timeseries_{data_hash}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            logger.info("从缓存中获取时序预测结果")
            return eval(cached_result)
        
        try:
            # 准备数据
            X, y = self._prepare_data(data)
            
            # 训练模型
            self.model.fit(
                X, y,
                epochs=10,
                batch_size=16,
                verbose=0
            )
            
            # 预测未来5个点
            last_window = X[-1].reshape(1, 10, 1)
            predictions = []
            for _ in range(5):
                pred = self.model.predict(last_window, verbose=0)[0, 0]
                predictions.append(pred)
                # 更新窗口
                last_window = np.roll(last_window, -1, axis=1)
                last_window[0, -1, 0] = pred
            
            result = {
                'algorithm': 'LSTM',
                'predictions': predictions,
                'actual': data,
                'window_size': 10
            }
            
            # 缓存结果
            self.cache.set(cache_key, str(result))
            
            return result
        except Exception as e:
            logger.error(f"时序预测失败: {str(e)}")
            raise ICardAnalysisError(f"时序预测失败: {str(e)}")
    
    def validate_input(self, data: Any) -> bool:
        """验证输入数据"""
        if not isinstance(data, list):
            return False
        if len(data) < 20:  # 至少需要20个数据点
            return False
        return all(isinstance(x, (int, float)) for x in data)
    
    def _prepare_data(self, data: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        X = []
        y = []
        window_size = 10
        
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])
        
        X = np.array(X).reshape(-1, window_size, 1)
        y = np.array(y)
        
        return X, y
    
    def cleanup(self):
        """清理资源"""
        if self.model:
            del self.model
            self.model = None
            gc.collect()

# 知识图谱插件
class KnowledgeGraph(AnalysisPlugin):
    """知识图谱插件"""
    
    def __init__(self):
        self.driver = None
        self.cache = RedisCache()
        self._initialize_driver()
    
    def _initialize_driver(self):
        """初始化Neo4j驱动"""
        try:
            self.driver = GraphDatabase.driver(
                "bolt://localhost:7687",
                auth=("neo4j", "password"),
                max_connection_pool_size=10,
                connection_timeout=5
            )
            # 测试连接
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j连接成功")
        except Exception as e:
            logger.error(f"Neo4j连接失败: {str(e)}")
            self.driver = None
    
    def process(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """构建知识图谱"""
        if not self.validate_input(data):
            raise FileValidationError("无效的输入数据")
        
        # 检查缓存
        data_hash = hashlib.sha256(str(data).encode()).hexdigest()
        cache_key = f"knowledgegraph_{data_hash}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            logger.info("从缓存中获取知识图谱结果")
            return eval(cached_result)
        
        if not self.driver:
            raise ICardAnalysisError("Neo4j连接不可用")
        
        try:
            result = {'nodes': [], 'relationships': []}
            
            with self.driver.session() as session:
                # 创建IC卡节点
                ic_card_id = data.get('header', 'unknown')
                session.run(
                    "MERGE (ic:ICCard {id: $id}) "
                    "SET ic.size = $size, ic.entropy = $entropy",
                    id=ic_card_id,
                    size=data.get('size', 0),
                    entropy=np.mean(data.get('entropy_features', []))
                )
                result['nodes'].append({
                    'id': ic_card_id,
                    'label': 'ICCard',
                    'properties': {
                        'size': data.get('size', 0),
                        'entropy': np.mean(data.get('entropy_features', []))
                    }
                })
                
                # 创建厂商节点（如果header中有厂商信息）
                if ic_card_id.startswith(('a0', 'a1')):
                    manufacturer = "ManufacturerX"
                    session.run(
                        "MERGE (m:Manufacturer {name: $name})",
                        name=manufacturer
                    )
                    session.run(
                        "MATCH (ic:ICCard {id: $ic_id}), (m:Manufacturer {name: $name}) "
                        "MERGE (ic)-[:PRODUCED_BY]->(m)",
                        ic_id=ic_card_id,
                        name=manufacturer
                    )
                    result['nodes'].append({
                        'id': manufacturer,
                        'label': 'Manufacturer'
                    })
                    result['relationships'].append({
                        'source': ic_card_id,
                        'target': manufacturer,
                        'type': 'PRODUCED_BY'
                    })
                
                # 添加漏洞信息（模拟）
                vulnerabilities = self._detect_vulnerabilities(data)
                for vuln in vulnerabilities:
                    session.run(
                        "MERGE (v:Vulnerability {name: $name}) "
                        "SET v.severity = $severity",
                        name=vuln['name'],
                        severity=vuln['severity']
                    )
                    session.run(
                        "MATCH (ic:ICCard {id: $ic_id}), (v:Vulnerability {name: $name}) "
                        "MERGE (ic)-[:HAS_VULNERABILITY]->(v)",
                        ic_id=ic_card_id,
                        name=vuln['name']
                    )
                    result['nodes'].append({
                        'id': vuln['name'],
                        'label': 'Vulnerability',
                        'properties': {
                            'severity': vuln['severity']
                        }
                    })
                    result['relationships'].append({
                        'source': ic_card_id,
                        'target': vuln['name'],
                        'type': 'HAS_VULNERABILITY'
                    })
            
            # 缓存结果
            self.cache.set(cache_key, str(result))
            
            return result
        except Exception as e:
            logger.error(f"知识图谱构建失败: {str(e)}")
            raise ICardAnalysisError(f"知识图谱构建失败: {str(e)}")
    
    def validate_input(self, data: Any) -> bool:
        """验证输入数据"""
        if not isinstance(data, dict):
            return False
        if 'header' not in data:
            return False
        return True
    
    def _detect_vulnerabilities(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检测潜在漏洞（模拟）"""
        vulnerabilities = []
        
        # 基于header检测漏洞
        header = data.get('header', '')
        if header.startswith('a0'):
            vulnerabilities.append({
                'name': 'CVE-2023-IC-A0',
                'severity': 'High'
            })
        
        # 基于熵值检测异常
        entropy = np.mean(data.get('entropy_features', []))
        if entropy > 6.5:  # 高熵值可能表示加密不足
            vulnerabilities.append({
                'name': 'CVE-2023-LOW-ENTROPY',
                'severity': 'Medium'
            })
        
        return vulnerabilities
    
    def cleanup(self):
        """清理资源"""
        if self.driver:
            self.driver.close()
            self.driver = None

# 插件管理器
class PluginManager:
    """插件管理器"""
    
    def __init__(self):
        self.plugins = {}
        self._initialize_plugins()
    
    def _initialize_plugins(self):
        """初始化所有插件"""
        self.register_plugin('feature_extractor', FeatureExtractor())
        self.register_plugin('cluster_analyzer', ClusterAnalyzer())
        self.register_plugin('anomaly_detector', AnomalyDetector())
        self.register_plugin('time_series', TimeSeriesPredictor())
        self.register_plugin('knowledge_graph', KnowledgeGraph())
    
    def register_plugin(self, name: str, plugin: AnalysisPlugin):
        """注册插件"""
        if not isinstance(plugin, AnalysisPlugin):
            raise ValueError("插件必须实现AnalysisPlugin接口")
        self.plugins[name] = plugin
    
    def get_plugin(self, name: str) -> Optional[AnalysisPlugin]:
        """获取插件"""
        return self.plugins.get(name)
    
    def process(self, plugin_name: str, data: Any, **kwargs) -> Any:
        """使用插件处理数据"""
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            raise ValueError(f"未知插件: {plugin_name}")
        
        try:
            return plugin.process(data, **kwargs)
        except Exception as e:
            logger.error(f"插件{plugin_name}处理失败: {str(e)}")
            raise
    
    def cleanup(self):
        """清理所有插件资源"""
        for plugin in self.plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"插件清理失败: {str(e)}")

# 文件处理器
class FileProcessor:
    """文件处理器"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.max_file_size = self.config.get('feature.max_file_size_mb', 100) * 1024 * 1024
        self.cache = RedisCache()
    
    def load_file(self, file_path: str) -> bytes:
        """加载文件内容"""
        try:
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                raise ResourceExhaustedError(f"文件大小超过限制: {file_size/1024/1024:.2f}MB")
            
            # 检查文件扩展名
            if not file_path.lower().endswith(('.bin', '.hex')):
                raise FileValidationError("仅支持.bin或.hex文件")
            
            # 读取文件内容
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # 验证文件内容
            if len(content) == 0:
                raise FileValidationError("文件为空")
            
            return content
        except Exception as e:
            logger.error(f"文件加载失败: {str(e)}")
            raise
    
    def validate_hex_dump(self, hex_str: str) -> bool:
        """验证hex dump格式"""
        hex_str = hex_str.strip()
        if not hex_str:
            return False
        
        # 移除可能的地址部分和ASCII部分
        lines = hex_str.split('\n')
        for line in lines:
            # 检查典型的hex dump格式: 地址 + 16字节hex + ASCII
            parts = line.split()
            if len(parts) < 2:
                continue
            
            # 检查hex部分
            hex_part = parts[1:17]  # 假设hex部分在地址之后
            for byte in hex_part:
                if len(byte) != 2:
                    return False
                try:
                    int(byte, 16)
                except ValueError:
                    return False
        
        return True
    
    def hex_to_bin(self, hex_str: str) -> bytes:
        """将hex字符串转换为二进制数据"""
        if not self.validate_hex_dump(hex_str):
            raise FileValidationError("无效的hex dump格式")
        
        try:
            # 移除所有空白字符
            hex_str = ''.join(hex_str.split())
            # 转换为bytes
            return bytes.fromhex(hex_str)
        except Exception as e:
            logger.error(f"hex转换失败: {str(e)}")
            raise FileValidationError(f"hex转换失败: {str(e)}")

# 分析引擎
class AnalysisEngine:
    """分析引擎"""
    
    def __init__(self):
        self.plugin_manager = PluginManager()
        self.file_processor = FileProcessor()
        self.config = ConfigManager()
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_threads = []
        self._initialize_workers()
    
    def _initialize_workers(self):
        """初始化工作线程"""
        max_threads = self.config.get('system.max_threads', 4)
        for _ in range(max_threads):
            thread = threading.Thread(
                target=self._worker_loop,
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
    
    def _worker_loop(self):
        """工作线程循环"""
        while True:
            try:
                task = self.task_queue.get()
                if task is None:  # 停止信号
                    break
                
                task_id, plugin_name, data, kwargs = task
                try:
                    result = self.plugin_manager.process(plugin_name, data, **kwargs)
                    self.result_queue.put((task_id, result, None))
                except Exception as e:
                    self.result_queue.put((task_id, None, str(e)))
            except Exception as e:
                logger.error(f"工作线程错误: {str(e)}")
    
    def submit_task(self, plugin_name: str, data: Any, **kwargs) -> str:
        """提交分析任务"""
        task_id = hashlib.md5(f"{plugin_name}{time.time()}".encode()).hexdigest()
        self.task_queue.put((task_id, plugin_name, data, kwargs))
        return task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Tuple[Any, Optional[str]]:
        """获取任务结果"""
        start_time = time.time()
        
        while True:
            try:
                # 检查结果队列
                result_task_id, result, error = self.result_queue.get_nowait()
                if result_task_id == task_id:
                    return result, error
                else:
                    # 不是我们要的结果，放回队列
                    self.result_queue.put((result_task_id, result, error))
            except queue.Empty:
                pass
            
            # 检查超时
            if timeout is not None and (time.time() - start_time) > timeout:
                return None, "获取结果超时"
            
            # 短暂休眠避免CPU占用过高
            time.sleep(0.1)
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """完整分析文件"""
        try:
            # 1. 加载文件
            file_content = self.file_processor.load_file(file_path)
            
            # 2. 提取特征
            feature_task = self.submit_task('feature_extractor', file_content)
            features, error = self.get_result(feature_task, timeout=30)
            if error:
                raise ICardAnalysisError(f"特征提取失败: {error}")
            
            # 3. 聚类分析
            cluster_task = self.submit_task(
                'cluster_analyzer',
                [features],  # 包装成列表，因为通常处理多个文件
                algorithm='gmm'
            )
            cluster_result, error = self.get_result(cluster_task, timeout=60)
            if error:
                raise ICardAnalysisError(f"聚类分析失败: {error}")
            
            # 4. 异常检测
            anomaly_task = self.submit_task('anomaly_detector', [features])
            anomaly_result, error = self.get_result(anomaly_task, timeout=45)
            if error:
                raise ICardAnalysisError(f"异常检测失败: {error}")
            
            # 5. 知识图谱
            kg_task = self.submit_task('knowledge_graph', features)
            kg_result, error = self.get_result(kg_task, timeout=30)
            if error:
                logger.warning(f"知识图谱构建失败: {error}")
                kg_result = {'error': error}
            
            # 6. 时序分析（如果有足够的数据）
            time_series_data = features.get('entropy_features', [])
            if len(time_series_data) >= 20:
                ts_task = self.submit_task('time_series', time_series_data)
                ts_result, error = self.get_result(ts_task, timeout=60)
                if error:
                    logger.warning(f"时序分析失败: {error}")
                    ts_result = {'error': error}
            else:
                ts_result = {'error': '数据不足，需要至少20个熵值点'}
            
            return {
                'status': 'success',
                'features': features,
                'clustering': cluster_result,
                'anomaly': anomaly_result,
                'knowledge_graph': kg_result,
                'time_series': ts_result,
                'file_info': {
                    'path': file_path,
                    'size': len(file_content),
                    'hash': hashlib.sha256(file_content).hexdigest()
                }
            }
        except Exception as e:
            logger.error(f"文件分析失败: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'file_info': {
                    'path': file_path,
                    'error': str(e)
                }
            }
    
    def cleanup(self):
        """清理资源"""
        # 发送停止信号给工作线程
        for _ in range(len(self.worker_threads)):
            self.task_queue.put(None)
        
        # 等待工作线程结束
        for thread in self.worker_threads:
            thread.join(timeout=5)
        
        # 清理插件
        self.plugin_manager.cleanup()

# GUI应用程序
class ICCardAnalyzerApp:
    """IC卡分析器GUI应用程序"""
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("IC卡大数据分析系统 v2.0")
        self.window.geometry("1000x700")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # 初始化组件
        self.engine = AnalysisEngine()
        self.current_files = []
        self.current_task = None
        self.analysis_results = {}
        self.plugin_manager = PluginManager()
        
        # 设置样式
        self.setup_styles()
        
        # 构建界面
        self.setup_ui()
        
        # 状态监控
        self.monitor_thread = threading.Thread(
            target=self._monitor_system_status,
            daemon=True
        )
        self.monitor_thread.start()
        
        # 异常处理
        self.window.report_callback_exception = self.handle_error
    
    def setup_styles(self):
        """设置界面样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 配置样式
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10), padding=5)
        style.configure('TCombobox', padding=5)
        style.configure('TNotebook', background='#f0f0f0')
        style.configure('TNotebook.Tab', padding=[10, 5], font=('Arial', 10))
        
        # 自定义样式
        style.configure('Status.TLabel', foreground='gray', font=('Arial', 9))
        style.configure('Error.TLabel', foreground='red', font=('Arial', 10))
        style.configure('Success.TLabel', foreground='green', font=('Arial', 10))
        style.configure('Title.TLabel', font=('Arial', 12, 'bold'))
    
    def setup_ui(self):
        """构建GUI界面"""
        # 主框架
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill='both', expand=True)
        
        # 顶部控制面板
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=(0, 10))
        
        # 文件加载按钮
        file_button = ttk.Button(
            control_frame,
            text="加载IC卡Dump文件",
            command=self.load_files
        )
        file_button.pack(side='left', padx=5)
        
        # 算法选择
        self.algorithm_var = tk.StringVar()
        algo_combo = ttk.Combobox(
            control_frame,
            textvariable=self.algorithm_var,
            values=["自动分析", "特征提取", "聚类分析", "异常检测", "时序预测", "知识图谱"],
            state='readonly',
            width=15
        )
        algo_combo.current(0)
        algo_combo.pack(side='left', padx=5)
        
        # 分析按钮
        analyze_button = ttk.Button(
            control_frame,
            text="开始分析",
            command=self.start_analysis
        )
        analyze_button.pack(side='left', padx=5)
        
        # 清除按钮
        clear_button = ttk.Button(
            control_frame,
            text="清除结果",
            command=self.clear_results
        )
        clear_button.pack(side='left', padx=5)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(
            control_frame,
            textvariable=self.status_var,
            style='Status.TLabel'
        )
        status_label.pack(side='right', padx=5)
        
        # 主内容区域
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill='both', expand=True)
        
        # 左侧文件列表
        file_list_frame = ttk.LabelFrame(
            content_frame,
            text="已加载文件",
            padding=10
        )
        file_list_frame.pack(side='left', fill='y', padx=(0, 10))
        
        self.file_listbox = tk.Listbox(
            file_list_frame,
            width=30,
            height=20,
            selectmode='extended'
        )
        self.file_listbox.pack(fill='both', expand=True)
        
        # 右侧结果区域
        result_frame = ttk.Frame(content_frame)
        result_frame.pack(side='right', fill='both', expand=True)
        
        # 结果标签页
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # 文本结果标签页
        text_tab = ttk.Frame(self.notebook)
        self.notebook.add(text_tab, text="分析结果")
        
        self.result_text = tk.Text(
            text_tab,
            wrap='word',
            font=('Courier New', 10)
        )
        self.result_text.pack(fill='both', expand=True)
        
        # 图表标签页
        plot_tab = ttk.Frame(self.notebook)
        self.notebook.add(plot_tab, text="可视化")
        
        self.figure = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, plot_tab)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # 底部信息栏
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill='x', pady=(10, 0))
        
        self.memory_var = tk.StringVar(value="内存: -")
        memory_label = ttk.Label(
            info_frame,
            textvariable=self.memory_var,
            style='Status.TLabel'
        )
        memory_label.pack(side='left')
        
        self.cpu_var = tk.StringVar(value="CPU: -")
        cpu_label = ttk.Label(
            info_frame,
            textvariable=self.cpu_var,
            style='Status.TLabel'
        )
        cpu_label.pack(side='left', padx=10)
        
        self.task_var = tk.StringVar(value="任务: 空闲")
        task_label = ttk.Label(
            info_frame,
            textvariable=self.task_var,
            style='Status.TLabel'
        )
        task_label.pack(side='right')
    
    def load_files(self):
        """加载文件"""
        try:
            file_paths = filedialog.askopenfilenames(
                title="选择IC卡Dump文件",
                filetypes=[("IC卡Dump文件", "*.bin *.hex"), ("所有文件", "*.*")]
            )
            
            if not file_paths:
                return
            
            self.current_files = []
            self.file_listbox.delete(0, tk.END)
            
            for file_path in file_paths:
                try:
                    # 验证文件
                    file_size = os.path.getsize(file_path)
                    max_size = ConfigManager().get('feature.max_file_size_mb', 100) * 1024 * 1024
                    if file_size > max_size:
                        messagebox.showwarning(
                            "文件过大",
                            f"文件 {os.path.basename(file_path)} 超过最大限制 {max_size/1024/1024}MB"
                        )
                        continue
                    
                    self.current_files.append(file_path)
                    self.file_listbox.insert(tk.END, os.path.basename(file_path))
                except Exception as e:
                    logger.error(f"加载文件 {file_path} 失败: {str(e)}")
                    messagebox.showerror(
                        "加载失败",
                        f"无法加载文件 {os.path.basename(file_path)}: {str(e)}"
                    )
            
            self.status_var.set(f"已加载 {len(self.current_files)} 个文件")
        except Exception as e:
            logger.error(f"加载文件失败: {str(e)}")
            self.handle_error(type(e), e, e.__traceback__)
    
    def start_analysis(self):
        """开始分析"""
        if not self.current_files:
            messagebox.showwarning("警告", "请先加载文件")
            return
        
        selected_algo = self.algorithm_var.get()
        if selected_algo == "自动分析":
            self.run_full_analysis()
        else:
            self.run_single_analysis(selected_algo)
    
    def run_full_analysis(self):
        """运行完整分析流程"""
        if self.current_task and self.current_task.is_alive():
            messagebox.showwarning("警告", "已有分析任务在进行中")
            return
        
        def analysis_thread():
            try:
                self.status_var.set("分析中...")
                self.task_var.set("任务: 分析中")
                
                for file_path in self.current_files:
                    self.result_text.insert(tk.END, f"\n分析文件: {file_path}\n")
                    self.result_text.see(tk.END)
                    self.window.update()
                    
                    try:
                        result = self.engine.analyze_file(file_path)
                        self.analysis_results[file_path] = result
                        
                        if result['status'] == 'success':
                            self.display_analysis_result(result)
                        else:
                            self.result_text.insert(
                                tk.END,
                                f"分析失败: {result['error']}\n",
                                'error'
                            )
                    except Exception as e:
                        logger.error(f"分析文件 {file_path} 失败: {str(e)}")
                        self.result_text.insert(
                            tk.END,
                            f"分析文件 {file_path} 失败: {str(e)}\n",
                            'error'
                        )
                    
                    self.result_text.see(tk.END)
                    self.window.update()
                
                self.status_var.set("分析完成")
                self.task_var.set("任务: 完成")
            except Exception as e:
                logger.error(f"分析失败: {str(e)}")
                self.handle_error(type(e), e, e.__traceback__)
            finally:
                self.current_task = None
        
        self.current_task = threading.Thread(target=analysis_thread, daemon=True)
        self.current_task.start()
    
    def run_single_analysis(self, analysis_type: str):
        """运行单个分析任务"""
        if self.current_task and self.current_task.is_alive():
            messagebox.showwarning("警告", "已有分析任务在进行中")
            return
        
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("警告", "请选择要分析的文件")
            return
        
        selected_files = [self.current_files[i] for i in selected_indices]
        
        def analysis_thread():
            try:
                self.status_var.set(f"{analysis_type}中...")
                self.task_var.set(f"任务: {analysis_type}")
                
                plugin_map = {
                    "特征提取": "feature_extractor",
                    "聚类分析": "cluster_analyzer",
                    "异常检测": "anomaly_detector",
                    "时序预测": "time_series",
                    "知识图谱": "knowledge_graph"
                }
                plugin_name = plugin_map.get(analysis_type)
                
                if not plugin_name:
                    self.result_text.insert(tk.END, f"未知分析类型: {analysis_type}\n", 'error')
                    return
                
                for file_path in selected_files:
                    self.result_text.insert(tk.END, f"\n{analysis_type}: {file_path}\n")
                    self.result_text.see(tk.END)
                    self.window.update()
                    
                    try:
                        # 加载文件
                        file_content = self.engine.file_processor.load_file(file_path)
                        
                        # 对于非特征提取插件，需要先提取特征
                        if plugin_name != "feature_extractor":
                            feature_task = self.engine.submit_task('feature_extractor', file_content)
                            features, error = self.engine.get_result(feature_task, timeout=30)
                            if error:
                                raise ICardAnalysisError(f"特征提取失败: {error}")
                            input_data = features
                        else:
                            input_data = file_content
                        
                        # 执行分析
                        task_id = self.engine.submit_task(plugin_name, input_data)
                        result, error = self.engine.get_result(task_id, timeout=60)
                        
                        if error:
                            raise ICardAnalysisError(error)
                        
                        # 显示结果
                        self.display_single_analysis_result(analysis_type, result)
                    except Exception as e:
                        logger.error(f"{analysis_type} 失败: {str(e)}")
                        self.result_text.insert(
                            tk.END,
                            f"{analysis_type} 失败: {str(e)}\n",
                            'error'
                        )
                    
                    self.result_text.see(tk.END)
                    self.window.update()
                
                self.status_var.set(f"{analysis_type}完成")
                self.task_var.set("任务: 完成")
            except Exception as e:
                logger.error(f"{analysis_type} 失败: {str(e)}")
                self.handle_error(type(e), e, e.__traceback__)
            finally:
                self.current_task = None
        
        self.current_task = threading.Thread(target=analysis_thread, daemon=True)
        self.current_task.start()
    
    def display_analysis_result(self, result: Dict[str, Any]):
        """显示完整分析结果"""
        try:
            # 清空图表
            self.figure.clear()
            
            # 显示基本信息
            file_info = result['file_info']
            self.result_text.insert(tk.END, f"\n文件信息:\n")
            self.result_text.insert(tk.END, f"路径: {file_info['path']}\n")
            self.result_text.insert(tk.END, f"大小: {file_info['size']} 字节\n")
            self.result_text.insert(tk.END, f"SHA256: {file_info['hash']}\n")
            
            # 显示特征信息
            features = result['features']
            self.result_text.insert(tk.END, f"\n特征提取结果:\n")
            self.result_text.insert(tk.END, f"文件头: {features.get('header', '无')}\n")
            self.result_text.insert(tk.END, f"N-gram数量: {len(features.get('byte_ngrams', {}))}\n")
            self.result_text.insert(tk.END, f"熵值计算点数: {len(features.get('entropy_features', []))}\n")
            
            # 显示聚类结果
            clustering = result['clustering']
            self.result_text.insert(tk.END, f"\n聚类分析结果 ({clustering['algorithm']}):\n")
            self.result_text.insert(tk.END, f"聚类数量: {clustering['n_clusters']}\n")
            
            # 绘制聚类结果
            if 'probabilities' in clustering:
                ax = self.figure.add_subplot(221)
                ax.set_title("聚类概率分布")
                ax.hist(clustering['probabilities'], bins=20)
            else:
                ax = self.figure.add_subplot(221)
                ax.set_title("聚类标签分布")
                ax.hist(clustering['labels'], bins=clustering['n_clusters'])
            
            # 显示异常检测结果
            anomaly = result['anomaly']
            self.result_text.insert(tk.END, f"\n异常检测结果 ({anomaly['algorithm']}):\n")
            anomaly_count = sum(anomaly['labels'])
            self.result_text.insert(tk.END, f"异常点数量: {anomaly_count}\n")
            self.result_text.insert(tk.END, f"异常阈值: {anomaly['threshold']:.2f}\n")
            
            # 绘制异常分数
            ax = self.figure.add_subplot(222)
            ax.set_title("异常分数分布")
            ax.hist(anomaly['scores'], bins=20)
            ax.axvline(anomaly['threshold'], color='r', linestyle='--')
            
            # 显示知识图谱结果
            kg = result['knowledge_graph']
            if 'error' in kg:
                self.result_text.insert(tk.END, f"\n知识图谱构建失败: {kg['error']}\n", 'error')
            else:
                self.result_text.insert(tk.END, f"\n知识图谱结果:\n")
                self.result_text.insert(tk.END, f"节点数量: {len(kg['nodes'])}\n")
                self.result_text.insert(tk.END, f"关系数量: {len(kg['relationships'])}\n")
                
                # 绘制简单的关系图
                if kg['relationships']:
                    ax = self.figure.add_subplot(223)
                    ax.set_title("知识图谱简图")
                    ax.axis('off')
                    
                    nodes = kg['nodes']
                    relationships = kg['relationships']
                    
                    # 简化的可视化
                    node_pos = {}
                    for i, node in enumerate(nodes[:5]):  # 只显示前5个节点
                        node_pos[node['id']] = (i, 0)
                    
                    for rel in relationships[:10]:  # 只显示前10个关系
                        src = rel['source']
                        tgt = rel['target']
                        if src in node_pos and tgt in node_pos:
                            x1, y1 = node_pos[src]
                            x2, y2 = node_pos[tgt]
                            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3)
                    
                    for node_id, (x, y) in node_pos.items():
                        ax.plot(x, y, 'o', markersize=10)
                        ax.text(x, y+0.1, node_id[:6]+"...", ha='center')
            
            # 显示时序分析结果
            ts = result['time_series']
            if 'error' in ts:
                self.result_text.insert(tk.END, f"\n时序分析失败: {ts['error']}\n", 'error')
            else:
                self.result_text.insert(tk.END, f"\n时序分析结果 ({ts['algorithm']}):\n")
                self.result_text.insert(tk.END, f"预测点数: {len(ts['predictions'])}\n")
                
                # 绘制时序图
                ax = self.figure.add_subplot(224)
                ax.set_title("熵值时序预测")
                ax.plot(ts['actual'], label='实际值')
                pred_x = range(len(ts['actual']), len(ts['actual']) + len(ts['predictions']))
                ax.plot(pred_x, ts['predictions'], label='预测值')
                ax.legend()
            
            # 刷新图表
            self.canvas.draw()
            self.notebook.select(1)  # 切换到可视化标签页
            
            self.result_text.insert(tk.END, "\n分析完成!\n", 'success')
        except Exception as e:
            logger.error(f"显示分析结果失败: {str(e)}")
            self.result_text.insert(tk.END, f"显示分析结果失败: {str(e)}\n", 'error')
    
    def display_single_analysis_result(self, analysis_type: str, result: Dict[str, Any]):
        """显示单个分析结果"""
        try:
            # 清空图表
            self.figure.clear()
            
            if analysis_type == "特征提取":
                self.result_text.insert(tk.END, f"\n特征提取结果:\n")
                self.result_text.insert(tk.END, f"文件头: {result.get('header', '无')}\n")
                self.result_text.insert(tk.END, f"N-gram数量: {len(result.get('byte_ngrams', {}))}\n")
                self.result_text.insert(tk.END, f"熵值计算点数: {len(result.get('entropy_features', []))}\n")
                
                # 绘制熵值图
                if result.get('entropy_features'):
                    ax = self.figure.add_subplot(111)
                    ax.set_title("熵值变化")
                    ax.plot(result['entropy_features'])
                    self.canvas.draw()
                    self.notebook.select(1)  # 切换到可视化标签页
            
            elif analysis_type == "聚类分析":
                self.result_text.insert(tk.END, f"\n聚类分析结果 ({result['algorithm']}):\n")
                self.result_text.insert(tk.END, f"聚类数量: {result['n_clusters']}\n")
                
                # 绘制聚类结果
                ax = self.figure.add_subplot(111)
                if 'probabilities' in result:
                    ax.set_title("聚类概率分布")
                    ax.hist(result['probabilities'], bins=20)
                else:
                    ax.set_title("聚类标签分布")
                    ax.hist(result['labels'], bins=result['n_clusters'])
                self.canvas.draw()
                self.notebook.select(1)  # 切换到可视化标签页
            
            elif analysis_type == "异常检测":
                self.result_text.insert(tk.END, f"\n异常检测结果 ({result['algorithm']}):\n")
                anomaly_count = sum(result['labels'])
                self.result_text.insert(tk.END, f"异常点数量: {anomaly_count}\n")
                self.result_text.insert(tk.END, f"异常阈值: {result['threshold']:.2f}\n")
                
                # 绘制异常分数
                ax = self.figure.add_subplot(111)
                ax.set_title("异常分数分布")
                ax.hist(result['scores'], bins=20)
                ax.axvline(result['threshold'], color='r', linestyle='--')
                self.canvas.draw()
                self.notebook.select(1)  # 切换到可视化标签页
            
            elif analysis_type == "时序预测":
                self.result_text.insert(tk.END, f"\n时序分析结果 ({result['algorithm']}):\n")
                self.result_text.insert(tk.END, f"预测点数: {len(result['predictions'])}\n")
                
                # 绘制时序图
                ax = self.figure.add_subplot(111)
                ax.set_title("时序预测")
                ax.plot(result['actual'], label='实际值')
                pred_x = range(len(result['actual']), len(result['actual']) + len(result['predictions']))
                ax.plot(pred_x, result['predictions'], label='预测值')
                ax.legend()
                self.canvas.draw()
                self.notebook.select(1)  # 切换到可视化标签页
            
            elif analysis_type == "知识图谱":
                if 'error' in result:
                    self.result_text.insert(tk.END, f"\n知识图谱构建失败: {result['error']}\n", 'error')
                else:
                    self.result_text.insert(tk.END, f"\n知识图谱结果:\n")
                    self.result_text.insert(tk.END, f"节点数量: {len(result['nodes'])}\n")
                    self.result_text.insert(tk.END, f"关系数量: {len(result['relationships'])}\n")
                    
                    # 绘制简单的关系图
                    if result['relationships']:
                        ax = self.figure.add_subplot(111)
                        ax.set_title("知识图谱简图")
                        ax.axis('off')
                        
                        nodes = result['nodes']
                        relationships = result['relationships']
                        
                        # 简化的可视化
                        node_pos = {}
                        for i, node in enumerate(nodes[:5]):  # 只显示前5个节点
                            node_pos[node['id']] = (i, 0)
                        
                        for rel in relationships[:10]:  # 只显示前10个关系
                            src = rel['source']
                            tgt = rel['target']
                            if src in node_pos and tgt in node_pos:
                                x1, y1 = node_pos[src]
                                x2, y2 = node_pos[tgt]
                                ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3)
                        
                        for node_id, (x, y) in node_pos.items():
                            ax.plot(x, y, 'o', markersize=10)
                            ax.text(x, y+0.1, node_id[:6]+"...", ha='center')
                        
                        self.canvas.draw()
                        self.notebook.select(1)  # 切换到可视化标签页
            
            self.result_text.insert(tk.END, f"\n{analysis_type}完成!\n", 'success')
        except Exception as e:
            logger.error(f"显示{analysis_type}结果失败: {str(e)}")
            self.result_text.insert(tk.END, f"显示{analysis_type}结果失败: {str(e)}\n", 'error')
    
    def clear_results(self):
        """清除结果"""
        self.result_text.delete(1.0, tk.END)
        self.figure.clear()
        self.canvas.draw()
        self.status_var.set("已清除结果")
    
    def _monitor_system_status(self):
        """监控系统状态"""
        while True:
            try:
                # 获取内存使用情况
                mem = psutil.virtual_memory()
                self.memory_var.set(f"内存: {mem.percent}%")
                
                # 获取CPU使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_var.set(f"CPU: {cpu_percent}%")
                
                time.sleep(2)
            except Exception as e:
                logger.error(f"系统监控失败: {str(e)}")
                time.sleep(5)
    
    def handle_error(self, exc_type, exc_value, exc_traceback):
        """处理未捕获的异常"""
        error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logger.critical(f"未处理的异常:\n{error_msg}")
        
        messagebox.showerror(
            "系统错误",
            f"发生未处理的异常:\n{str(exc_value)}\n\n详细信息请查看日志文件"
        )
        
        # 更新状态
        self.status_var.set("发生错误")
        self.task_var.set("任务: 错误")
    
    def on_close(self):
        """关闭窗口时的清理工作"""
        try:
            # 停止分析引擎
            self.engine.cleanup()
            
            # 关闭窗口
            self.window.destroy()
        except Exception as e:
            logger.error(f"关闭应用程序失败: {str(e)}")
            self.window.destroy()

# 单元测试
class TestICCardAnalyzer(unittest.TestCase):
    """IC卡分析器单元测试"""
    
    def setUp(self):
        """测试前准备"""
        self.file_processor = FileProcessor()
        self.plugin_manager = PluginManager()
        self.test_data = bytes([0x00, 0x01, 0x02, 0x03, 0x04, 0x05])
        self.test_file = 'test.bin'
        
        # 创建测试文件
        with open(self.test_file, 'wb') as f:
            f.write(self.test_data)
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_file_loading(self):
        """测试文件加载"""
        content = self.file_processor.load_file(self.test_file)
        self.assertEqual(content, self.test_data)
        
        # 测试无效文件
        with self.assertRaises(FileValidationError):
            self.file_processor.load_file('nonexistent.bin')
    
    def test_feature_extraction(self):
        """测试特征提取"""
        extractor = self.plugin_manager.get_plugin('feature_extractor')
        features = extractor.process(self.test_data)
        
        self.assertIn('byte_ngrams', features)
        self.assertIn('entropy_features', features)
        self.assertEqual(features['size'], len(self.test_data))
        
        # 测试无效输入
        with self.assertRaises(FileValidationError):
            extractor.process(b'')
    
    def test_clustering(self):
        """测试聚类分析"""
        # 先提取特征
        extractor = self.plugin_manager.get_plugin('feature_extractor')
        features = extractor.process(self.test_data)
        
        # 测试聚类
        clusterer = self.plugin_manager.get_plugin('cluster_analyzer')
        result = clusterer.process([features, features])  # 两个相同的特征
        
        self.assertIn('labels', result)
        self.assertEqual(len(result['labels']), 2)
        
        # 测试超时
        with self.assertRaises(AlgorithmTimeoutError):
            # 设置非常短的超时时间
            ConfigManager().set('clustering.timeout', 0.001)
            clusterer.process([features, features])
    
    def test_anomaly_detection(self):
        """测试异常检测"""
        # 先提取特征
        extractor = self.plugin_manager.get_plugin('feature_extractor')
        features = extractor.process(self.test_data)
        
        # 测试异常检测
        detector = self.plugin_manager.get_plugin('anomaly_detector')
        result = detector.process([features, features])  # 两个相同的特征
        
        self.assertIn('scores', result)
        self.assertEqual(len(result['scores']), 2)
    
    def test_time_series(self):
        """测试时序预测"""
        # 生成测试数据
        test_data = [i % 10 for i in range(30)]  # 简单的周期性数据
        
        # 测试时序预测
        predictor = self.plugin_manager.get_plugin('time_series')
        result = predictor.process(test_data)
        
        self.assertIn('predictions', result)
        self.assertEqual(len(result['predictions']), 5)
        
        # 测试数据不足
        with self.assertRaises(FileValidationError):
            predictor.process([1, 2, 3])
    
    def test_knowledge_graph(self):
        """测试知识图谱"""
        # 先提取特征
        extractor = self.plugin_manager.get_plugin('feature_extractor')
        features = extractor.process(self.test_data)
        
        # 测试知识图谱
        kg = self.plugin_manager.get_plugin('knowledge_graph')
        result = kg.process(features)
        
        self.assertIn('nodes', result)
        self.assertIn('relationships', result)

# 主函数
def main():
    """主函数"""
    try:
        # 初始化配置
        config = ConfigManager()
        
        # 创建并运行应用程序
        app = ICCardAnalyzerApp()
        app.window.mainloop()
    except Exception as e:
        logger.critical(f"应用程序启动失败: {str(e)}")
        messagebox.showerror(
            "启动失败",
            f"应用程序启动失败:\n{str(e)}\n\n详细信息请查看日志文件"
        )

if __name__ == "__main__":
    # 运行单元测试
    test_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    if not test_result.wasSuccessful():
        logger.error("单元测试失败，应用程序将不会启动")
        sys.exit(1)
    
    # 运行主程序
    main()