import sys
import os
import logging
import time
import ctypes
import numpy as np

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logging.warning("pycuda模块未安装，GPU加速功能不可用")


from PyQt6.QtCore import QThread, pyqtSignal, QObject, QTimer, QMutex, QMutexLocker, Qt, QSize
from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QTabWidget, QTextEdit, 
    QProgressBar, QLabel, QPushButton, QComboBox, QCheckBox, 
    QFileDialog, QMessageBox, QWidget, QSplitter, QStatusBar
)
from PyQt6.QtGui import QIcon, QFont, QColor, QPalette

# 图形可视化支持检测
VISUALIZATION_AVAILABLE = True
try:
    import pyqtgraph as pg
    from pyqtgraph import PlotWidget
    import pyqtgraph.opengl as gl
    from OpenGL.GL import *
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    logging.warning(f"图形可视化功能不可用: {e}")

    gl = type('', (), {'GLViewWidget': QWidget})  # 创建模拟的gl模块



# 确保在所有平台上都能优雅降级
TF_AVAILABLE = True
TF_ERROR = None
TF_WARNING_SHOWN = False

try:
    import tensorflow as tf
    try:
        # 验证 TensorFlow 是否能实际运行
        tf.constant([1, 2, 3])  # 简单操作测试
        if not hasattr(tf, '__version__') or tf.__version__ < '2.0':
            raise ImportError("需要 TensorFlow 2.0 或更高版本")
    except Exception as e:
        TF_AVAILABLE = False
        TF_ERROR = f"TensorFlow 测试失败: {str(e)}"
except ImportError as e:
    TF_AVAILABLE = False
    TF_ERROR = str(e)
except Exception as e:
    TF_AVAILABLE = False
    TF_ERROR = f"TensorFlow 初始化错误: {str(e)}"

if not TF_AVAILABLE:
    logging.warning(f"TensorFlow 不可用: {TF_ERROR}")
    
    # 增强的模拟实现
    class FakeKerasLayer:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            
        def __call__(self, inputs):
            return inputs
            
    class FakeTF:
        def __init__(self):
            self.keras = type('', (), {
                'layers': type('', (), {
                    'InputLayer': FakeKerasLayer,
                    'Conv1D': FakeKerasLayer,
                    'GlobalMaxPooling1D': lambda: FakeKerasLayer(),
                    'Dense': lambda units, activation: FakeKerasLayer()
                }),
                'models': type('', (), {
                    'load_model': lambda path: FakeTF.Sequential([])
                }),
                'Sequential': lambda layers: type('', (), {
                    'predict': lambda x, verbose=0: [[0.5, 0.3, 0.2]],
                    'summary': lambda: print("模拟模型 - TensorFlow 不可用"),
                    'evaluate': lambda *args, **kwargs: (0.0, 0.0)
                })
            })
            
            self.__version__ = "2.0.0 (模拟)"
    
    tf = FakeTF()



from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from dataclasses import dataclass
import subprocess
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import hashlib
import yaml
import json
import platform
import psutil
import gc
import inspect
import traceback
from enum import Enum, auto
import re
import tempfile
import shutil
from pathlib import Path
import zipfile
import uuid
import signal
import socket
import select
import threading
import queue
import mmap
import struct
import binascii
import numpy as np
from PyQt6.QtWidgets import QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QMutex, QMutexLocker
from PyQt6.QtCore import Qt


try:
    import fcntl  # Linux/Unix
except ImportError:
    # Windows 平台模拟 fcntl 的基本功能
    class fake_fcntl:
        LOCK_EX = 0x02
        LOCK_NB = 0x04
        @staticmethod
        def flock(fd, op):
            pass  # Windows 不需要文件锁定
    fcntl = fake_fcntl()


import atexit

# 替换原有的 resource 导入
try:
    import resource  # Unix/Linux
except ImportError:
    # Windows 平台模拟 resource 模块
    class FakeResource:
        RLIMIT_AS = 0
        RLIMIT_CORE = 0
        RLIMIT_CPU = 0
        RLIMIT_DATA = 0
        RLIMIT_FSIZE = 0
        RLIMIT_NOFILE = 0
        RLIMIT_STACK = 0
        
        @staticmethod
        def getrlimit(resource):
            return (0, 0)
            
        @staticmethod
        def setrlimit(resource, limits):
            pass
            
        @staticmethod
        def prlimit(pid, resource, limits=None):
            return (0, 0)
    
    resource = FakeResource()


import cProfile
import pstats
from io import StringIO

# ==================== 全局配置 ====================
class AppConfig:
    """应用程序全局配置类"""
    VERSION = "5.0.1"
    APP_NAME = "IC卡终极分析平台"
    LOG_FILE = "ic_analyzer.log"
    CONFIG_FILE = "config_advanced.yaml"
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5
    DEFAULT_TIMEOUT = 30  # 秒
    MAX_THREADS = 8
    GPU_BLOCK_SIZE = 256
    GPU_GRID_SIZE = 1024
    AES_BLOCK_SIZE = 16
    MAX_HEX_DATA_SIZE = 4096  # 4KB
    MODEL_INPUT_SIZE = 16
    SAMPLE_RATE = 106000  # Proxmark3默认采样率
    SIGNAL_THRESHOLD = 0.8  # 信号相似度阈值
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1  # 秒

# ==================== 日志系统 ====================
class RotatingFileHandler(logging.Handler):
    """支持日志轮转的文件处理器"""
    def __init__(self, filename, max_size=AppConfig.MAX_LOG_SIZE, backup_count=AppConfig.BACKUP_COUNT):
        super().__init__()
        self.filename = filename
        self.max_size = max_size
        self.backup_count = backup_count
        self.file = None
        self._open_file()
        
    def _open_file(self):
        if self.file is not None:
            self.file.close()
        self.file = open(self.filename, 'a', encoding='utf-8')
        
    def _rotate(self):
        if os.path.exists(self.filename):
            if os.path.getsize(self.filename) >= self.max_size:
                for i in range(self.backup_count - 1, 0, -1):
                    src = f"{self.filename}.{i}"
                    dst = f"{self.filename}.{i+1}"
                    if os.path.exists(src):
                        shutil.move(src, dst)
                shutil.move(self.filename, f"{self.filename}.1")
                self._open_file()
        
    def emit(self, record):
        try:
            self._rotate()
            msg = self.format(record)
            self.file.write(msg + '\n')
            self.file.flush()
        except Exception as e:
            print(f"日志记录失败: {e}", file=sys.stderr)

def setup_logging():
    """配置应用程序日志系统"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # 文件处理器
    file_handler = RotatingFileHandler(AppConfig.LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # 捕获未处理异常
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("未捕获的异常", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception

# ==================== 异常类 ====================
class ICardAnalysisError(Exception):
    """IC卡分析基础异常类"""
    pass

class GPUAccelerationError(ICardAnalysisError):
    """GPU加速错误"""
    pass

class DecryptionError(ICardAnalysisError):
    """解密错误"""
    pass

class InvalidInputError(ICardAnalysisError):
    """无效输入错误"""
    pass

class ResourceBusyError(ICardAnalysisError):
    """资源忙错误"""
    pass

class TimeoutError(ICardAnalysisError):
    """超时错误"""
    pass

class ModelLoadError(ICardAnalysisError):
    """模型加载错误"""
    pass

class HardwareError(ICardAnalysisError):
    """硬件错误"""
    pass

# ==================== 工具函数 ====================
def validate_hex_data(data: Union[str, bytes]) -> bytes:
    """验证并规范化HEX数据输入"""
    if isinstance(data, bytes):
        if len(data) > AppConfig.MAX_HEX_DATA_SIZE:
            raise InvalidInputError(f"数据大小超过最大限制 {AppConfig.MAX_HEX_DATA_SIZE} 字节")
        return data
    
    if isinstance(data, str):
        # 移除可能的空格和分隔符
        clean_data = re.sub(r'[^0-9A-Fa-f]', '', data)
        if len(clean_data) % 2 != 0:
            raise InvalidInputError("HEX字符串长度必须为偶数")
        try:
            bin_data = binascii.unhexlify(clean_data)
            if len(bin_data) > AppConfig.MAX_HEX_DATA_SIZE:
                raise InvalidInputError(f"数据大小超过最大限制 {AppConfig.MAX_HEX_DATA_SIZE} 字节")
            return bin_data
        except binascii.Error as e:
            raise InvalidInputError(f"无效的HEX数据: {e}")
    
    raise InvalidInputError("输入数据类型必须是str或bytes")

def measure_time(func):
    """测量函数执行时间的装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        logging.debug(f"函数 {func.__name__} 执行时间: {elapsed:.6f} 秒")
        return result
    return wrapper

def check_system_compatibility():
    """检查系统兼容性并返回报告"""
    report = {
        'os': platform.system(),
        'python_version': platform.python_version(),
        'gpu_available': GPU_AVAILABLE,
        'visualization_available': VISUALIZATION_AVAILABLE,
        'tensorflow_available': TF_AVAILABLE,
        'tensorflow_error': TF_ERROR if not TF_AVAILABLE else None,
        'issues': []
    }
    
    if os.name == 'nt':  # Windows
        if not GPU_AVAILABLE:
            report['issues'].append("Windows 上 GPU 加速可能不可用")
        if not VISUALIZATION_AVAILABLE:
            report['issues'].append("Windows 上 3D 可视化可能不可用")
    
    return report


def retry_on_failure(max_attempts=AppConfig.MAX_RETRY_ATTEMPTS, delay=AppConfig.RETRY_DELAY, 
                    exceptions=(Exception,)):
    """重试装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logging.warning(f"尝试 {attempt} 失败, {delay}秒后重试... 错误: {e}")
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

def singleton(cls):
    """单例装饰器"""
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

def check_system_resources():
    """检查系统资源是否充足"""
    mem = psutil.virtual_memory()
    if mem.available < 512 * 1024 * 1024:  # 512MB
        raise ResourceBusyError("系统内存不足")
    
    load = os.getloadavg()
    if load[0] > os.cpu_count() * 0.8:
        raise ResourceBusyError("系统负载过高")

# ==================== 数据模型 ====================
@dataclass
class KeySearchResult:
    """密钥搜索结果数据类"""
    key: Optional[int]
    attempts: int
    elapsed_time: float
    speed: float  # keys/sec

@dataclass
class EncryptionType:
    """加密类型数据类"""
    name: str
    confidence: float
    description: str

@dataclass
class CardDump:
    """卡片数据转储类"""
    uid: bytes
    sector_data: Dict[int, bytes]
    timestamp: float
    signal_quality: float

# ==================== GPU加速模块 ====================
class GPUCryptoManager:
    """GPU加密管理类"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._mutex = QMutex()
        self._cuda_context = None
        self._cuda_module = None
        self.gpu_available = GPU_AVAILABLE
        if self.gpu_available:
            self._init_cuda()
        else:
            logging.warning("GPU不可用，将使用CPU模式")
        
    def _init_cuda(self):
        """初始化CUDA环境"""
        try:
            # 检查CUDA设备
            device_count = cuda.Device.count()
            if device_count == 0:
                raise GPUAccelerationError("未找到可用的CUDA设备")
            
            # 选择第一个设备
            self._cuda_device = cuda.Device(0)
            self._cuda_context = self._cuda_device.make_context()
            
            # 编译CUDA内核
            cuda_code = """
            __device__ bool aes_decrypt(const unsigned char *ciphertext, int key) {
                // 简化的AES解密逻辑 - 实际实现需要完整的AES算法
                return (key % 10007) == 0;  // 模拟解密成功
            }
            
            __global__ void brute_force(const unsigned char *dumps, int *results, int size) {
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                if (idx < size) {
                    if (aes_decrypt(dumps, idx)) {
                        results[0] = idx;  // 返回找到的密钥
                    }
                }
            }"""
            
            self._cuda_module = SourceModule(cuda_code)
            logging.info("CUDA初始化成功")
        except Exception as e:
            raise GPUAccelerationError(f"CUDA初始化失败: {e}")
    
    @retry_on_failure()
    def gpu_aes_break(self, dumps: bytes) -> Optional[int]:
        """GPU加速AES破解"""
        if not self.gpu_available:
            logging.warning("GPU不可用，自动切换到CPU破解模式")
            return self._cpu_aes_break(dumps)
            
        if not isinstance(dumps, bytes):
            raise InvalidInputError("输入必须是bytes类型")
        
        if len(dumps) % AppConfig.AES_BLOCK_SIZE != 0:
            raise InvalidInputError(f"数据长度必须是 {AppConfig.AES_BLOCK_SIZE} 的倍数")
        
        with QMutexLocker(self._mutex):
            try:
                # 准备设备内存
                dumps_gpu = cuda.mem_alloc(len(dumps))
                results_gpu = cuda.mem_alloc(4)  # 单个int结果
                
                # 初始化结果
                cuda.memcpy_htod(dumps_gpu, dumps)
                cuda.memcpy_htod(results_gpu, np.array([-1], dtype=np.int32))
                
                # 获取CUDA函数
                brute_force = self._cuda_module.get_function("brute_force")
                
                # 计算网格大小
                block_size = AppConfig.GPU_BLOCK_SIZE
                grid_size = AppConfig.GPU_GRID_SIZE
                total_threads = block_size * grid_size
                
                # 执行内核
                start_time = time.perf_counter()
                brute_force(dumps_gpu, results_gpu, np.int32(total_threads),
                           block=(block_size, 1, 1), grid=(grid_size, 1))
                
                # 获取结果
                result = np.empty(1, dtype=np.int32)
                cuda.memcpy_dtoh(result, results_gpu)
                
                # 释放设备内存
                dumps_gpu.free()
                results_gpu.free()
                
                elapsed = time.perf_counter() - start_time
                speed = total_threads / elapsed if elapsed > 0 else 0
                
                logging.info(f"GPU破解完成 - 耗时: {elapsed:.3f}秒, 速度: {speed:,.0f} keys/sec")
                
                return result[0] if result[0] != -1 else None
            except Exception as e:
                logging.error(f"GPU破解失败: {e}")
                raise GPUAccelerationError(f"GPU破解失败: {e}")
    
    def _cpu_aes_break(self, dumps: bytes) -> Optional[int]:
        """CPU模式AES破解"""
        try:
            max_key = 1000000
            chunk_size = 1000
            start_time = time.perf_counter()
            
            for i in range(0, max_key, chunk_size):
                # 模拟CPU破解
                time.sleep(0.01)  # 模拟计算时间
                
                # 随机"找到"密钥
                if np.random.random() < 0.0001:
                    result = i + np.random.randint(0, chunk_size)
                    elapsed = time.perf_counter() - start_time
                    speed = (i + chunk_size) / elapsed if elapsed > 0 else 0
                    logging.info(f"CPU破解完成 - 耗时: {elapsed:.3f}秒, 速度: {speed:,.0f} keys/sec")
                    return result
                    
            elapsed = time.perf_counter() - start_time
            logging.info(f"CPU破解完成 - 未找到密钥, 耗时: {elapsed:.3f}秒")
            return None
        except Exception as e:
            logging.error(f"CPU破解失败: {e}")
            raise GPUAccelerationError(f"CPU破解失败: {e}")


    def __del__(self):
        """清理CUDA资源"""
        if hasattr(self, '_cuda_context') and self._cuda_context is not None:
            try:
                self._cuda_context.pop()
                self._cuda_context = None
            except:
                pass

# ==================== AI识别模块 ====================
class EncryptionClassifier:
    """加密算法分类器"""
    def __init__(self, model_path: Optional[str] = None):
        self._model = None
        self._mutex = QMutex()
        self._load_model(model_path)
    
    def _load_model(self, model_path: Optional[str]):
        """加载预训练模型"""
        with QMutexLocker(self._mutex):
            try:
                if not TF_AVAILABLE:
                    logging.warning("TensorFlow 不可用，使用模拟模式")
                    self._model = tf.keras.Sequential([
                        tf.keras.layers.InputLayer(input_shape=(AppConfig.MODEL_INPUT_SIZE, 1)),
                        tf.keras.layers.Conv1D(32, 3, activation='relu'),
                        tf.keras.layers.GlobalMaxPooling1D(),
                        tf.keras.layers.Dense(3, activation='softmax')
                    ])
                    return
                    self._model.compile(optimizer='adam', loss='categorical_crossentropy')
                    logging.warning("使用模拟模型 - 实际应用中应加载预训练模型")
                else:
                    if not os.path.exists(model_path):
                        raise ModelLoadError(f"模型文件不存在: {model_path}")
                    self._model = tf.keras.models.load_model(model_path)
                    logging.info(f"成功加载模型: {model_path}")
            except Exception as e:
                raise ModelLoadError(f"模型加载失败: {e}")
    
    @retry_on_failure()
    def predict(self, hex_data: bytes) -> EncryptionType:
        """识别加密算法类型"""
        with QMutexLocker(self._mutex):
            try:
                input_tensor = self._preprocess(hex_data)
                predictions = self._model.predict(input_tensor, verbose=0)
                
                # 模拟结果 - 实际应用中应根据模型输出解析
                return EncryptionType(
                    name="AES-128",
                    confidence=float(predictions[0][0]),
                    description="高级加密标准128位"
                )
            except Exception as e:
                raise ModelLoadError(f"预测失败: {e}")
    
    def _preprocess(self, data: bytes) -> np.ndarray:
        """预处理输入数据"""
        try:
            # 填充或截断数据到模型输入大小
            if len(data) < AppConfig.MODEL_INPUT_SIZE:
                padded = data + bytes([0] * (AppConfig.MODEL_INPUT_SIZE - len(data)))
            else:
                padded = data[:AppConfig.MODEL_INPUT_SIZE]
            
            # 转换为模型输入格式
            array = np.frombuffer(padded, dtype=np.uint8).astype(np.float32)
            array = array.reshape((1, AppConfig.MODEL_INPUT_SIZE, 1))
            return array
        except Exception as e:
            raise InvalidInputError(f"数据预处理失败: {e}")

# ==================== 卡片操作模块 ====================
class Proxmark3Manager:
    """Proxmark3设备管理类"""
    def __init__(self):
        self._mutex = QMutex()
        self._connected = False
        self._device_path = None
    
    def connect(self, device_path: str) -> bool:
        """连接Proxmark3设备"""
        with QMutexLocker(self._mutex):
            if self._connected:
                logging.warning("设备已连接")
                return True
            
            try:
                # 模拟连接 - 实际应用中应使用真实设备通信
                if not os.path.exists(device_path):
                    raise HardwareError(f"设备路径不存在: {device_path}")
                
                self._device_path = device_path
                self._connected = True
                logging.info(f"成功连接到设备: {device_path}")
                return True
            except Exception as e:
                raise HardwareError(f"连接设备失败: {e}")
    
    def disconnect(self):
        """断开Proxmark3设备"""
        with QMutexLocker(self._mutex):
            if not self._connected:
                return
            
            try:
                # 模拟断开
                self._connected = False
                self._device_path = None
                logging.info("设备已断开")
            except Exception as e:
                raise HardwareError(f"断开设备失败: {e}")
    
    @retry_on_failure()
    def read_card(self, timeout: float = AppConfig.DEFAULT_TIMEOUT) -> CardDump:
        """读取卡片数据"""
        with QMutexLocker(self._mutex):
            if not self._connected:
                raise HardwareError("设备未连接")
            
            try:
                start_time = time.time()
                
                # 模拟读取卡片数据
                time.sleep(0.1)  # 模拟读取延迟
                
                # 生成模拟数据
                uid = os.urandom(4)
                sectors = {
                    0: os.urandom(16),
                    1: os.urandom(16),
                    2: os.urandom(16)
                }
                
                return CardDump(
                    uid=uid,
                    sector_data=sectors,
                    timestamp=time.time(),
                    signal_quality=0.95  # 模拟信号质量
                )
            except Exception as e:
                raise HardwareError(f"读取卡片失败: {e}")
    
    def is_connected(self) -> bool:
        """检查设备是否已连接"""
        with QMutexLocker(self._mutex):
            return self._connected

# ==================== 渗透测试模块 ====================
class MifareAttacker:
    """Mifare卡攻击类"""
    def __init__(self, proxmark_manager: Proxmark3Manager):
        self._pm3 = proxmark_manager
        self._mutex = QMutex()
    
    @retry_on_failure()
    def nested_attack(self, key_known: int, timeout: float = AppConfig.DEFAULT_TIMEOUT) -> Optional[bytes]:
        """使用已知密钥破解其他扇区"""
        with QMutexLocker(self._mutex):
            try:
                if not self._pm3.is_connected():
                    raise HardwareError("Proxmark3设备未连接")
                
                # 模拟攻击过程
                logging.info(f"使用已知密钥 {key_known} 发起嵌套攻击...")
                time.sleep(0.5)  # 模拟攻击时间
                
                # 生成模拟数据
                dump_data = os.urandom(64)
                
                # 模拟保存到文件
                temp_file = tempfile.mktemp(suffix=".bin")
                with open(temp_file, 'wb') as f:
                    f.write(dump_data)
                
                logging.info(f"攻击完成，数据已保存到: {temp_file}")
                return dump_data
            except Exception as e:
                raise DecryptionError(f"嵌套攻击失败: {e}")
    
    @retry_on_failure()
    def parallel_attack(self, key_list: List[int], timeout: float = AppConfig.DEFAULT_TIMEOUT) -> Dict[int, Optional[bytes]]:
        """多线程密钥爆破"""
        results = {}
        
        def worker(key):
            try:
                result = self.nested_attack(key, timeout)
                return key, result
            except Exception as e:
                logging.warning(f"密钥 {key} 攻击失败: {e}")
                return key, None
        
        with ThreadPoolExecutor(max_workers=min(len(key_list), AppConfig.MAX_THREADS)) as executor:
            futures = {executor.submit(worker, key): key for key in key_list}
            
            for future in futures:
                try:
                    key, result = future.result(timeout=timeout)
                    results[key] = result
                except FutureTimeoutError:
                    logging.warning(f"密钥 {futures[future]} 攻击超时")
                    results[futures[future]] = None
                except Exception as e:
                    logging.error(f"并行攻击出错: {e}")
                    results[futures[future]] = None
        
        return results

# ==================== 数据可视化模块 ====================
class HexVisualizerBase(QWidget):
    """可视化组件基类"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._mutex = QMutex()
        self._current_plot = None
        self._init_fallback_ui()
    
    def _init_fallback_ui(self):
        """初始化兼容模式UI"""
        self.label = QLabel("3D可视化不可用\n(需要安装pyopengl和pyqtgraph.opengl)")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("""
            font-size: 14px; 
            color: #ccc; 
            background-color: #000;
            padding: 20px;
        """)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
    
    def plot_dump(self, data: bytes):
        """HEX数据可视化（兼容模式）"""
        with QMutexLocker(self._mutex):
            try:
                hex_str = binascii.hexlify(data[:256]).decode('ascii')
                formatted_hex = '\n'.join(
                    [f"{i:04x}: {hex_str[j:j+32]}" 
                     for i, j in enumerate(range(0, len(hex_str), 32))]
                )
                self.label.setText(f"HEX数据预览 (3D可视化不可用):\n{formatted_hex}")
            except Exception as e:
                self.label.setText(f"数据解析错误:\n{str(e)}")

if VISUALIZATION_AVAILABLE:
    class HexVisualizer(gl.GLViewWidget):
        """OpenGL可视化组件"""
        def __init__(self, parent=None):
            super().__init__(parent)
            self._mutex = QMutex()
            self._current_plot = None
            self.setBackgroundColor('k')
            self.setCameraPosition(distance=300)
        
        def plot_dump(self, data: bytes):
            """3D可视化HEX数据"""
            with QMutexLocker(self._mutex):
                try:
                    validated_data = validate_hex_data(data)
                    self.clear()
                    
                    points = np.array([(i, b, i*b % 256) 
                                     for i, b in enumerate(validated_data)])
                    colors = np.array([(b/255, (i % 256)/255, 0.8, 1.0) 
                                     for i, b in enumerate(validated_data)])
                    
                    scatter = gl.GLScatterPlotItem(
                        pos=points,
                        color=colors,
                        size=2,
                        pxMode=True
                    )
                    
                    self.addItem(scatter)
                    self._current_plot = scatter
                except Exception as e:
                    logging.error(f"3D可视化失败: {e}")
                    self.clear()
else:
    HexVisualizer = HexVisualizerBase  # 使用兼容模式实现

    
    def plot_dump(self, data: bytes):
        """将HEX数据映射为可视化"""
        with QMutexLocker(self._mutex):
            try:
                if not VISUALIZATION_AVAILABLE:
                    # 兼容模式下的简单显示
                    hex_str = binascii.hexlify(data[:128]).decode('ascii')
                    formatted_hex = '\n'.join([hex_str[i:i+32] for i in range(0, len(hex_str), 32)])
                    self.label.setText(f"HEX数据预览 (3D可视化不可用):\n{formatted_hex}")
                    return
                    
                # 以下是原有的OpenGL可视化代码
                validated_data = validate_hex_data(data)
                self.clear()
                
                points = np.array([(i, b, i*b % 256) for i, b in enumerate(validated_data)])
                colors = np.array([(b/255, (i % 256)/255, 0.8, 1.0) for i, b in enumerate(validated_data)])
                
                scatter = gl.GLScatterPlotItem(
                    pos=points,
                    color=colors,
                    size=2,
                    pxMode=True
                )
                
                self.addItem(scatter)
                self._current_plot = scatter
                
            except Exception as e:
                logging.error(f"可视化失败: {e}")
                if not VISUALIZATION_AVAILABLE:
                    self.label.setText(f"错误: {str(e)}")
                else:
                    self.label = QLabel(f"可视化错误: {str(e)}")
                    self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    layout = QVBoxLayout()
                    layout.addWidget(self.label)
                    self.setLayout(layout)



# ==================== 任务线程 ====================
class BreakThread(QThread):
    """密钥破解线程"""
    finished = pyqtSignal(KeySearchResult)
    progress = pyqtSignal(int, float)  # 当前尝试的密钥, 速度(keys/sec)
    error = pyqtSignal(str)
    
    def __init__(self, dumps: bytes, use_gpu: bool = True):
        super().__init__()
        self.dumps = dumps
        self.use_gpu = use_gpu
        self._stop_flag = False
        self._mutex = QMutex()
    
    def run(self):
        """线程主逻辑"""
        try:
            start_time = time.perf_counter()
            result = None
            
            if self.use_gpu:
                gpu_manager = GPUCryptoManager()
                result = gpu_manager.gpu_aes_break(self.dumps)
                elapsed = time.perf_counter() - start_time
                speed = AppConfig.GPU_BLOCK_SIZE * AppConfig.GPU_GRID_SIZE / elapsed
                
                self.finished.emit(KeySearchResult(
                    key=result,
                    attempts=AppConfig.GPU_BLOCK_SIZE * AppConfig.GPU_GRID_SIZE,
                    elapsed_time=elapsed,
                    speed=speed
                ))
            else:
                # CPU破解模拟 (仅用于演示)
                max_key = 1000000
                chunk_size = 1000
                for i in range(0, max_key, chunk_size):
                    if self._stop_flag:
                        break
                    
                    # 模拟CPU破解
                    time.sleep(0.01)  # 模拟计算时间
                    
                    # 随机"找到"密钥
                    if np.random.random() < 0.0001:
                        result = i + np.random.randint(0, chunk_size)
                        break
                    
                    # 更新进度
                    current_speed = chunk_size / 0.01
                    self.progress.emit(i + chunk_size, current_speed)
                
                elapsed = time.perf_counter() - start_time
                self.finished.emit(KeySearchResult(
                    key=result,
                    attempts=min(max_key, i + chunk_size),
                    elapsed_time=elapsed,
                    speed=min(max_key, i + chunk_size) / elapsed if elapsed > 0 else 0
                ))
        except Exception as e:
            self.error.emit(str(e))
    
    def stop(self):
        """停止线程"""
        with QMutexLocker(self._mutex):
            self._stop_flag = True

class CardReadThread(QThread):
    """卡片读取线程"""
    data_ready = pyqtSignal(CardDump)
    error = pyqtSignal(str)
    
    def __init__(self, pm3_manager: Proxmark3Manager):
        super().__init__()
        self._pm3 = pm3_manager
        self._stop_flag = False
        self._mutex = QMutex()
    
    def run(self):
        """线程主逻辑"""
        try:
            while not self._stop_flag:
                if not self._pm3.is_connected():
                    self.error.emit("设备未连接")
                    break
                
                try:
                    card_data = self._pm3.read_card()
                    self.data_ready.emit(card_data)
                    break  # 读取成功后退出
                except HardwareError as e:
                    logging.warning(f"读取失败: {e}")
                    time.sleep(1)
        except Exception as e:
            self.error.emit(str(e))
    
    def stop(self):
        """停止线程"""
        with QMutexLocker(self._mutex):
            self._stop_flag = True

# ==================== 主界面 ====================
class AnalyzerGUI(QMainWindow):
    """主界面类"""
    def __init__(self):
        super().__init__()

        # 添加系统兼容性检查
        compat_report = check_system_compatibility()
        if compat_report['issues']:
            msg = "系统兼容性警告:\n" + "\n".join(f"• {issue}" for issue in compat_report['issues'])
            QMessageBox.warning(self, "兼容性警告", msg)
            
        self._init_ui()
        self._init_components()
        self._setup_connections()
        self._load_config()

        if not TF_AVAILABLE:
            QMessageBox.warning(self, "警告", 
                "TensorFlow 不可用，AI 功能将使用模拟模式\n"
                "如需完整功能，请正确安装 TensorFlow")
        
        if not GPU_AVAILABLE:
            self.status_bar.showMessage("警告: GPU 加速不可用", 5000)
        
        if not VISUALIZATION_AVAILABLE:
            self.status_bar.showMessage("警告: 3D 可视化不可用", 5000)
        
        # 初始化状态
        self.current_dumps = None
        self.current_card_data = None
        self.break_thread = None
        self.read_thread = None
        
        # 设置窗口属性
        self.setWindowTitle(f"{AppConfig.APP_NAME} v{AppConfig.VERSION}")
        self.setMinimumSize(800, 600)
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")
    
    def _init_ui(self):
        """初始化UI界面"""
        # 主窗口布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 主水平布局
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # 分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(5, 5, 5, 5)
        control_layout.setSpacing(10)
        
        # 设备控制组
        device_group = QWidget()
        device_layout = QVBoxLayout(device_group)
        device_layout.setContentsMargins(0, 0, 0, 0)
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(["/dev/ttyACM0", "COM3", "模拟设备"])
        self.connect_btn = QPushButton("连接设备")
        self.disconnect_btn = QPushButton("断开连接")
        self.disconnect_btn.setEnabled(False)
        self.read_card_btn = QPushButton("读取卡片")
        self.read_card_btn.setEnabled(False)
        
        device_layout.addWidget(QLabel("设备:"))
        device_layout.addWidget(self.device_combo)
        device_layout.addWidget(self.connect_btn)
        device_layout.addWidget(self.disconnect_btn)
        device_layout.addWidget(self.read_card_btn)
        device_layout.addStretch()
        
        # 破解控制组
        break_group = QWidget()
        break_layout = QVBoxLayout(break_group)
        break_layout.setContentsMargins(0, 0, 0, 0)
        
        self.gpu_check = QCheckBox("使用GPU加速")
        self.gpu_check.setChecked(GPU_AVAILABLE)  # 根据GPU可用性设置初始状态
        self.gpu_check.setEnabled(GPU_AVAILABLE)  # 如果GPU不可用则禁用选项
        self.gpu_check.setChecked(True)
        self.start_break_btn = QPushButton("开始破解")
        self.start_break_btn.setEnabled(False)
        self.stop_break_btn = QPushButton("停止破解")
        self.stop_break_btn.setEnabled(False)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_label = QLabel("进度: 0%")
        self.speed_label = QLabel("速度: 0 keys/sec")
        
        break_layout.addWidget(QLabel("破解选项:"))
        break_layout.addWidget(self.gpu_check)
        break_layout.addWidget(self.start_break_btn)
        break_layout.addWidget(self.stop_break_btn)
        break_layout.addWidget(self.progress_bar)
        break_layout.addWidget(self.progress_label)
        break_layout.addWidget(self.speed_label)
        break_layout.addStretch()
        
        # 添加到控制面板
        control_layout.addWidget(device_group)
        control_layout.addWidget(break_group)
        control_layout.addStretch()
        
        # 右侧主显示区
        display_panel = QWidget()
        display_layout = QVBoxLayout(display_panel)
        display_layout.setContentsMargins(0, 0, 0, 0)
        
        # 标签页
        self.tab_widget = QTabWidget()
        
        # HEX数据标签页
        self.hex_view = QTextEdit()
        self.hex_view.setReadOnly(True)
        self.hex_view.setFont(QFont("Courier New", 10))
        
        # 3D可视化标签页
        self.visualizer = HexVisualizer()
        
        # 日志标签页
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        
        # 添加标签页
        self.tab_widget.addTab(self.hex_view, "HEX数据")
        self.tab_widget.addTab(self.visualizer, "3D可视化")
        self.tab_widget.addTab(self.log_view, "日志")
        
        display_layout.addWidget(self.tab_widget)
        
        # 添加到分割器
        splitter.addWidget(control_panel)
        splitter.addWidget(display_panel)
        splitter.setStretchFactor(1, 4)
        
        # 设置主布局
        main_layout.addWidget(splitter)
    
    def _init_components(self):
        """初始化组件"""
        # 创建Proxmark3管理器
        self.pm3_manager = Proxmark3Manager()
        
        # 创建加密分类器
        self.classifier = EncryptionClassifier()
        
        # 创建Mifare攻击器
        self.attacker = MifareAttacker(self.pm3_manager)
        
        # 创建GPU管理器
        self.gpu_manager = GPUCryptoManager()
    
    def _setup_connections(self):
        """设置信号槽连接"""
        # 设备控制
        self.connect_btn.clicked.connect(self._connect_device)
        self.disconnect_btn.clicked.connect(self._disconnect_device)
        self.read_card_btn.clicked.connect(self._read_card)
        
        # 破解控制
        self.start_break_btn.clicked.connect(self._start_break)
        self.stop_break_btn.clicked.connect(self._stop_break)
    
    def _load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(AppConfig.CONFIG_FILE):
                with open(AppConfig.CONFIG_FILE, 'r') as f:
                    config = yaml.safe_load(f)
                    if config and 'device' in config:
                        self.device_combo.setCurrentText(config['device'])
                    logging.info("配置加载成功")
        except Exception as e:
            logging.warning(f"加载配置失败: {e}")
    
    def _save_config(self):
        """保存配置文件"""
        try:
            config = {
                'device': self.device_combo.currentText(),
                'use_gpu': self.gpu_check.isChecked()
            }
            with open(AppConfig.CONFIG_FILE, 'w') as f:
                yaml.dump(config, f)
        except Exception as e:
            logging.warning(f"保存配置失败: {e}")
    
    def _connect_device(self):
        """连接设备"""
        device_path = self.device_combo.currentText()
        
        try:
            if self.pm3_manager.connect(device_path):
                self.connect_btn.setEnabled(False)
                self.disconnect_btn.setEnabled(True)
                self.read_card_btn.setEnabled(True)
                self.status_bar.showMessage(f"已连接到设备: {device_path}")
                logging.info(f"设备连接成功: {device_path}")
        except Exception as e:
            QMessageBox.critical(self, "连接失败", f"无法连接设备: {e}")
            logging.error(f"设备连接失败: {e}")
    
    def _disconnect_device(self):
        """断开设备"""
        try:
            self.pm3_manager.disconnect()
            self.connect_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(False)
            self.read_card_btn.setEnabled(False)
            self.status_bar.showMessage("设备已断开")
            logging.info("设备已断开")
        except Exception as e:
            QMessageBox.critical(self, "断开失败", f"断开设备时出错: {e}")
            logging.error(f"断开设备失败: {e}")
    
    def _read_card(self):
        """读取卡片数据"""
        if self.read_thread is not None:
            self.read_thread.stop()
            self.read_thread.wait()
        
        self.read_thread = CardReadThread(self.pm3_manager)
        self.read_thread.data_ready.connect(self._on_card_data_ready)
        self.read_thread.error.connect(self._on_card_read_error)
        self.read_thread.start()
        
        self.read_card_btn.setEnabled(False)
        self.status_bar.showMessage("正在读取卡片...")
    
    def _on_card_data_ready(self, card_data: CardDump):
        """卡片数据读取完成"""
        self.current_card_data = card_data
        self.current_dumps = b''.join(card_data.sector_data.values())
        
        # 显示HEX数据
        hex_str = binascii.hexlify(self.current_dumps).decode('ascii')
        formatted_hex = ' '.join(hex_str[i:i+2] for i in range(0, len(hex_str), 2))
        self.hex_view.setPlainText(formatted_hex)
        
        # 更新3D可视化
        try:
            self.visualizer.plot_dump(self.current_dumps)
        except Exception as e:
            logging.error(f"3D可视化失败: {e}")
        
        # 启用破解按钮
        self.start_break_btn.setEnabled(True)
        
        # 更新状态
        self.read_card_btn.setEnabled(True)
        self.status_bar.showMessage(f"卡片读取完成 - UID: {binascii.hexlify(card_data.uid).decode('ascii')}")
        
        # 自动分类加密类型
        try:
            enc_type = self.classifier.predict(self.current_dumps)
            QMessageBox.information(
                self, 
                "加密类型识别", 
                f"识别结果: {enc_type.name}\n置信度: {enc_type.confidence:.2%}\n描述: {enc_type.description}"
            )
        except Exception as e:
            logging.warning(f"加密类型识别失败: {e}")
    
    def _on_card_read_error(self, error_msg: str):
        """卡片读取错误"""
        self.read_card_btn.setEnabled(True)
        QMessageBox.critical(self, "读取错误", f"读取卡片失败: {error_msg}")
        self.status_bar.showMessage("读取卡片失败")
    
    def _start_break(self):
        """开始破解"""
        if self.current_dumps is None:
            QMessageBox.warning(self, "警告", "没有可用的卡片数据")
            return
        
        if self.break_thread is not None:
            self.break_thread.stop()
            self.break_thread.wait()
        
        self.break_thread = BreakThread(
            self.current_dumps,
            self.gpu_check.isChecked()
        )
        self.break_thread.finished.connect(self._on_break_finished)
        self.break_thread.progress.connect(self._on_break_progress)
        self.break_thread.error.connect(self._on_break_error)
        self.break_thread.start()
        
        self.start_break_btn.setEnabled(False)
        self.stop_break_btn.setEnabled(True)
        self.status_bar.showMessage("正在破解...")
    
    def _on_break_progress(self, current_key: int, speed: float):
        """破解进度更新"""
        self.progress_bar.setValue(min(100, int(current_key / 1000000 * 100)))
        self.progress_label.setText(f"进度: {self.progress_bar.value()}%")
        self.speed_label.setText(f"速度: {speed:,.0f} keys/sec")
    
    def _on_break_finished(self, result: KeySearchResult):
        """破解完成"""
        self.stop_break_btn.setEnabled(False)
        self.start_break_btn.setEnabled(True)
        
        if result.key is not None:
            QMessageBox.information(
                self,
                "破解成功",
                f"找到密钥: {result.key}\n"
                f"尝试次数: {result.attempts:,}\n"
                f"耗时: {result.elapsed_time:.2f}秒\n"
                f"平均速度: {result.speed:,.0f} keys/sec"
            )
            self.status_bar.showMessage(f"破解成功 - 找到密钥: {result.key}")
        else:
            QMessageBox.information(
                self,
                "破解完成",
                f"未找到有效密钥\n"
                f"尝试次数: {result.attempts:,}\n"
                f"耗时: {result.elapsed_time:.2f}秒\n"
                f"平均速度: {result.speed:,.0f} keys/sec"
            )
            self.status_bar.showMessage("破解完成 - 未找到有效密钥")
    
    def _on_break_error(self, error_msg: str):
        """破解错误"""
        self.stop_break_btn.setEnabled(False)
        self.start_break_btn.setEnabled(True)
        QMessageBox.critical(self, "破解错误", f"破解过程中出错: {error_msg}")
        self.status_bar.showMessage("破解失败")
    
    def _stop_break(self):
        """停止破解"""
        if self.break_thread is not None:
            self.break_thread.stop()
            self.break_thread.wait()
            self.stop_break_btn.setEnabled(False)
            self.start_break_btn.setEnabled(True)
            self.status_bar.showMessage("破解已停止")
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 停止所有线程
        if self.break_thread is not None:
            self.break_thread.stop()
            self.break_thread.wait()
        
        if self.read_thread is not None:
            self.read_thread.stop()
            self.read_thread.wait()
        
        # 断开设备
        if self.pm3_manager.is_connected():
            try:
                self.pm3_manager.disconnect()
            except:
                pass
        
        # 保存配置
        self._save_config()
        
        event.accept()

    def show_system_info(self):
        """显示系统信息对话框"""
        compat = check_system_compatibility()
        info = f"""
        ICCardAnalyzerApp 版本: {AppConfig.VERSION}
        操作系统: {compat['os']}
        Python 版本: {compat['python_version']}
        GPU 加速: {'可用' if compat['gpu_available'] else '不可用'}
        3D 可视化: {'可用' if compat['visualization_available'] else '不可用'}
        TensorFlow: {'可用' if compat['tensorflow_available'] else '不可用'}
        """
        
        if compat['issues']:
            info += "\n警告:\n" + "\n".join(f"• {issue}" for issue in compat['issues'])
        
        QMessageBox.information(self, "系统信息", info.strip())

# ==================== 应用程序入口 ====================
def main():
    """应用程序入口"""
    # Windows 特定检查
    if os.name == 'nt':
        if not TF_AVAILABLE and TF_ERROR and "DLL" in TF_ERROR:
            print("\nTensorFlow 安装建议:")
            print("1. 推荐使用: pip install tensorflow-cpu")
            print("2. 或安装完整版: pip install tensorflow")
            print("3. 确保安装了 Visual C++ Redistributable\n")
            print("\nWindows 系统兼容性说明:")
            print("- GPU 加速和 3D 可视化功能可能不可用")
            if not TF_AVAILABLE:
                print("- TensorFlow 不可用，使用模拟模式")
                print("  建议安装: pip install tensorflow-cpu")
            print()

    # 设置日志
    setup_logging()
    
    # 创建应用
    app = QApplication(sys.argv)
    
    # 设置样式
    app.setStyle('Fusion')
    
    # 创建并显示主窗口
    window = AnalyzerGUI()
    window.show()
    
    # 执行应用
    sys.exit(app.exec())

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    main()