import redis
import numpy as np
import struct


class RedisPool:
    def __init__(self, host, password, port, max_connections=10):
        # 创建 Redis 连接池
        self.pool = redis.ConnectionPool(host=host, port=port, password=password, max_connections=max_connections)

    def get_redis_conn(self):
        # 获取 Redis 连接
        redis_conn = redis.StrictRedis(connection_pool=self.pool)
        try:
            # 检查连接是否可用
            redis_conn.ping()
        except Exception:
            # 如果连接不可用，则重建连接
            redis_conn.connection_pool.disconnect()
            redis_conn = redis.StrictRedis(connection_pool=self.pool)
        return redis_conn


class RedisService:
    def __init__(self, host, password, port, max_connections=10):
        # 创建 Redis 连接池对象
        self.pool = RedisPool(host=host, port=port, password=password, max_connections=max_connections)
        # 获取 Redis 连接
        self.redis_conn = self.pool.get_redis_conn()

    def set(self, **kwargs):
        for key, value in kwargs.items():
            self.redis_conn.__setitem__(key, value)

    def push(self, key, value):
        self.redis_conn.lpush(key, value)

    def pop(self, key):
        value = self.redis_conn.rpop(key)
        self.redis_conn.lrem(key, 0, value)
        return value

    def list_get_index(self, key, index):
        return self.redis_conn.lindex(key, index)

    def llen(self, key):
        return self.redis_conn.llen(key)

    def set_add(self, key, *value):
        self.redis_conn.sadd(key, *value)

    def set_member_exists(self, key, value):
        return self.redis_conn.sismember(key, value)

    def exitst(self, key):
        return self.redis_conn.exists(key)

    def __setitem__(self, key, value):
        if self.redis_conn.exists(key):
            self.redis_conn.delete(key)
        if isinstance(value, str):
            self.redis_conn.set(key, value)
        elif isinstance(value, dict):
            self.redis_conn.hmset(key, value)
        elif isinstance(value, list):
            self.redis_conn.rpush(key, *value)
        else:
            self.redis_conn.set(key, value)

    def __getitem__(self, key):
        key_type = self.redis_conn.type(key)
        if key_type == b'none':
            return None
        elif key_type == b'list':
            return self.redis_conn.lrange(key, 0, -1)
        elif key_type == b'string':
            return self.redis_conn.get(key)
        elif key_type == b'hash':
            return self.redis_conn.hgetall(key)
        else:
            print(f"Key {key} type {key_type} not support")
            return None


def encode_wb(wb_np: np.ndarray,
              dtype: np.dtype,
              shape: tuple,
              ) -> bytes:
    # 将numpy数组转换为bytes,头部为shape和dtype,变长编码
    wb_bytes = struct.pack('i', len(shape))
    for i in shape:
        wb_bytes += struct.pack('i', i)

    wb_bytes += struct.pack('i', len(dtype.name))
    wb_bytes += dtype.name.encode('utf-8')
    wb_bytes += wb_np.tobytes()
    return wb_bytes


def decode_wb(wb_bytes: bytes) -> np.ndarray:
    # 将上文编码的bytes转换回numpy数组
    shape_len = struct.unpack('i', wb_bytes[:4])[0]
    shape = []
    for i in range(shape_len):
        shape.append(struct.unpack('i', wb_bytes[4 + i * 4: 8 + i * 4])[0])
    dtype_len = struct.unpack('i', wb_bytes[4 + shape_len * 4: 8 + shape_len * 4])[0]
    dtype = np.dtype(struct.unpack(f'{dtype_len}s', wb_bytes[8 + shape_len * 4: 8 + shape_len * 4 + dtype_len])[0])
    wb_np = np.frombuffer(wb_bytes[8 + shape_len * 4 + dtype_len:], dtype=dtype)
    wb_np = wb_np.reshape(shape)
    return wb_np
