"""torch 없이 torch.save(.pt) 텐서를 numpy 로 읽는다.

로컬(윈도우 노트북)에는 torch 가 없고, G0 검증은 CPU numpy 만으로 충분하다.
200MB 짜리 torch 를 까는 대신 zip+pickle 포맷을 직접 연다.
서버에서 torch 로 읽은 것과 값이 같아야 하므로, 읽은 뒤 shape/dtype/체크섬을 찍는다.
"""
import pickle
import zipfile

import numpy as np

_DTYPE = {
    "FloatStorage": np.float32,
    "HalfStorage": np.float16,
    "DoubleStorage": np.float64,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
    "BFloat16Storage": np.float16,  # 근사 — 이 저장소에서는 안 쓰임
}


class _Storage:
    def __init__(self, key, dtype, zf, prefix):
        self.key, self.dtype, self.zf, self.prefix = key, dtype, zf, prefix

    def array(self):
        with self.zf.open(f"{self.prefix}data/{self.key}") as f:
            return np.frombuffer(f.read(), dtype=self.dtype)


def _rebuild_tensor(storage, offset, size, stride, *_):
    arr = storage.array()
    if not size:
        return arr[offset]
    it = arr.dtype.itemsize
    return np.lib.stride_tricks.as_strided(
        arr[offset:], shape=tuple(size), strides=tuple(s * it for s in stride)
    )


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self._zf, self._prefix = zf, prefix

    def find_class(self, mod, name):
        if mod == "torch._utils" and name.startswith("_rebuild_tensor"):
            return _rebuild_tensor
        if mod == "torch" and name in _DTYPE:
            return _DTYPE[name]
        if mod == "collections" and name == "OrderedDict":
            return dict
        return super().find_class(mod, name)

    def persistent_load(self, pid):
        # ('storage', <storage_type>, key, location, numel)
        _, stype, key, _loc, _numel = pid
        return _Storage(key, np.dtype(stype), self._zf, self._prefix)


def load(path):
    """.pt 를 읽어 numpy 배열(또는 dict/list) 로 돌려준다."""
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        prefix = pkl[: -len("data.pkl")]
        with zf.open(pkl) as f:
            obj = _Unpickler(f, zf, prefix).load()
        # as_strided 는 zip 핸들이 닫히기 전에 실체화해야 한다
        return _materialize(obj)


def _materialize(o):
    if isinstance(o, np.ndarray):
        return np.ascontiguousarray(o)
    if isinstance(o, dict):
        return {k: _materialize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_materialize(v) for v in o)
    return o


if __name__ == "__main__":
    import sys

    a = load(sys.argv[1])
    if isinstance(a, np.ndarray):
        print(f"shape={a.shape} dtype={a.dtype}")
        print(f"min={a.min():.6f} max={a.max():.6f} mean={a.mean():.6f} sum={a.sum():.4f}")
    else:
        print(type(a), list(a)[:10] if hasattr(a, "__iter__") else a)
